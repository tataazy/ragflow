#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""SAG extraction executor.

Entry point invoked by task_executor when task_type == 'sag_extract' or 'sag_rebuild'.
Implements the chunk-level event-entity extraction pipeline with
concurrency control, checkpoint/resume, progress tracking, and vector indexing.
"""

import asyncio
import logging
from datetime import datetime
from typing import Callable

from common import settings
from common.misc_utils import thread_pool_exec
from rag.nlp import search
from rag.sag.config import KB_DEFAULTS, get_entity_types, get_sag_system_config
from rag.sag.models import SagEvent, SagEntity, SagEventEntity, SagExtractCheckpoint

logger = logging.getLogger(__name__)

# Retry intervals in seconds (index 0 = first retry delay, index 1 = second)
_RETRY_DELAYS = [5, 15]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


async def run_sag_extract(
    task_context,
    chat_model,
    embedding_model,
    progress_cb: Callable,
) -> None:
    """Execute SAG extraction for a single document.

    Called by TaskHandler when task_type == 'sag_extract'.
    Reads chunks from doc_store, runs LLM extraction per chunk with
    concurrency control, writes events/entities to MySQL, indexes event
    vectors, and maintains checkpoint for pause/resume.

    Args:
        task_context: TaskContext with doc_id, kb_id, tenant_id, etc.
        chat_model: LLMBundle for extraction LLM calls.
        embedding_model: LLMBundle for generating event embeddings.
        progress_cb: Callback for progress updates (prog=, msg=).
    """
    ctx = task_context
    await _execute_doc_extraction(
        doc_id=ctx.doc_id,
        kb_id=ctx.kb_id,
        tenant_id=ctx.tenant_id,
        task_id=ctx.id,
        ctx=ctx,
        chat_model=chat_model,
        embedding_model=embedding_model,
        progress_cb=progress_cb,
    )


async def run_sag_rebuild(
    task_context,
    chat_model,
    embedding_model,
    progress_cb: Callable,
) -> None:
    """Execute SAG full rebuild for a knowledge base.

    Called by TaskHandler when task_type == 'sag_rebuild'.
    Cleans all existing SAG data, then re-extracts each document sequentially
    within this task. The rebuild task is marked completed only after all
    documents have been processed.

    Args:
        task_context: TaskContext with kb_id, tenant_id, doc_ids, etc.
        chat_model: LLMBundle for extraction LLM calls.
        embedding_model: LLMBundle for generating event embeddings.
        progress_cb: Callback for progress updates (prog=, msg=).
    """
    from rag.sag.cleanup import cleanup_sag_data_for_kb

    ctx = task_context
    kb_id = ctx.kb_id
    tenant_id = ctx.tenant_id
    task_id = ctx.id
    doc_ids = ctx.doc_ids or []

    progress_cb(prog=0.0, msg="[SAG] Rebuild started: cleaning old data...")

    # 1. Clean all existing SAG data for this KB
    await thread_pool_exec(cleanup_sag_data_for_kb, kb_id, tenant_id)
    progress_cb(prog=0.02, msg="[SAG] Old data cleaned, starting re-extraction...")

    if not doc_ids:
        progress_cb(prog=1.0, msg="[SAG] Rebuild complete: no documents to process")
        return

    # 2. Execute extraction for each document sequentially
    total_docs = len(doc_ids)
    for i, doc_id in enumerate(doc_ids):
        if ctx.has_canceled_func(task_id):
            progress_cb(prog=(i + 1) / total_docs, msg="[SAG] Rebuild cancelled")
            return

        doc_prog_base = 0.02 + 0.98 * (i / total_docs)
        doc_prog_span = 0.98 / total_docs

        def _doc_progress(prog=None, msg="", _base=doc_prog_base, _span=doc_prog_span):
            overall = _base + _span * (prog or 0.0)
            progress_cb(prog=overall, msg=msg)

        await _execute_doc_extraction(
            doc_id=doc_id,
            kb_id=kb_id,
            tenant_id=tenant_id,
            task_id=task_id,
            ctx=ctx,
            chat_model=chat_model,
            embedding_model=embedding_model,
            progress_cb=_doc_progress,
        )

        progress_cb(
            prog=0.02 + 0.98 * ((i + 1) / total_docs),
            msg=f"[SAG] Rebuild progress: doc {i + 1}/{total_docs} done",
        )

    progress_cb(prog=1.0, msg=f"[SAG] Rebuild completed: {total_docs} documents re-extracted")
    logger.info("[SAG] Rebuild task %s completed for kb %s (%d docs)", task_id, kb_id, total_docs)


# ---------------------------------------------------------------------------
# Core per-document extraction logic
# ---------------------------------------------------------------------------


async def _execute_doc_extraction(
    doc_id: str,
    kb_id: str,
    tenant_id: str,
    task_id: str,
    ctx,
    chat_model,
    embedding_model,
    progress_cb: Callable,
) -> None:
    """Core extraction logic for a single document.

    Shared by both run_sag_extract (single doc task) and run_sag_rebuild
    (sequential per-doc within rebuild task). Handles checkpoint/resume,
    concurrency control, retry, and vector indexing.

    Args:
        doc_id: Document to extract from.
        kb_id: Knowledge base ID.
        tenant_id: Tenant ID.
        task_id: Task ID (for checkpoint and cancellation).
        ctx: TaskContext (provides kb_parser_config, has_canceled_func, chat_limiter).
        chat_model: LLMBundle for extraction LLM calls.
        embedding_model: LLMBundle for event embeddings.
        progress_cb: Progress callback (prog=, msg=).
    """
    # Load KB-level SAG config
    sag_config = (ctx.kb_parser_config or {}).get("sag", {})
    concurrency = sag_config.get("extract_concurrency", KB_DEFAULTS["extract_concurrency"])
    sys_config = get_sag_system_config()
    max_retries = sys_config.get("extract_max_retries", 2)
    entity_types = get_entity_types()

    # Create or resume checkpoint
    checkpoint = _get_or_create_checkpoint(kb_id, doc_id, task_id)
    processed_set = set(checkpoint.processed_chunk_ids or [])

    # Check if checkpoint was cancelled externally before we start
    if checkpoint.status == "cancelled":
        progress_cb(prog=1.0, msg="[SAG] Task was cancelled before execution")
        return

    # Load all chunks for this document from doc_store
    all_chunks = await _load_doc_chunks(doc_id, tenant_id, kb_id)
    if not all_chunks:
        _finish_checkpoint(checkpoint, status="completed")
        progress_cb(prog=1.0, msg="[SAG] No chunks found for document, extraction skipped")
        return

    # Filter to pending chunks only
    pending_chunks = [(cid, text) for cid, text in all_chunks if cid not in processed_set]
    total = len(all_chunks)
    already_done = total - len(pending_chunks)

    if not pending_chunks:
        _finish_checkpoint(checkpoint, status="completed")
        progress_cb(prog=1.0, msg="[SAG] All chunks already processed, extraction complete")
        return

    progress_cb(
        prog=already_done / total,
        msg=f"[SAG] Starting extraction: {len(pending_chunks)} chunks pending, {already_done} already done",
    )

    # Reset checkpoint status to running
    checkpoint.status = "running"
    checkpoint.save()

    # Concurrency control
    semaphore = asyncio.Semaphore(concurrency)
    processed_count = already_done
    new_event_ids: list[int] = list(checkpoint.event_ids or [])
    lock = asyncio.Lock()

    async def _worker(chunk_id: str, chunk_text: str) -> None:
        nonlocal processed_count
        async with semaphore:
            # Check task-level cancellation
            if ctx.has_canceled_func(task_id):
                return
            # Check checkpoint-level pause/cancel
            if _is_checkpoint_halted(kb_id, doc_id, task_id):
                return

            # Extract with retry
            events_data, error_msg = await _extract_with_retry(
                chunk_text, entity_types, ctx, chat_model, max_retries
            )

            if events_data is None:
                # All retries exhausted — record failure, continue others
                logger.warning("[SAG] Chunk %s failed after %d retries: %s", chunk_id, max_retries, error_msg)
                async with lock:
                    processed_count += 1
                    _update_checkpoint_incremental(
                        checkpoint, chunk_id, [], processed_count, total,
                        status_note=f"chunk {chunk_id} failed",
                    )
                    progress_cb(
                        prog=processed_count / total,
                        msg=f"[SAG] Chunk {chunk_id[:8]}... failed: {error_msg[:100] if error_msg else 'unknown'}, {processed_count}/{total}",
                    )
                return

            # Persist events/entities to DB
            event_ids = await thread_pool_exec(
                _persist_chunk_events, events_data, chunk_id, kb_id, doc_id
            )

            # Update counters and checkpoint
            async with lock:
                processed_count += 1
                new_event_ids.extend(event_ids)
                _update_checkpoint_incremental(
                    checkpoint, chunk_id, event_ids, processed_count, total
                )
                progress_cb(
                    prog=processed_count / total,
                    msg=f"[SAG] Extracted {len(event_ids)} events from chunk {chunk_id[:8]}..., {processed_count}/{total}",
                )

    # Run all workers concurrently (bounded by semaphore)
    workers = [_worker(cid, text) for cid, text in pending_chunks]
    await asyncio.gather(*workers, return_exceptions=True)

    # Post-run status check - re-fetch checkpoint from DB
    checkpoint = _get_or_create_checkpoint(kb_id, doc_id, task_id)
    if checkpoint.status in ("paused", "cancelled"):
        progress_cb(prog=processed_count / total, msg=f"[SAG] Task {checkpoint.status}")
        return

    if ctx.has_canceled_func(task_id):
        checkpoint.status = "cancelled"
        checkpoint.save()
        progress_cb(prog=processed_count / total, msg="[SAG] Task cancelled")
        return

    # Index event vectors to doc_store
    if new_event_ids:
        await _index_event_vectors(new_event_ids, kb_id, doc_id, tenant_id, embedding_model)

    # Mark completed
    _finish_checkpoint(checkpoint, status="completed")
    progress_cb(prog=1.0, msg=f"[SAG] Extraction completed: {len(new_event_ids)} events from {total} chunks")
    logger.info(
        "[SAG] Extraction done for doc %s in kb %s: %d events, %d chunks",
        doc_id, kb_id, len(new_event_ids), total,
    )


# ---------------------------------------------------------------------------
# Chunk loading
# ---------------------------------------------------------------------------


async def _load_doc_chunks(doc_id: str, tenant_id: str, kb_id: str) -> list[tuple[str, str]]:
    """Load all available chunks for a document from doc_store.

    Returns:
        List of (chunk_id, chunk_text) tuples, ordered by position.
    """
    index_nm = search.index_name(tenant_id)
    if not settings.docStoreConn.index_exist(index_nm, kb_id):
        return []

    select_fields = ["id", "content_with_weight"]
    chunks: list[tuple[str, str]] = []
    offset = 0
    PAGE = 500

    while True:
        try:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                select_fields,
                [],
                {"doc_id": [doc_id], "available_int": 1},
                [],
                _order_by_position(),
                offset,
                PAGE,
                index_nm,
                [kb_id],
            )
            field_map = settings.docStoreConn.get_fields(res, select_fields)
        except Exception:
            logger.exception("[SAG] Failed to load chunks for doc %s", doc_id)
            break

        if not field_map:
            break

        for row_id, row in field_map.items():
            text = row.get("content_with_weight") or ""
            if text.strip():
                chunks.append((str(row_id), text))

        if len(field_map) < PAGE:
            break
        offset += PAGE

    return chunks


def _order_by_position():
    """Build an OrderByExpr for chunk position ordering."""
    from common.doc_store.doc_store_base import OrderByExpr

    order = OrderByExpr()
    order.asc("page_num_int")
    order.asc("top_int")
    return order


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


def _get_or_create_checkpoint(kb_id: str, doc_id: str, task_id: str) -> SagExtractCheckpoint:
    """Get existing checkpoint or create a new one."""
    try:
        cp = SagExtractCheckpoint.get(
            SagExtractCheckpoint.kb_id == kb_id,
            SagExtractCheckpoint.doc_id == doc_id,
            SagExtractCheckpoint.task_id == task_id,
        )
        return cp
    except SagExtractCheckpoint.DoesNotExist:
        pass

    # Generate a unique ID
    cp = SagExtractCheckpoint.create(
        id=_generate_id(),
        kb_id=kb_id,
        doc_id=doc_id,
        task_id=task_id,
        processed_chunk_ids=[],
        event_ids=[],
        event_count=0,
        token_usage=0,
        status="running",
    )
    return cp


def _is_checkpoint_halted(kb_id: str, doc_id: str, task_id: str) -> bool:
    """Check if checkpoint has been paused or cancelled externally."""
    try:
        cp = SagExtractCheckpoint.get(
            SagExtractCheckpoint.kb_id == kb_id,
            SagExtractCheckpoint.doc_id == doc_id,
            SagExtractCheckpoint.task_id == task_id,
        )
        return cp.status in ("paused", "cancelled")
    except SagExtractCheckpoint.DoesNotExist:
        return False


def _update_checkpoint_incremental(
    checkpoint: SagExtractCheckpoint,
    chunk_id: str,
    event_ids: list[int],
    processed_count: int,
    total_count: int,
    status_note: str = "",
) -> None:
    """Update checkpoint after processing one chunk (called under lock)."""
    processed = list(checkpoint.processed_chunk_ids or [])
    processed.append(chunk_id)
    checkpoint.processed_chunk_ids = processed

    if event_ids:
        eids = list(checkpoint.event_ids or [])
        eids.extend(event_ids)
        checkpoint.event_ids = eids

    checkpoint.event_count = len(checkpoint.event_ids or [])
    checkpoint.save()


def _finish_checkpoint(checkpoint: SagExtractCheckpoint, status: str = "completed") -> None:
    """Mark checkpoint with a terminal status."""
    checkpoint.status = status
    checkpoint.save()


# ---------------------------------------------------------------------------
# LLM extraction with retry
# ---------------------------------------------------------------------------


async def _extract_with_retry(
    chunk_text: str,
    entity_types: list[str],
    ctx,
    chat_model,
    max_retries: int,
) -> tuple[list[dict] | None, str | None]:
    """Call LLM extraction with retry logic.

    Returns:
        Tuple of (event list on success or None, error message or None).
    """
    last_error = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            delay = _RETRY_DELAYS[min(attempt - 1, len(_RETRY_DELAYS) - 1)]
            logger.info("[SAG] Retry attempt %d after %ds delay", attempt, delay)
            await asyncio.sleep(delay)
        try:
            result = await extract_events_from_chunk(chunk_text, entity_types, ctx, chat_model)
            return result, None
        except Exception as e:
            last_error = e
            logger.warning("[SAG] Extraction attempt %d failed: %s", attempt + 1, e)

    error_msg = str(last_error) if last_error else "Unknown error"
    logger.error("[SAG] All %d attempts failed. Last error: %s", max_retries + 1, error_msg)
    return None, error_msg


async def extract_events_from_chunk(
    chunk_text: str,
    entity_types: list[str],
    ctx,
    chat_model,
) -> list[dict]:
    """Extract events and entities from a single chunk via LLM.

    Builds the extraction prompt, calls the LLM, parses the JSON response,
    and validates/normalizes entity types against the configured list.

    Args:
        chunk_text: The chunk text to extract from.
        entity_types: Configured entity type list.
        ctx: TaskContext (provides chat_limiter).
        chat_model: LLMBundle for the extraction call.

    Returns:
        List of event dicts (usually 1 element):
        [{
            "title": str,
            "summary": str,
            "content": str,
            "category": str | None,
            "start_time": str | None,
            "entities": [{"name": str, "type": str, "description": str | None}],
        }]
        Empty list if chunk is skipped (too short or null response).

    Raises:
        Exception: If LLM call fails or response cannot be parsed.
    """
    import json
    import re

    from rag.sag.prompts.extract import build_extract_messages

    # Skip chunks that are too short (spec: < 20 chars)
    if len(chunk_text.strip()) < 20:
        return []

    # Build prompts
    system_prompt, history = build_extract_messages(chunk_text, entity_types)

    # Call LLM with rate limiting
    async with ctx.chat_limiter:
        response = await asyncio.wait_for(
            chat_model.async_chat(system_prompt, history, {"temperature": 0.1}),
            timeout=120,
        )

    if not response or not isinstance(response, str):
        raise ValueError(f"LLM returned empty or invalid response: {type(response)}")

    # Check for error marker
    if "**ERROR**" in response:
        raise RuntimeError(f"LLM error: {response[:200]}")

    # Parse JSON response
    try:
        event_data = _parse_extraction_response(response)
    except Exception as e:
        logger.error("[SAG] Failed to parse response: %s\nResponse: %s", e, response[:500])
        raise
    if event_data is None:
        # LLM returned null — chunk has no extractable content
        logger.info("[SAG] LLM returned null for chunk (no extractable content). Response: %s", response[:200])
        return []

    # Validate and normalize entities
    event_data["entities"] = _normalize_entities(
        event_data.get("entities", []), entity_types
    )

    return [event_data]


def _parse_extraction_response(response: str) -> dict | None:
    """Parse the LLM extraction response into an event dict.

    Handles common LLM output quirks: markdown code fences, leading/trailing
    whitespace, and think tags.

    Returns:
        Parsed event dict, or None if the response is a null/skip indicator.

    Raises:
        ValueError: If the response cannot be parsed as valid JSON.
    """
    import json
    import re

    text = response.strip()

    # Remove think tags (some models emit <think>...</think>)
    text = re.sub(r"^.*</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (```json or ```) and last line (```)
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    # Handle null response (chunk skipped)
    if text.lower() in ("null", "none", ""):
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {text[:500]}") from e

    if data is None:
        return None

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")

    # Validate required fields
    title = data.get("title")
    if not title or not isinstance(title, str):
        raise ValueError(f"Missing or invalid 'title' field in response")

    # Normalize to expected schema
    return {
        "title": title[:255],
        "summary": (data.get("summary") or "")[:2000] or None,
        "content": data.get("content") or data.get("summary") or title,
        "category": (data.get("category") or "")[:64] or None,
        "start_time": data.get("start_time"),
        "entities": data.get("entities", []),
    }


# Entity type alias mapping for fuzzy matching (English → Chinese)
_ENTITY_TYPE_ALIASES: dict[str, str] = {
    "time": "时间", "date": "时间", "temporal": "时间",
    "location": "地点", "place": "地点", "geo": "地点", "region": "地点",
    "person": "人物", "people": "人物", "human": "人物",
    "organization": "组织", "org": "组织", "company": "组织", "institution": "组织",
    "group": "群体", "community": "群体", "population": "群体",
    "topic": "主题", "theme": "主题", "subject": "主题",
    "work": "作品", "artwork": "作品", "publication": "作品",
    "product": "产品", "item": "产品", "goods": "产品",
    "action": "动作", "activity": "动作", "event": "动作",
    "metric": "指标", "indicator": "指标", "measure": "指标",
    "tag": "标签", "label": "标签", "keyword": "标签",
}


def _normalize_entities(
    entities: list,
    valid_types: list[str],
) -> list[dict]:
    """Validate and normalize entity types against the configured list.

    - If type is valid, keep as-is.
    - If type matches an alias, map to the canonical Chinese type.
    - If type cannot be matched, log warning and discard the entity.
    - Also handles the common LLM error of putting entity type as field name.

    Args:
        entities: Raw entity list from LLM response.
        valid_types: List of valid entity type strings.

    Returns:
        Cleaned list of entity dicts with validated types.
    """
    if not entities or not isinstance(entities, list):
        return []

    valid_set = set(valid_types)
    result = []

    for ent in entities:
        if not isinstance(ent, dict):
            continue

        name = ent.get("name")
        if not name or not isinstance(name, str) or not name.strip():
            continue

        ent_type = ent.get("type")

        # Handle common LLM error: type used as field name
        # e.g., {"location": "中东", "name": "中东", "description": "地区"}
        if not ent_type:
            extra_keys = [k for k in ent if k not in ("name", "description")]
            if len(extra_keys) == 1:
                alias_key = extra_keys[0]
                if alias_key in _ENTITY_TYPE_ALIASES:
                    ent_type = _ENTITY_TYPE_ALIASES[alias_key]
                elif alias_key in valid_set:
                    ent_type = alias_key

        if not ent_type or not isinstance(ent_type, str):
            logger.warning("[SAG] Entity '%s' has no valid type, discarding", name)
            continue

        ent_type = ent_type.strip()

        # Direct match
        if ent_type in valid_set:
            result.append({
                "name": name.strip()[:255],
                "type": ent_type,
                "description": (ent.get("description") or "")[:512] or None,
            })
            continue

        # Fuzzy match via alias table
        alias_match = _ENTITY_TYPE_ALIASES.get(ent_type.lower())
        if alias_match and alias_match in valid_set:
            result.append({
                "name": name.strip()[:255],
                "type": alias_match,
                "description": (ent.get("description") or "")[:512] or None,
            })
            continue

        # Cannot match — discard
        logger.warning(
            "[SAG] Entity '%s' has unknown type '%s', discarding", name, ent_type
        )

    return result


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------


def _generate_id() -> int:
    """Generate a unique numeric ID for SAG tables (fits MySQL signed BIGINT)."""
    import uuid
    return uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF  # Mask to 63 bits for signed BIGINT


def _persist_chunk_events(
    events_data: list[dict],
    chunk_id: str,
    kb_id: str,
    doc_id: str,
) -> list[int]:
    """Persist extracted events and entities to the database.

    Creates SagEvent rows, deduplicates SagEntity rows (upsert by
    kb_id+entity_name+entity_type), and links them via SagEventEntity.

    Args:
        events_data: List of event dicts from extract_events_from_chunk.
        chunk_id: Source chunk ID.
        kb_id: Knowledge base ID.
        doc_id: Document ID.

    Returns:
        List of created event IDs.
    """
    event_ids: list[int] = []

    for evt in events_data:
        # Create event row
        event = SagEvent.create(
            id=_generate_id(),
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_id=chunk_id,
            title=evt.get("title", "")[:255],
            summary=evt.get("summary"),
            content=evt.get("content", ""),
            category=evt.get("category"),
            start_time=evt.get("start_time"),
            status="completed",
        )
        event_ids.append(event.id)

        # Process entities
        for ent_data in evt.get("entities", []):
            entity = _upsert_entity(
                kb_id=kb_id,
                entity_name=ent_data.get("name", "")[:255],
                entity_type=ent_data.get("type", "")[:32],
                description=ent_data.get("description"),
            )
            if entity:
                # Create association (skip if duplicate)
                try:
                    SagEventEntity.create(
                        id=_generate_id(),
                        event_id=event.id,
                        entity_id=entity.id,
                        weight=ent_data.get("weight", 1.0),
                        description=(ent_data.get("description") or "")[:512] or None,
                    )
                except Exception:
                    # Duplicate (event_id, entity_id) — skip
                    pass

    return event_ids


def _upsert_entity(kb_id: str, entity_name: str, entity_type: str, description: str | None):
    """Get or create an entity, incrementing heat on existing."""
    if not entity_name or not entity_type:
        return None

    try:
        entity = SagEntity.get(
            SagEntity.kb_id == kb_id,
            SagEntity.entity_name == entity_name,
            SagEntity.entity_type == entity_type,
        )
        # Increment heat
        SagEntity.update(heat=SagEntity.heat + 1).where(
            SagEntity.id == entity.id
        ).execute()
        return entity
    except SagEntity.DoesNotExist:
        return SagEntity.create(
            id=_generate_id(),
            kb_id=kb_id,
            entity_name=entity_name,
            entity_type=entity_type,
            description=description,
            heat=1,
        )


# ---------------------------------------------------------------------------
# Vector indexing
# ---------------------------------------------------------------------------


async def _index_event_vectors(
    event_ids: list[int],
    kb_id: str,
    doc_id: str,
    tenant_id: str,
    embedding_model,
) -> None:
    """Generate embeddings for events and insert into doc_store.

    Each event is indexed with sag_kwd='event' marker for later
    retrieval and cleanup.
    """
    if not event_ids:
        return

    # Load events from DB
    events = list(SagEvent.select().where(SagEvent.id.in_(event_ids)))
    if not events:
        return

    # Prepare texts for embedding (title + summary)
    texts = []
    valid_events = []
    for evt in events:
        text = evt.title
        if evt.summary:
            text = f"{evt.title}\n{evt.summary}"
        if text.strip():
            texts.append(text)
            valid_events.append(evt)

    if not texts:
        return

    # Generate embeddings
    try:
        vectors, _ = await thread_pool_exec(embedding_model.encode, texts)
    except Exception:
        logger.exception("[SAG] Failed to generate event embeddings for doc %s", doc_id)
        return

    if vectors is None or len(vectors) == 0:
        return

    # Determine vector field name
    vctr_nm = "q_%d_vec" % len(vectors[0])
    index_nm = search.index_name(tenant_id)

    # Build doc_store rows
    rows = []
    for evt, vec in zip(valid_events, vectors):
        row = {
            "id": f"sag_event_{evt.id}",
            "doc_id": doc_id,
            "kb_id": kb_id,
            "sag_kwd": "event",
            "sag_event_id": evt.id,
            "content_with_weight": f"{evt.title}\n{evt.summary or ''}",
            "content_ltks": evt.title,
            "available_int": 1,
            "create_timestamp_flt": datetime.now().timestamp(),
            vctr_nm: vec.tolist() if hasattr(vec, "tolist") else list(vec),
        }
        rows.append(row)

    # Insert to doc_store
    try:
        await thread_pool_exec(settings.docStoreConn.insert, rows, index_nm, kb_id)
        logger.info("[SAG] Indexed %d event vectors for doc %s in kb %s", len(rows), doc_id, kb_id)
    except Exception:
        logger.exception("[SAG] Failed to index event vectors for doc %s", doc_id)
