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
"""SAG Retriever.

Provides structured retrieval over the event-entity graph built by the
SAG extraction pipeline. Supports two strategies:
- vector: fast path using event vector similarity only
- multi: precise path with entity recall + event vector + lexical + multi-hop expansion

Returns chunks in the same format as RAGFlow's Dealer.retrieval for seamless
integration with the existing reranking and answer generation pipeline.
"""

import asyncio
import logging
import re

from common.misc_utils import thread_pool_exec
from rag.nlp import search
from rag.sag.config import get_sag_system_config
from rag.sag.models import SagEvent, SagEntity, SagEventEntity

logger = logging.getLogger(__name__)

# Query noise words to strip before lexical matching
_QUERY_NOISE = (
    "知识库", "资料库", "资料中", "文档中", "告诉我", "帮我查",
    "搜索", "查询", "请问", "关于", "最新", "最近", "动态",
    "消息", "新闻", "内容", "资料", "一下", "是什么", "有哪些", "有什么",
)


class SAGRetriever:
    """Structured retrieval engine over SAG event-entity data.

    Initialized once at startup via common/settings.py and shared across
    all retrieval requests. Thread-safe by design (stateless per call).
    """

    def __init__(self, doc_store_conn):
        """Initialize SAGRetriever.

        Args:
            doc_store_conn: The shared doc store connection (ES/Infinity/OpenSearch).
        """
        self.doc_store_conn = doc_store_conn

    async def retrieval(
        self,
        question: str,
        tenant_ids: list[str],
        kb_ids: list[str],
        emb_mdl,
        llm_mdl=None,
        top_k: int = 10,
        hop_num: int = 1,
        strategy: str = "multi",
    ) -> dict:
        """Execute SAG structured retrieval.

        Args:
            question: User query text.
            tenant_ids: Tenant IDs for index resolution.
            kb_ids: Knowledge base IDs to search within.
            emb_mdl: Embedding model for query vectorization.
            llm_mdl: Optional LLM for entity extraction (multi strategy).
            top_k: Number of results to return.
            hop_num: Number of hops for graph expansion (1-2).
            strategy: 'vector' (fast) or 'multi' (precise).

        Returns:
            dict with keys: chunks (list), total (int), doc_aggs (list).
            Returns empty structure if no results or on error.
        """
        empty = {"chunks": [], "total": 0, "doc_aggs": []}
        if not question or not question.strip() or not kb_ids:
            return empty

        sys_config = get_sag_system_config()
        timeout = sys_config.get("search_source_timeout", 10)
        fallback_vector = sys_config.get("search_fallback_vector", True)

        try:
            if strategy == "vector" or llm_mdl is None:
                chunks = await asyncio.wait_for(
                    self._vector_retrieval(question, tenant_ids, kb_ids, emb_mdl, top_k),
                    timeout=timeout,
                )
            else:
                try:
                    chunks = await asyncio.wait_for(
                        self._multi_retrieval(question, tenant_ids, kb_ids, emb_mdl, llm_mdl, top_k, hop_num),
                        timeout=timeout,
                    )
                except (asyncio.TimeoutError, Exception) as e:
                    if fallback_vector:
                        logger.warning("[SAG] Multi strategy failed (%s), falling back to vector", e)
                        chunks = await self._vector_retrieval(question, tenant_ids, kb_ids, emb_mdl, top_k)
                    else:
                        raise
        except asyncio.TimeoutError:
            logger.warning("[SAG] Retrieval timed out after %ds for kb_ids=%s", timeout, kb_ids)
            return empty
        except Exception:
            logger.exception("[SAG] Retrieval failed for kb_ids=%s", kb_ids)
            return empty

        if not chunks:
            return empty

        # Filter out topically-irrelevant chunks (ports the original SAG
        # reranker's relevance gate) before truncating to top_k, so off-topic
        # vector/expansion hits don't crowd out the on-topic evidence.
        chunks = self._apply_relevance_gate(question, chunks)
        if not chunks:
            return empty

        return {"chunks": chunks[:top_k], "total": len(chunks), "doc_aggs": []}

    # ------------------------------------------------------------------
    # Vector strategy (fast path)
    # ------------------------------------------------------------------

    async def _vector_retrieval(
        self,
        question: str,
        tenant_ids: list[str],
        kb_ids: list[str],
        emb_mdl,
        top_k: int,
    ) -> list[dict]:
        """Fast retrieval: embed question → search event vectors → map to chunks."""
        # 1. Embed the question
        query_vec = await self._embed_query(question, emb_mdl)
        if query_vec is None:
            return []

        # 2. Search event vectors in doc_store
        event_hits = await self._search_event_vectors(query_vec, tenant_ids, kb_ids, top_k * 4)
        if not event_hits:
            return []

        # 3. Map events to source chunks
        event_ids = [h["event_id"] for h in event_hits if h.get("event_id")]
        score_by_event = {h["event_id"]: h["score"] for h in event_hits}
        return await self._map_events_to_chunks(event_ids, score_by_event, kb_ids, tenant_ids, top_k)

    # ------------------------------------------------------------------
    # Multi strategy (precise path)
    # ------------------------------------------------------------------

    async def _multi_retrieval(
        self,
        question: str,
        tenant_ids: list[str],
        kb_ids: list[str],
        emb_mdl,
        llm_mdl,
        top_k: int,
        hop_num: int,
    ) -> list[dict]:
        """Precise retrieval: entity recall + vector + lexical + multi-hop expansion."""
        # Run three recall paths concurrently
        entity_task = self._entity_recall(question, kb_ids, llm_mdl)
        vector_task = self._vector_recall(question, tenant_ids, kb_ids, emb_mdl, top_k)
        lexical_task = self._lexical_recall(question, kb_ids)

        entity_events, vector_events, lexical_events = await asyncio.gather(
            entity_task, vector_task, lexical_task,
            return_exceptions=True,
        )

        # Collect seed event IDs (handle failures gracefully)
        seed_event_ids: set[int] = set()
        score_by_event: dict[int, float] = {}

        for result in [entity_events, vector_events, lexical_events]:
            if isinstance(result, Exception):
                logger.warning("[SAG] Recall path failed: %s", result)
                continue
            for eid, score in result:
                seed_event_ids.add(eid)
                score_by_event[eid] = max(score_by_event.get(eid, 0.0), score)

        if not seed_event_ids:
            return []

        # Multi-hop expansion
        hop_num = max(1, min(hop_num, 2))
        expanded_ids = await self._multi_hop_expand(
            list(seed_event_ids), kb_ids, hop_num, top_k * 3
        )

        # Merge seeds + expanded (expanded get lower base score)
        all_event_ids = list(seed_event_ids)
        for eid in expanded_ids:
            if eid not in score_by_event:
                score_by_event[eid] = 0.3  # base score for expanded events
                all_event_ids.append(eid)

        # Map to chunks
        return await self._map_events_to_chunks(all_event_ids, score_by_event, kb_ids, tenant_ids, top_k)

    # ------------------------------------------------------------------
    # Recall paths
    # ------------------------------------------------------------------

    async def _entity_recall(self, question: str, kb_ids: list[str], llm_mdl) -> list[tuple[int, float]]:
        """Extract entities from query via LLM, match in DB, return event IDs."""
        if llm_mdl is None:
            return []

        # Extract entity keywords from query using LLM
        entities = await self._extract_query_entities(question, llm_mdl)
        if not entities:
            # Fallback: use query terms directly
            entities = _extract_query_terms(question)
        if not entities:
            return []

        # Match entities in DB and get associated events
        return await thread_pool_exec(self._match_entities_to_events, entities, kb_ids)

    async def _vector_recall(self, question: str, tenant_ids: list[str], kb_ids: list[str], emb_mdl, top_k: int) -> list[tuple[int, float]]:
        """Vector similarity recall on event embeddings."""
        query_vec = await self._embed_query(question, emb_mdl)
        if query_vec is None:
            return []

        hits = await self._search_event_vectors(query_vec, tenant_ids, kb_ids, top_k * 4)
        return [(h["event_id"], h["score"]) for h in hits if h.get("event_id")]

    async def _lexical_recall(self, question: str, kb_ids: list[str]) -> list[tuple[int, float]]:
        """Lexical keyword matching on event content."""
        terms = _extract_query_terms(question)
        if not terms:
            return []
        return await thread_pool_exec(self._match_terms_to_events, terms, kb_ids)

    # ------------------------------------------------------------------
    # Multi-hop expansion
    # ------------------------------------------------------------------

    async def _multi_hop_expand(
        self,
        seed_event_ids: list[int],
        kb_ids: list[str],
        hop_num: int,
        max_per_hop: int,
    ) -> list[int]:
        """Expand seed events via shared entities (SQL JOIN)."""
        return await thread_pool_exec(
            self._sql_multi_hop, seed_event_ids, kb_ids, hop_num, max_per_hop
        )

    def _sql_multi_hop(self, seed_event_ids: list[int], kb_ids: list[str], hop_num: int, max_per_hop: int) -> list[int]:
        """Execute N-hop expansion via SQL JOIN on shared entities."""
        visited: set[int] = set(seed_event_ids)
        frontier: list[int] = list(seed_event_ids)
        expanded: list[int] = []

        for _hop in range(hop_num):
            if not frontier:
                break

            # Find events sharing entities with the current frontier
            new_events = (
                SagEvent.select(SagEvent.id)
                .join(SagEventEntity, on=(SagEventEntity.event_id == SagEvent.id))
                .where(
                    SagEventEntity.entity_id.in_(
                        SagEventEntity.select(SagEventEntity.entity_id).where(
                            SagEventEntity.event_id.in_(frontier)
                        )
                    ),
                    SagEvent.id.not_in(list(visited)),
                    SagEvent.kb_id.in_(kb_ids),
                    SagEvent.status == "completed",
                )
                .limit(max_per_hop)
            )

            new_ids = [e.id for e in new_events]
            if not new_ids:
                break

            expanded.extend(new_ids)
            visited.update(new_ids)
            frontier = new_ids

        return expanded

    # ------------------------------------------------------------------
    # Event → Chunk mapping
    # ------------------------------------------------------------------

    async def _map_events_to_chunks(
        self,
        event_ids: list[int],
        score_by_event: dict[int, float],
        kb_ids: list[str],
        tenant_ids: list[str],
        top_k: int,
    ) -> list[dict]:
        """Map event IDs back to source chunks and build result dicts."""
        if not event_ids:
            return []

        # Load events from DB to get chunk_ids
        events = list(SagEvent.select(
            SagEvent.id, SagEvent.chunk_id, SagEvent.doc_id, SagEvent.kb_id, SagEvent.title, SagEvent.summary
        ).where(
            SagEvent.id.in_(event_ids),
            SagEvent.status == "completed",
        ))

        if not events:
            return []

        # Deduplicate by chunk_id (keep highest scoring event)
        chunk_best: dict[str, tuple[float, object]] = {}
        for evt in events:
            score = score_by_event.get(evt.id, 0.5)
            if evt.chunk_id not in chunk_best or score > chunk_best[evt.chunk_id][0]:
                chunk_best[evt.chunk_id] = (score, evt)

        # Sort by score descending. Fetch a wider candidate pool than top_k so
        # the relevance gate in retrieval() can drop topically-irrelevant chunks
        # (e.g. multi-hop expansions that share an entity but miss the query's
        # actual terms) without starving the final result set.
        candidate_limit = max(top_k, min(top_k * 3, 30))
        sorted_chunks = sorted(chunk_best.items(), key=lambda x: x[1][0], reverse=True)[:candidate_limit]

        # Fetch actual chunk content from doc_store
        chunk_ids = [cid for cid, _ in sorted_chunks]
        chunk_contents = await self._fetch_chunks_from_store(chunk_ids, tenant_ids, kb_ids)

        # Fetch document names for citation display
        doc_name_map: dict[str, str] = {}
        try:
            from api.db.db_models import Document
            doc_ids = list({evt.doc_id for _, (_, evt) in sorted_chunks if evt.doc_id})
            if doc_ids:
                for d in Document.select(Document.id, Document.name).where(Document.id.in_(doc_ids)):
                    doc_name_map[d.id] = d.name
        except Exception:
            logger.warning("[SAG] Failed to fetch document names for SAG chunks")

        # Build result dicts
        results = []
        for chunk_id, (score, evt) in sorted_chunks:
            content = chunk_contents.get(chunk_id, "")
            if not content:
                # Fallback to event content if chunk not found in store
                content = f"{evt.title}\n{evt.summary or ''}"

            results.append({
                "chunk_id": chunk_id,
                "content_ltks": "",
                "content_with_weight": content,
                "doc_id": evt.doc_id,
                "docnm_kwd": doc_name_map.get(evt.doc_id, ""),
                "kb_id": evt.kb_id or (kb_ids[0] if kb_ids else ""),
                "important_kwd": [],
                "image_id": "",
                "similarity": round(score, 6),
                "vector_similarity": round(score, 6),
                "term_similarity": 0,
                "vector": [],
                "positions": [],
                # Mark the chunk as originating from SAG structured retrieval so
                # downstream surfaces (retrieval test, chat references) can badge
                # it and users can verify SAG is contributing.
                "sag_source": True,
                "sag_event_title": evt.title or "",
            })

        return results

    async def _fetch_chunks_from_store(
        self,
        chunk_ids: list[str],
        tenant_ids: list[str],
        kb_ids: list[str],
    ) -> dict[str, str]:
        """Fetch chunk content from doc_store by IDs."""
        if not chunk_ids:
            return {}

        result: dict[str, str] = {}
        for tid in tenant_ids:
            idxnm = search.index_name(tid)
            try:
                res = await thread_pool_exec(
                    self.doc_store_conn.search,
                    ["id", "content_with_weight"],
                    [],
                    {"id": chunk_ids, "available_int": 1},
                    [],
                    _order_by_expr(),
                    0,
                    len(chunk_ids),
                    idxnm,
                    kb_ids,
                )
                field_map = self.doc_store_conn.get_fields(res, ["id", "content_with_weight"])
                for row_id, row in field_map.items():
                    content = row.get("content_with_weight", "")
                    if content:
                        result[str(row_id)] = content
            except Exception:
                logger.warning("[SAG] Failed to fetch chunks from store for tenant %s", tid)
                continue

            if result:
                break

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_relevance_gate(self, question: str, chunks: list[dict]) -> list[dict]:
        """Filter out topically-irrelevant SAG chunks and re-score the survivors.

        Ports the original SAG reranker's relevance gate. Vector recall and
        multi-hop expansion both return same-domain-but-off-topic events (e.g.
        黄金价格 for a 精炼产量趋势 query) because they share entities/embedding
        neighbourhoods. Without a gate those flood the results. When the query
        carries a lexical signal, a chunk must lexically match to survive;
        otherwise survivors must clear a semantic floor relative to the best
        raw score. Returning nothing is the correct outcome when SAG has no
        on-topic evidence — it must not pollute the standard retrieval results.
        """
        if not chunks:
            return []

        for chunk in chunks:
            chunk["_lexical"] = _lexical_relevance(
                question,
                chunk.get("sag_event_title", "") or "",
                chunk.get("content_with_weight", "") or "",
            )

        has_lexical_signal = any(chunk["_lexical"] >= 0.2 for chunk in chunks)

        if has_lexical_signal:
            survivors = [chunk for chunk in chunks if chunk["_lexical"] >= 0.2]
        else:
            raw_scores = [max(0.0, chunk.get("similarity", 0.0)) for chunk in chunks]
            top_raw = max(raw_scores, default=0.0)
            semantic_floor = max(0.35, top_raw * 0.68)
            survivors = [
                chunk for chunk in chunks
                if max(0.0, chunk.get("similarity", 0.0)) >= semantic_floor
            ]

        if not survivors:
            for chunk in chunks:
                chunk.pop("_lexical", None)
            return []

        # Re-rank survivors by a combined semantic + rank + lexical score,
        # mirroring the original SAG reranker's weighting.
        survivors.sort(key=lambda c: max(0.0, c.get("similarity", 0.0)), reverse=True)
        denominator = max(1, len(survivors) - 1)
        for position, chunk in enumerate(survivors):
            raw = max(0.0, min(1.0, chunk.get("similarity", 0.0)))
            rank_score = 1.0 - position / denominator
            lexical = chunk.pop("_lexical", 0.0)
            combined = min(1.0, raw * 0.5 + rank_score * 0.2 + lexical * 0.3)
            chunk["similarity"] = round(combined, 6)
            chunk["vector_similarity"] = round(combined, 6)

        survivors.sort(key=lambda c: c.get("similarity", 0.0), reverse=True)
        return survivors

    async def _embed_query(self, question: str, emb_mdl) -> list[float] | None:
        """Embed the query text into a vector."""
        try:
            vectors, _ = await thread_pool_exec(emb_mdl.encode, [question])
            if vectors is not None and len(vectors) > 0:
                return vectors[0].tolist() if hasattr(vectors[0], "tolist") else list(vectors[0])
        except Exception:
            logger.exception("[SAG] Failed to embed query")
        return None

    async def _search_event_vectors(
        self,
        query_vec: list[float],
        tenant_ids: list[str],
        kb_ids: list[str],
        limit: int,
    ) -> list[dict]:
        """Search event vectors in doc_store with sag_kwd='event' filter."""
        from common.doc_store.doc_store_base import MatchDenseExpr

        hits: list[dict] = []
        for tid in tenant_ids:
            idxnm = search.index_name(tid)
            try:
                vctr_nm = "q_%d_vec" % len(query_vec)
                match_dense = MatchDenseExpr(vctr_nm, query_vec, "float", "cosine", limit)
                res = await thread_pool_exec(
                    self.doc_store_conn.search,
                    ["id", "sag_event_id", "content_with_weight", "_score"],
                    [],
                    {"sag_kwd": "event", "available_int": 1},
                    [match_dense],
                    _order_by_expr(),
                    0,
                    limit,
                    idxnm,
                    kb_ids,
                )
                field_map = self.doc_store_conn.get_fields(res, ["id", "sag_event_id", "content_with_weight", "_score"])
                for row_id, row in field_map.items():
                    event_id = row.get("sag_event_id")
                    score = row.get("_score", 0.5)
                    if event_id:
                        hits.append({"event_id": int(event_id), "score": float(score), "chunk_id": str(row_id)})
            except Exception:
                logger.warning("[SAG] Event vector search failed for tenant %s", tid)
                continue

        # Sort by score descending
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[:limit]

    async def _extract_query_entities(self, question: str, llm_mdl) -> list[str]:
        """Use LLM to extract entity keywords from the query."""
        system = (
            "从用户查询中提取关键实体名称（人名、地名、组织、产品、主题等）。"
            "仅输出实体名称，用逗号分隔，不要输出其他内容。"
            "如果没有明确实体，输出查询中的核心关键词。"
        )
        history = [{"role": "user", "content": question}]
        try:
            response = await asyncio.wait_for(
                llm_mdl.async_chat(system, history, {"temperature": 0.0, "max_tokens": 200}),
                timeout=10,
            )
            if response and isinstance(response, str) and "**ERROR**" not in response:
                entities = [e.strip() for e in response.split(",") if e.strip()]
                return entities[:8]
        except Exception:
            logger.warning("[SAG] LLM entity extraction failed, using term fallback")
        return []

    def _match_entities_to_events(self, entities: list[str], kb_ids: list[str]) -> list[tuple[int, float]]:
        """Match entity names in DB and return associated event IDs with scores."""
        results: list[tuple[int, float]] = []
        seen_events: set[int] = set()

        for entity_name in entities:
            # Find matching entities (case-insensitive LIKE)
            matched_ents = (
                SagEntity.select(SagEntity.id, SagEntity.heat)
                .where(
                    SagEntity.kb_id.in_(kb_ids),
                    SagEntity.entity_name.contains(entity_name),
                )
                .limit(10)
            )

            for ent in matched_ents:
                # Get associated events
                event_links = (
                    SagEventEntity.select(SagEventEntity.event_id, SagEventEntity.weight)
                    .where(SagEventEntity.entity_id == ent.id)
                    .limit(20)
                )
                for link in event_links:
                    if link.event_id not in seen_events:
                        seen_events.add(link.event_id)
                        # Score based on entity heat and link weight
                        score = min(1.0, 0.5 + 0.1 * min(ent.heat, 5) * link.weight)
                        results.append((link.event_id, score))

        return results[:100]

    def _match_terms_to_events(self, terms: list[str], kb_ids: list[str]) -> list[tuple[int, float]]:
        """Match lexical terms against event titles/summaries."""
        results: list[tuple[int, float]] = []
        seen_events: set[int] = set()

        for term in terms[:4]:
            events = (
                SagEvent.select(SagEvent.id)
                .where(
                    SagEvent.kb_id.in_(kb_ids),
                    SagEvent.status == "completed",
                    (SagEvent.title.contains(term) | SagEvent.summary.contains(term)),
                )
                .limit(5)
            )
            for evt in events:
                if evt.id not in seen_events:
                    seen_events.add(evt.id)
                    results.append((evt.id, 0.4))

        return results[:50]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_query_terms(question: str) -> list[str]:
    """Extract deterministic lexical terms from query (no LLM needed)."""
    cleaned = question.strip().lower()
    for phrase in _QUERY_NOISE:
        cleaned = cleaned.replace(phrase, " ")
    candidates = re.findall(
        r"[a-z0-9][a-z0-9_.+-]{1,31}|[\u3400-\u9fff]{2,16}",
        cleaned,
    )
    terms: list[str] = []
    for candidate in candidates:
        value = candidate.strip()
        if value and not value.isdigit() and value not in terms:
            terms.append(value)
    return terms[:4]


def _normalized(value: str) -> str:
    """Lowercase and keep only alphanumeric + CJK characters."""
    return "".join(re.findall(r"[a-z0-9\u3400-\u9fff]+", value.lower()))


def _chinese_bigrams(value: str) -> set[str]:
    """Character bigrams over CJK runs.

    Chinese carries no whitespace word boundaries, so the whole query often
    collapses into a single ``query_terms`` token and yields an almost useless
    lexical signal. Character bigrams give a segmentation-free overlap measure
    that still distinguishes on-topic results (sharing words like 精炼/产量)
    from merely same-domain ones (黄金价格/美联储).
    """
    chars = re.findall(r"[\u3400-\u9fff]", value)
    return {chars[i] + chars[i + 1] for i in range(len(chars) - 1)}


def _lexical_relevance(query: str, heading: str, content: str) -> float:
    """Segmentation-free lexical relevance between the query and one result.

    Mirrors the original SAG reranker's relevance-gate signal (whole-phrase
    match + per-term match, with heading matches weighted higher), extended
    with CJK character bigrams so Chinese queries produce a meaningful score.
    """
    heading_norm = _normalized(heading)
    content_norm = _normalized(content)
    text_norm = f"{heading_norm}{content_norm}"
    if not text_norm:
        return 0.0

    cleaned_query = query
    for phrase in _QUERY_NOISE:
        cleaned_query = cleaned_query.replace(phrase, " ")
    phrase = _normalized(cleaned_query)

    score = 0.0
    if phrase and len(phrase) >= 2 and phrase in text_norm:
        score += 0.45
        if phrase in heading_norm:
            score += 0.15

    terms = [_normalized(term) for term in _extract_query_terms(query)]
    terms = [term for term in terms if term]
    if terms:
        matched = sum(term in text_norm for term in terms)
        heading_matched = sum(term in heading_norm for term in terms)
        score += 0.25 * matched / len(terms)
        score += 0.1 * heading_matched / len(terms)

    query_bigrams = _chinese_bigrams(query)
    if query_bigrams:
        text_bigrams = _chinese_bigrams(text_norm)
        heading_bigrams = _chinese_bigrams(heading_norm)
        score += 0.35 * len(query_bigrams & text_bigrams) / len(query_bigrams)
        score += 0.1 * len(query_bigrams & heading_bigrams) / len(query_bigrams)

    return min(1.0, score)


def _order_by_expr():
    """Build a default OrderByExpr."""
    from common.doc_store.doc_store_base import OrderByExpr
    return OrderByExpr()


async def augment_with_sag(
    question: str,
    tenant_ids: list[str],
    kb_ids: list[str],
    embd_mdl,
    chat_mdl,
    existing_chunks: list[dict],
    max_chunks: int = 5,
) -> list[dict]:
    """Run SAG retrieval for SAG-enabled KBs and return deduped chunks to append.

    This is a non-invasive augmentation hook for the main retrieval path. It
    never raises: whenever SAG is uninitialized, disabled for all target KBs,
    or fails for any reason, it returns an empty list so the standard
    retrieval results are left untouched.

    Args:
        question: User query text.
        tenant_ids: Tenant IDs for index resolution.
        kb_ids: Candidate knowledge base IDs.
        embd_mdl: Embedding model for query vectorization.
        chat_mdl: Optional chat model for entity extraction (multi strategy).
            When None the retriever automatically falls back to vector strategy.
        existing_chunks: Chunks already returned by standard retrieval, used
            for chunk_id deduplication.
        max_chunks: Maximum number of SAG chunks to append.

    Returns:
        List of SAG-retrieved chunks (deduplicated against existing_chunks).
    """
    try:
        from common import settings
        if getattr(settings, "sag_retriever", None) is None:
            return []

        from api.db.services.knowledgebase_service import KnowledgebaseService
        from rag.sag.config import is_sag_enabled

        # Filter down to KBs that have SAG enabled
        sag_kbs = []
        for kb_id in kb_ids:
            try:
                ok, kb = KnowledgebaseService.get_by_id(kb_id)
            except Exception:
                continue
            if ok and kb and is_sag_enabled(kb.parser_config or {}):
                sag_kbs.append(kb)
        if not sag_kbs:
            return []

        # Use the first SAG-enabled KB's search config
        sag_cfg = (sag_kbs[0].parser_config or {}).get("sag", {}) or {}
        strategy = sag_cfg.get("search_strategy", "multi")
        hop_num = sag_cfg.get("hop_num", 1)
        top_k = sag_cfg.get("search_top_k", 10)

        result = await settings.sag_retriever.retrieval(
            question,
            tenant_ids,
            [kb.id for kb in sag_kbs],
            embd_mdl,
            llm_mdl=chat_mdl,
            top_k=top_k,
            hop_num=hop_num,
            strategy=strategy,
        )
        sag_chunks = result.get("chunks", []) if result else []
        if not sag_chunks:
            return []

        # Deduplicate against existing chunks by chunk_id
        existing_ids = {c.get("chunk_id") for c in existing_chunks if c.get("chunk_id")}
        new_chunks = [
            c for c in sag_chunks
            if c.get("chunk_id") and c.get("chunk_id") not in existing_ids
        ]
        if new_chunks:
            logger.info("[SAG] Augmented retrieval with %d SAG chunk(s) for kb_ids=%s", len(new_chunks[:max_chunks]), kb_ids)
        return new_chunks[:max_chunks]
    except Exception:
        logger.warning("[SAG] augment_with_sag failed, skipping SAG augmentation", exc_info=True)
        return []
