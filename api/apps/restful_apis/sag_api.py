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
"""SAG Knowledge Graph Visualization API.

Provides endpoints for browsing the event-entity graph built by SAG extraction:
- GET /sag/kb/{kb_id}/graph - Get graph slice (events + entities + associations)
- GET /sag/kb/{kb_id}/nodes/{node_kind}/{node_id} - Get node details
- POST /sag/kb/{kb_id}/expand - Expand entity (lazy load associated events)
- GET /sag/kb/{kb_id}/entities - Get entity list (with type filter)
- GET /sag/kb/{kb_id}/events - Get event list (with pagination/category filter)
- GET /sag/kb/{kb_id}/status - Get SAG build status
- POST /sag/kb/{kb_id}/rebuild - Trigger full rebuild
- POST /sag/kb/{kb_id}/pause - Pause current task
- POST /sag/kb/{kb_id}/resume - Resume paused task
- POST /sag/kb/{kb_id}/cancel - Cancel current task
- GET /sag/kb/{kb_id}/config - Get SAG config
- PUT /sag/kb/{kb_id}/config - Update SAG config
"""

import logging

from quart import request

from api.apps import login_required
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.utils.api_utils import (
    add_tenant_id_to_kwargs,
    get_error_data_result,
    get_json_result,
    get_request_json,
    server_error_response,
)
from rag.sag.config import is_sag_enabled, validate_kb_sag_config, normalize_kb_sag_config, KB_DEFAULTS
from rag.sag.models import SagEvent, SagEntity, SagEventEntity, SagExtractCheckpoint
from api.db.db_models import Document

logger = logging.getLogger(__name__)

# Limits
MAX_EVENT_LIMIT = 1000
MAX_ENTITY_LIMIT = 1000
DEFAULT_EVENT_LIMIT = 100
DEFAULT_ENTITY_LIMIT = 80


@manager.route("/sag/kb/<kb_id>/graph", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def get_graph(tenant_id, kb_id):
    """
    Get SAG knowledge graph slice for a knowledge base.
    ---
    tags:
      - SAG
    security:
      - ApiKeyAuth: []
    parameters:
      - in: path
        name: kb_id
        type: string
        required: true
        description: Knowledge base ID
      - in: query
        name: event_limit
        type: integer
        default: 200
        description: Max events to return (max 1000)
      - in: query
        name: entity_limit
        type: integer
        default: 200
        description: Max entities to return (max 1000)
      - in: query
        name: doc_ids
        type: string
        description: Comma-separated document IDs to filter events
      - in: query
        name: entity_types
        type: string
        description: Comma-separated entity types to filter the graph
    responses:
      200:
        description: Graph slice with events, entities, and associations
    """
    try:
        # Validate KB access
        if not KnowledgebaseService.accessible(kb_id=kb_id, user_id=tenant_id):
            return get_error_data_result(message=f"You don't own the dataset {kb_id}.")

        ok, kb = KnowledgebaseService.get_by_id(kb_id)
        if not ok:
            return get_error_data_result(message=f"Dataset {kb_id} not found.")

        # Check if SAG is enabled
        kb_parser_config = kb.parser_config or {}
        if not is_sag_enabled(kb_parser_config):
            return get_json_result(data={
                "events": [],
                "entities": [],
                "associations": [],
                "total_events": 0,
                "total_entities": 0,
                "sag_enabled": False,
            })

        # Parse query params
        event_limit = min(int(request.args.get("event_limit", DEFAULT_EVENT_LIMIT)), MAX_EVENT_LIMIT)
        entity_limit = min(int(request.args.get("entity_limit", DEFAULT_ENTITY_LIMIT)), MAX_ENTITY_LIMIT)
        doc_ids_param = request.args.get("doc_ids", "")
        doc_ids = [d.strip() for d in doc_ids_param.split(",") if d.strip()] if doc_ids_param else None
        entity_types_param = request.args.get("entity_types", "")
        entity_types = [t.strip() for t in entity_types_param.split(",") if t.strip()] if entity_types_param else None

        # Query events
        event_query = SagEvent.select(
            SagEvent.id,
            SagEvent.title,
            SagEvent.summary,
            SagEvent.category,
            SagEvent.start_time,
            SagEvent.chunk_id,
            SagEvent.doc_id,
            SagEvent.rank,
        ).where(
            SagEvent.kb_id == kb_id,
            SagEvent.status == "completed",
        )
        if doc_ids:
            event_query = event_query.where(SagEvent.doc_id.in_(doc_ids))

        # When filtering by entity type, restrict events to those linked to at
        # least one entity of the requested types so the graph stays focused.
        if entity_types:
            typed_entity_ids = [
                ent.id for ent in SagEntity.select(SagEntity.id).where(
                    SagEntity.kb_id == kb_id,
                    SagEntity.entity_type.in_(entity_types),
                )
            ]
            if not typed_entity_ids:
                return get_json_result(data={
                    "events": [],
                    "entities": [],
                    "associations": [],
                    "total_events": 0,
                    "total_entities": 0,
                    "entity_types": [],
                    "sag_enabled": True,
                })
            linked_event_ids = [
                a.event_id for a in SagEventEntity.select(SagEventEntity.event_id).where(
                    SagEventEntity.entity_id.in_(typed_entity_ids),
                ).distinct()
            ]
            event_query = event_query.where(SagEvent.id.in_(linked_event_ids))

        total_events = event_query.count()
        events_rows = event_query.order_by(SagEvent.rank.desc(), SagEvent.id.desc()).limit(event_limit)

        # Collect event IDs for association lookup
        event_ids = [e.id for e in events_rows]

        # Query associations for these events
        associations_rows = []
        entity_ids_set = set()
        if event_ids:
            associations_rows = list(
                SagEventEntity.select(
                    SagEventEntity.event_id,
                    SagEventEntity.entity_id,
                    SagEventEntity.weight,
                    SagEventEntity.description,
                ).where(SagEventEntity.event_id.in_(event_ids))
            )
            entity_ids_set = {a.entity_id for a in associations_rows}

        # Query entities
        entities_rows = []
        total_entities = 0
        if entity_ids_set:
            entity_query = SagEntity.select(
                SagEntity.id,
                SagEntity.entity_name,
                SagEntity.entity_type,
                SagEntity.description,
                SagEntity.heat,
            ).where(
                SagEntity.kb_id == kb_id,
                SagEntity.id.in_(list(entity_ids_set)),
            )
            if entity_types:
                entity_query = entity_query.where(SagEntity.entity_type.in_(entity_types))
            total_entities = entity_query.count()
            entities_rows = entity_query.order_by(SagEntity.heat.desc()).limit(entity_limit)

        # Count entities per event
        entity_count_by_event = {}
        for a in associations_rows:
            entity_count_by_event[a.event_id] = entity_count_by_event.get(a.event_id, 0) + 1

        # Build response
        events = []
        for e in events_rows:
            events.append({
                "id": str(e.id),
                "title": e.title,
                "summary": e.summary or "",
                "category": e.category or "",
                "start_time": e.start_time.isoformat() if e.start_time else None,
                "chunk_id": e.chunk_id,
                "doc_id": e.doc_id,
                "rank": e.rank,
                "entity_count": entity_count_by_event.get(e.id, 0),
            })

        entities = []
        for ent in entities_rows:
            entities.append({
                "id": str(ent.id),
                "name": ent.entity_name,
                "type": ent.entity_type,
                "description": ent.description or "",
                "heat": ent.heat,
            })

        # Only include associations whose entity is in the returned set
        # (entity_limit may have truncated entities_rows, avoid dangling edges)
        returned_entity_ids = {ent.id for ent in entities_rows}
        returned_event_ids = {e.id for e in events_rows}

        associations = []
        for a in associations_rows:
            if a.entity_id not in returned_entity_ids:
                continue
            if a.event_id not in returned_event_ids:
                continue
            associations.append({
                "event_id": str(a.event_id),
                "entity_id": str(a.entity_id),
                "weight": a.weight,
                "description": a.description or "",
            })

        # Distinct entity types present in this KB (stable options for the
        # type filter dropdown, independent of the current filter).
        available_entity_types = [
            r.entity_type for r in SagEntity.select(SagEntity.entity_type).where(
                SagEntity.kb_id == kb_id,
            ).distinct() if r.entity_type
        ]

        return get_json_result(data={
            "events": events,
            "entities": entities,
            "associations": associations,
            "total_events": total_events,
            "total_entities": total_entities,
            "entity_types": available_entity_types,
            "sag_enabled": True,
        })
    except Exception as e:
        logger.exception("get_graph failed")
        return server_error_response(e)


@manager.route("/sag/kb/<kb_id>/nodes/<node_kind>/<node_id>", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def get_node_detail(tenant_id, kb_id, node_kind, node_id):
    """
    Get detailed information for a graph node (event or entity).
    ---
    tags:
      - SAG
    security:
      - ApiKeyAuth: []
    parameters:
      - in: path
        name: kb_id
        type: string
        required: true
      - in: path
        name: node_kind
        type: string
        required: true
        enum: [event, entity]
      - in: path
        name: node_id
        type: string
        required: true
    responses:
      200:
        description: Node detail with associated items
    """
    try:
        if not KnowledgebaseService.accessible(kb_id=kb_id, user_id=tenant_id):
            return get_error_data_result(message=f"You don't own the dataset {kb_id}.")

        if node_kind not in ("event", "entity"):
            return get_error_data_result(message="node_kind must be 'event' or 'entity'.")

        try:
            node_id_int = int(node_id)
        except ValueError:
            return get_error_data_result(message="Invalid node_id.")

        if node_kind == "event":
            return await _get_event_detail(kb_id, node_id_int)
        else:
            return await _get_entity_detail(kb_id, node_id_int)
    except Exception as e:
        logger.exception("get_node_detail failed")
        return server_error_response(e)


async def _get_event_detail(kb_id: str, event_id: int):
    """Get event detail with associated entities."""
    event = SagEvent.get_or_none(
        SagEvent.id == event_id,
        SagEvent.kb_id == kb_id,
    )
    if not event:
        return get_error_data_result(message="Event not found.")

    # Get associated entities
    associations = list(
        SagEventEntity.select(
            SagEventEntity.entity_id,
            SagEventEntity.weight,
            SagEventEntity.description,
        ).where(SagEventEntity.event_id == event_id)
    )

    entity_ids = [a.entity_id for a in associations]
    entities = []
    if entity_ids:
        entities_rows = SagEntity.select().where(SagEntity.id.in_(entity_ids))
        entity_map = {e.id: e for e in entities_rows}
        for a in associations:
            ent = entity_map.get(a.entity_id)
            if ent:
                entities.append({
                    "id": str(ent.id),
                    "name": ent.entity_name,
                    "type": ent.entity_type,
                    "description": ent.description or "",
                    "heat": ent.heat,
                    "weight": a.weight,
                    "association_description": a.description or "",
                })

    return get_json_result(data={
        "id": str(event.id),
        "title": event.title,
        "summary": event.summary or "",
        "content": event.content or "",
        "category": event.category or "",
        "start_time": event.start_time.isoformat() if event.start_time else None,
        "chunk_id": event.chunk_id,
        "doc_id": event.doc_id,
        "rank": event.rank,
        "status": event.status,
        "entities": entities,
    })


async def _get_entity_detail(kb_id: str, entity_id: int):
    """Get entity detail with associated events."""
    entity = SagEntity.get_or_none(
        SagEntity.id == entity_id,
        SagEntity.kb_id == kb_id,
    )
    if not entity:
        return get_error_data_result(message="Entity not found.")

    # Get associated events
    associations = list(
        SagEventEntity.select(
            SagEventEntity.event_id,
            SagEventEntity.weight,
            SagEventEntity.description,
        ).where(SagEventEntity.entity_id == entity_id)
    )

    event_ids = [a.event_id for a in associations]
    events = []
    if event_ids:
        events_rows = SagEvent.select(
            SagEvent.id,
            SagEvent.title,
            SagEvent.summary,
            SagEvent.category,
            SagEvent.start_time,
            SagEvent.doc_id,
        ).where(
            SagEvent.id.in_(event_ids),
            SagEvent.status == "completed",
        )
        event_map = {e.id: e for e in events_rows}
        for a in associations:
            evt = event_map.get(a.event_id)
            if evt:
                events.append({
                    "id": str(evt.id),
                    "title": evt.title,
                    "summary": evt.summary or "",
                    "category": evt.category or "",
                    "start_time": evt.start_time.isoformat() if evt.start_time else None,
                    "doc_id": evt.doc_id,
                    "weight": a.weight,
                    "association_description": a.description or "",
                })

    return get_json_result(data={
        "id": str(entity.id),
        "name": entity.entity_name,
        "type": entity.entity_type,
        "description": entity.description or "",
        "heat": entity.heat,
        "events": events,
    })


@manager.route("/sag/kb/<kb_id>/expand", methods=["POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def expand_node(tenant_id, kb_id):
    """
    Expand a node to load associated items (lazy loading).
    ---
    tags:
      - SAG
    security:
      - ApiKeyAuth: []
    parameters:
      - in: path
        name: kb_id
        type: string
        required: true
      - in: body
        name: body
        schema:
          type: object
          properties:
            node_kind:
              type: string
              enum: [entity, event]
            node_id:
              type: string
            limit:
              type: integer
              default: 20
    responses:
      200:
        description: Associated items with has_more flag
    """
    try:
        if not KnowledgebaseService.accessible(kb_id=kb_id, user_id=tenant_id):
            return get_error_data_result(message=f"You don't own the dataset {kb_id}.")

        req = await get_request_json()
        node_kind = req.get("node_kind", "entity")
        node_id = req.get("node_id")
        limit = min(int(req.get("limit", 20)), 100)

        if not node_id:
            return get_error_data_result(message="node_id is required.")

        try:
            node_id_int = int(node_id)
        except ValueError:
            return get_error_data_result(message="Invalid node_id.")

        if node_kind == "entity":
            return await _expand_entity(kb_id, node_id_int, limit)
        elif node_kind == "event":
            return await _expand_event(kb_id, node_id_int, limit)
        else:
            return get_error_data_result(message="node_kind must be 'entity' or 'event'.")
    except Exception as e:
        logger.exception("expand_node failed")
        return server_error_response(e)


async def _expand_entity(kb_id: str, entity_id: int, limit: int):
    """Expand entity to load associated events."""
    entity = SagEntity.get_or_none(SagEntity.id == entity_id, SagEntity.kb_id == kb_id)
    if not entity:
        return get_error_data_result(message="Entity not found.")

    # Get associations with pagination
    associations_query = SagEventEntity.select().where(SagEventEntity.entity_id == entity_id)
    total = associations_query.count()
    associations = list(associations_query.limit(limit + 1))
    has_more = len(associations) > limit
    associations = associations[:limit]

    event_ids = [a.event_id for a in associations]
    events = []
    if event_ids:
        events_rows = SagEvent.select(
            SagEvent.id,
            SagEvent.title,
            SagEvent.summary,
            SagEvent.category,
            SagEvent.start_time,
            SagEvent.doc_id,
        ).where(
            SagEvent.id.in_(event_ids),
            SagEvent.status == "completed",
        )
        event_map = {e.id: e for e in events_rows}
        for a in associations:
            evt = event_map.get(a.event_id)
            if evt:
                events.append({
                    "id": str(evt.id),
                    "title": evt.title,
                    "summary": evt.summary or "",
                    "category": evt.category or "",
                    "start_time": evt.start_time.isoformat() if evt.start_time else None,
                    "doc_id": evt.doc_id,
                })

    return get_json_result(data={
        "events": events,
        "associations": [
            {"event_id": str(a.event_id), "entity_id": str(a.entity_id), "weight": a.weight, "description": a.description or ""}
            for a in associations
        ],
        "has_more": has_more,
        "total": total,
    })


async def _expand_event(kb_id: str, event_id: int, limit: int):
    """Expand event to load associated entities."""
    event = SagEvent.get_or_none(SagEvent.id == event_id, SagEvent.kb_id == kb_id)
    if not event:
        return get_error_data_result(message="Event not found.")

    associations_query = SagEventEntity.select().where(SagEventEntity.event_id == event_id)
    total = associations_query.count()
    associations = list(associations_query.limit(limit + 1))
    has_more = len(associations) > limit
    associations = associations[:limit]

    entity_ids = [a.entity_id for a in associations]
    entities = []
    if entity_ids:
        entities_rows = SagEntity.select().where(SagEntity.id.in_(entity_ids))
        entity_map = {e.id: e for e in entities_rows}
        for a in associations:
            ent = entity_map.get(a.entity_id)
            if ent:
                entities.append({
                    "id": str(ent.id),
                    "name": ent.entity_name,
                    "type": ent.entity_type,
                    "description": ent.description or "",
                    "heat": ent.heat,
                })

    return get_json_result(data={
        "entities": entities,
        "associations": [
            {"event_id": str(a.event_id), "entity_id": str(a.entity_id), "weight": a.weight, "description": a.description or ""}
            for a in associations
        ],
        "has_more": has_more,
        "total": total,
    })


# ---------------------------------------------------------------------------
# Entity and Event List APIs
# ---------------------------------------------------------------------------


@manager.route("/sag/kb/<kb_id>/entities", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def list_entities(tenant_id, kb_id):
    """
    Get entity list for a knowledge base with optional type filter.
    ---
    tags:
      - SAG
    security:
      - ApiKeyAuth: []
    parameters:
      - in: path
        name: kb_id
        type: string
        required: true
      - in: query
        name: entity_type
        type: string
        description: Filter by entity type
      - in: query
        name: page
        type: integer
        default: 1
      - in: query
        name: page_size
        type: integer
        default: 20
    responses:
      200:
        description: Paginated entity list
    """
    try:
        if not KnowledgebaseService.accessible(kb_id=kb_id, user_id=tenant_id):
            return get_error_data_result(message=f"You don't own the dataset {kb_id}.")

        entity_type = request.args.get("entity_type", "")
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(100, max(1, int(request.args.get("page_size", 20))))

        query = SagEntity.select().where(SagEntity.kb_id == kb_id)
        if entity_type:
            query = query.where(SagEntity.entity_type == entity_type)

        total = query.count()
        entities = list(
            query.order_by(SagEntity.heat.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        return get_json_result(data={
            "page": page,
            "page_size": page_size,
            "total": total,
            "data": [
                {
                    "id": str(e.id),
                    "name": e.entity_name,
                    "type": e.entity_type,
                    "description": e.description or "",
                    "heat": e.heat,
                }
                for e in entities
            ],
        })
    except Exception as e:
        logger.exception("list_entities failed")
        return server_error_response(e)


@manager.route("/sag/kb/<kb_id>/events", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def list_events(tenant_id, kb_id):
    """
    Get event list for a knowledge base with pagination and category filter.
    ---
    tags:
      - SAG
    security:
      - ApiKeyAuth: []
    parameters:
      - in: path
        name: kb_id
        type: string
        required: true
      - in: query
        name: category
        type: string
        description: Filter by event category
      - in: query
        name: doc_id
        type: string
        description: Filter by document ID
      - in: query
        name: page
        type: integer
        default: 1
      - in: query
        name: page_size
        type: integer
        default: 20
    responses:
      200:
        description: Paginated event list
    """
    try:
        if not KnowledgebaseService.accessible(kb_id=kb_id, user_id=tenant_id):
            return get_error_data_result(message=f"You don't own the dataset {kb_id}.")

        category = request.args.get("category", "")
        doc_id = request.args.get("doc_id", "")
        page = max(1, int(request.args.get("page", 1)))
        page_size = min(100, max(1, int(request.args.get("page_size", 20))))

        query = SagEvent.select().where(
            SagEvent.kb_id == kb_id,
            SagEvent.status == "completed",
        )
        if category:
            query = query.where(SagEvent.category == category)
        if doc_id:
            query = query.where(SagEvent.doc_id == doc_id)

        total = query.count()
        events = list(
            query.order_by(SagEvent.rank.desc(), SagEvent.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        return get_json_result(data={
            "page": page,
            "page_size": page_size,
            "total": total,
            "data": [
                {
                    "id": str(e.id),
                    "title": e.title,
                    "summary": e.summary or "",
                    "category": e.category or "",
                    "start_time": e.start_time.isoformat() if e.start_time else None,
                    "chunk_id": e.chunk_id,
                    "doc_id": e.doc_id,
                    "rank": e.rank,
                }
                for e in events
            ],
        })
    except Exception as e:
        logger.exception("list_events failed")
        return server_error_response(e)


# ---------------------------------------------------------------------------
# Document Grouping API
# ---------------------------------------------------------------------------


@manager.route("/sag/kb/<kb_id>/docs", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def list_sag_docs(tenant_id, kb_id):
    """
    List documents that have SAG events, with per-document event/entity counts.

    Used by the frontend to group/filter the graph by document when a knowledge
    base contains many documents.
    ---
    tags:
      - SAG
    security:
      - ApiKeyAuth: []
    parameters:
      - in: path
        name: kb_id
        type: string
        required: true
        description: Knowledge base ID
    responses:
      200:
        description: Documents with SAG event counts
    """
    try:
        if not KnowledgebaseService.accessible(kb_id=kb_id, user_id=tenant_id):
            return get_error_data_result(message=f"You don't own the dataset {kb_id}.")

        # Aggregate event count per document
        event_counts = {}
        rows = (
            SagEvent.select(SagEvent.doc_id, SagEvent.id)
            .where(SagEvent.kb_id == kb_id, SagEvent.status == "completed")
        )
        for r in rows:
            event_counts[r.doc_id] = event_counts.get(r.doc_id, 0) + 1

        if not event_counts:
            return get_json_result(data={"docs": []})

        # Fetch document names (only documents that still exist)
        doc_ids = list(event_counts.keys())
        doc_name_map = {}
        docs_rows = Document.select(Document.id, Document.name).where(
            Document.id.in_(doc_ids)
        )
        for d in docs_rows:
            doc_name_map[d.id] = d.name

        # Purge SAG data left behind for documents that no longer exist
        # (e.g. deleted before per-document SAG cleanup was in place), so the
        # graph and dropdown never surface orphaned events. This is a targeted,
        # self-healing cleanup that runs only when orphans are detected.
        orphan_doc_ids = [doc_id for doc_id in doc_ids if doc_id not in doc_name_map]
        if orphan_doc_ids:
            from rag.sag.cleanup import cleanup_sag_data_for_docs

            cleanup_sag_data_for_docs(orphan_doc_ids, kb_id, tenant_id)
            for doc_id in orphan_doc_ids:
                event_counts.pop(doc_id, None)

        docs = []
        for doc_id in doc_ids:
            if doc_id not in doc_name_map:
                continue
            docs.append({
                "doc_id": doc_id,
                "name": doc_name_map[doc_id],
                "event_count": event_counts[doc_id],
            })

        # Sort by event count desc
        docs.sort(key=lambda x: x["event_count"], reverse=True)

        return get_json_result(data={"docs": docs})
    except Exception as e:
        logger.exception("list_sag_docs failed")
        return server_error_response(e)


# ---------------------------------------------------------------------------
# Task Management APIs
# ---------------------------------------------------------------------------


@manager.route("/sag/kb/<kb_id>/status", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def get_sag_status(tenant_id, kb_id):
    """
    Get SAG build status for a knowledge base.
    ---
    tags:
      - SAG
    security:
      - ApiKeyAuth: []
    parameters:
      - in: path
        name: kb_id
        type: string
        required: true
    responses:
      200:
        description: SAG status including task progress and statistics
    """
    try:
        if not KnowledgebaseService.accessible(kb_id=kb_id, user_id=tenant_id):
            return get_error_data_result(message=f"You don't own the dataset {kb_id}.")

        ok, kb = KnowledgebaseService.get_by_id(kb_id)
        if not ok:
            return get_error_data_result(message=f"Dataset {kb_id} not found.")

        kb_parser_config = kb.parser_config or {}
        enabled = is_sag_enabled(kb_parser_config)

        # Get task status
        task_id = getattr(kb, "sag_task_id", None) or ""
        task_status = "idle"
        progress = 0.0

        if task_id:
            from api.db.db_models import Task
            task = Task.get_or_none(Task.id == task_id)
            if task:
                task_status = task.progress_msg if hasattr(task, "progress_msg") else "unknown"
                progress = task.progress if hasattr(task, "progress") else 0.0
                if progress >= 1.0:
                    task_status = "completed"
                elif progress < 0:
                    task_status = "failed"

        # Get statistics
        event_count = SagEvent.select().where(
            SagEvent.kb_id == kb_id,
            SagEvent.status == "completed",
        ).count()
        entity_count = SagEntity.select().where(SagEntity.kb_id == kb_id).count()

        # Get token usage from checkpoints
        token_usage = 0
        checkpoints = SagExtractCheckpoint.select(SagExtractCheckpoint.token_usage).where(
            SagExtractCheckpoint.kb_id == kb_id
        )
        for cp in checkpoints:
            token_usage += cp.token_usage or 0

        return get_json_result(data={
            "enabled": enabled,
            "task_id": task_id,
            "task_status": task_status,
            "progress": progress,
            "event_count": event_count,
            "entity_count": entity_count,
            "token_usage": token_usage,
        })
    except Exception as e:
        logger.exception("get_sag_status failed")
        return server_error_response(e)


@manager.route("/sag/kb/<kb_id>/rebuild", methods=["POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def trigger_rebuild(tenant_id, kb_id):
    """
    Trigger a full SAG rebuild for a knowledge base.
    ---
    tags:
      - SAG
    security:
      - ApiKeyAuth: []
    parameters:
      - in: path
        name: kb_id
        type: string
        required: true
    responses:
      200:
        description: Rebuild task created
    """
    try:
        if not KnowledgebaseService.accessible(kb_id=kb_id, user_id=tenant_id):
            return get_error_data_result(message=f"You don't own the dataset {kb_id}.")

        ok, kb = KnowledgebaseService.get_by_id(kb_id)
        if not ok:
            return get_error_data_result(message=f"Dataset {kb_id} not found.")

        kb_parser_config = kb.parser_config or {}
        if not is_sag_enabled(kb_parser_config):
            return get_error_data_result(message="SAG is not enabled for this dataset.")

        # Get all document IDs in this KB
        from api.db.services.document_service import DocumentService
        docs = DocumentService.query(kb_id=kb_id)
        doc_ids = [doc.id for doc in docs]

        if not doc_ids:
            return get_error_data_result(message="No documents found in this dataset.")

        # Queue rebuild task
        from rag.sag.task_queue import queue_sag_rebuild_task
        task_id = queue_sag_rebuild_task(
            kb_id=kb_id,
            tenant_id=kb.tenant_id,
            doc_ids=doc_ids,
        )

        return get_json_result(data={
            "task_id": task_id,
            "message": "SAG rebuild task created",
            "doc_count": len(doc_ids),
        })
    except Exception as e:
        logger.exception("trigger_rebuild failed")
        return server_error_response(e)


@manager.route("/sag/kb/<kb_id>/pause", methods=["POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def pause_task(tenant_id, kb_id):
    """
    Pause the current SAG task for a knowledge base.
    ---
    tags:
      - SAG
    security:
      - ApiKeyAuth: []
    """
    try:
        if not KnowledgebaseService.accessible(kb_id=kb_id, user_id=tenant_id):
            return get_error_data_result(message=f"You don't own the dataset {kb_id}.")

        ok, kb = KnowledgebaseService.get_by_id(kb_id)
        if not ok:
            return get_error_data_result(message=f"Dataset {kb_id} not found.")

        task_id = getattr(kb, "sag_task_id", None)
        if not task_id:
            return get_error_data_result(message="No active SAG task.")

        # Update checkpoint status to paused
        updated = SagExtractCheckpoint.update(status="paused").where(
            SagExtractCheckpoint.kb_id == kb_id,
            SagExtractCheckpoint.task_id == task_id,
            SagExtractCheckpoint.status == "running",
        ).execute()

        if updated == 0:
            return get_error_data_result(message="No running task to pause.")

        return get_json_result(data={"message": "Task paused", "task_id": task_id})
    except Exception as e:
        logger.exception("pause_task failed")
        return server_error_response(e)


@manager.route("/sag/kb/<kb_id>/resume", methods=["POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def resume_task(tenant_id, kb_id):
    """
    Resume a paused SAG task for a knowledge base.
    ---
    tags:
      - SAG
    security:
      - ApiKeyAuth: []
    """
    try:
        if not KnowledgebaseService.accessible(kb_id=kb_id, user_id=tenant_id):
            return get_error_data_result(message=f"You don't own the dataset {kb_id}.")

        ok, kb = KnowledgebaseService.get_by_id(kb_id)
        if not ok:
            return get_error_data_result(message=f"Dataset {kb_id} not found.")

        task_id = getattr(kb, "sag_task_id", None)
        if not task_id:
            return get_error_data_result(message="No active SAG task.")

        # Update checkpoint status to running
        updated = SagExtractCheckpoint.update(status="running").where(
            SagExtractCheckpoint.kb_id == kb_id,
            SagExtractCheckpoint.task_id == task_id,
            SagExtractCheckpoint.status == "paused",
        ).execute()

        if updated == 0:
            return get_error_data_result(message="No paused task to resume.")

        return get_json_result(data={"message": "Task resumed", "task_id": task_id})
    except Exception as e:
        logger.exception("resume_task failed")
        return server_error_response(e)


@manager.route("/sag/kb/<kb_id>/cancel", methods=["POST"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def cancel_task(tenant_id, kb_id):
    """
    Cancel the current SAG task for a knowledge base.
    ---
    tags:
      - SAG
    security:
      - ApiKeyAuth: []
    """
    try:
        if not KnowledgebaseService.accessible(kb_id=kb_id, user_id=tenant_id):
            return get_error_data_result(message=f"You don't own the dataset {kb_id}.")

        ok, kb = KnowledgebaseService.get_by_id(kb_id)
        if not ok:
            return get_error_data_result(message=f"Dataset {kb_id} not found.")

        task_id = getattr(kb, "sag_task_id", None)
        if not task_id:
            return get_error_data_result(message="No active SAG task.")

        # Update checkpoint status to cancelled
        updated = SagExtractCheckpoint.update(status="cancelled").where(
            SagExtractCheckpoint.kb_id == kb_id,
            SagExtractCheckpoint.task_id == task_id,
            SagExtractCheckpoint.status.in_(["running", "paused"]),
        ).execute()

        if updated == 0:
            return get_error_data_result(message="No active task to cancel.")

        return get_json_result(data={"message": "Task cancelled", "task_id": task_id})
    except Exception as e:
        logger.exception("cancel_task failed")
        return server_error_response(e)


# ---------------------------------------------------------------------------
# Config Management APIs
# ---------------------------------------------------------------------------


@manager.route("/sag/kb/<kb_id>/config", methods=["GET"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def get_sag_config(tenant_id, kb_id):
    """
    Get SAG configuration for a knowledge base.
    ---
    tags:
      - SAG
    security:
      - ApiKeyAuth: []
    """
    try:
        if not KnowledgebaseService.accessible(kb_id=kb_id, user_id=tenant_id):
            return get_error_data_result(message=f"You don't own the dataset {kb_id}.")

        ok, kb = KnowledgebaseService.get_by_id(kb_id)
        if not ok:
            return get_error_data_result(message=f"Dataset {kb_id} not found.")

        kb_parser_config = kb.parser_config or {}
        sag_config = kb_parser_config.get("sag", {})

        # Merge with defaults
        merged_config = {**KB_DEFAULTS, **sag_config}

        return get_json_result(data=merged_config)
    except Exception as e:
        logger.exception("get_sag_config failed")
        return server_error_response(e)


@manager.route("/sag/kb/<kb_id>/config", methods=["PUT"])  # noqa: F821
@login_required
@add_tenant_id_to_kwargs
async def update_sag_config(tenant_id, kb_id):
    """
    Update SAG configuration for a knowledge base.
    ---
    tags:
      - SAG
    security:
      - ApiKeyAuth: []
    parameters:
      - in: body
        name: body
        schema:
          type: object
          properties:
            enabled:
              type: boolean
            extract_model:
              type: string
            extract_concurrency:
              type: integer
            chunk_max_tokens:
              type: integer
            search_strategy:
              type: string
              enum: [vector, multi]
            search_top_k:
              type: integer
            hop_num:
              type: integer
    """
    try:
        if not KnowledgebaseService.accessible(kb_id=kb_id, user_id=tenant_id):
            return get_error_data_result(message=f"You don't own the dataset {kb_id}.")

        ok, kb = KnowledgebaseService.get_by_id(kb_id)
        if not ok:
            return get_error_data_result(message=f"Dataset {kb_id} not found.")

        req = await get_request_json()

        # Validate config
        error = validate_kb_sag_config(req)
        if error:
            return get_error_data_result(message=error)

        # Normalize and merge with existing config
        kb_parser_config = kb.parser_config or {}
        existing_sag = kb_parser_config.get("sag", {})
        new_sag = normalize_kb_sag_config({**existing_sag, **req})
        kb_parser_config["sag"] = new_sag

        # Update KB
        if not KnowledgebaseService.update_by_id(kb_id, {"parser_config": kb_parser_config}):
            return get_error_data_result(message="Failed to update configuration.")

        return get_json_result(data=new_sag)
    except Exception as e:
        logger.exception("update_sag_config failed")
        return server_error_response(e)
