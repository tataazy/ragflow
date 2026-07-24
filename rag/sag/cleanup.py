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
"""SAG data cleanup utilities.

Provides functions to remove SAG data (events, entities, associations,
checkpoints, and doc_store vectors) when documents are deleted or re-parsed.
"""

import logging

logger = logging.getLogger(__name__)


def cleanup_sag_data_for_docs(doc_ids: list[str], kb_id: str, tenant_id: str) -> None:
    """Remove all SAG data associated with the given documents.

    Deletes events, event-entity associations, orphaned entities,
    checkpoints, and event vectors from doc_store.

    Args:
        doc_ids: List of document IDs to clean up.
        kb_id: Knowledge base ID.
        tenant_id: Tenant ID for doc_store index resolution.
    """
    if not doc_ids:
        return

    from rag.sag.models import SagEvent, SagEntity, SagEventEntity, SagExtractCheckpoint

    try:
        # 1. Get event IDs for these documents
        events = SagEvent.select(SagEvent.id).where(
            SagEvent.doc_id.in_(doc_ids),
            SagEvent.kb_id == kb_id,
        )
        event_ids = [e.id for e in events]

        if event_ids:
            # 2. Delete event-entity associations
            SagEventEntity.delete().where(
                SagEventEntity.event_id.in_(event_ids)
            ).execute()

            # 3. Delete events
            SagEvent.delete().where(
                SagEvent.id.in_(event_ids)
            ).execute()

        # 4. Delete checkpoints for these documents
        SagExtractCheckpoint.delete().where(
            SagExtractCheckpoint.doc_id.in_(doc_ids),
            SagExtractCheckpoint.kb_id == kb_id,
        ).execute()

        # 5. Clean up orphaned entities (entities with no remaining associations)
        _cleanup_orphaned_entities(kb_id)

        # 6. Delete event vectors from doc_store
        _cleanup_doc_store_vectors(doc_ids, kb_id, tenant_id)

        logger.info(
            "[SAG] Cleaned up SAG data for %d docs in kb %s (%d events removed)",
            len(doc_ids), kb_id, len(event_ids),
        )
    except Exception:
        logger.exception("[SAG] Failed to cleanup SAG data for docs %s in kb %s", doc_ids, kb_id)


def cleanup_sag_data_for_kb(kb_id: str, tenant_id: str) -> None:
    """Remove ALL SAG data for a knowledge base (used by sag_rebuild).

    Args:
        kb_id: Knowledge base ID.
        tenant_id: Tenant ID for doc_store index resolution.
    """
    from rag.sag.models import SagEvent, SagEntity, SagEventEntity, SagExtractCheckpoint

    try:
        # Delete all associations for this KB's events
        event_ids_subquery = SagEvent.select(SagEvent.id).where(SagEvent.kb_id == kb_id)
        SagEventEntity.delete().where(
            SagEventEntity.event_id.in_(event_ids_subquery)
        ).execute()

        # Delete all events
        SagEvent.delete().where(SagEvent.kb_id == kb_id).execute()

        # Delete all entities
        SagEntity.delete().where(SagEntity.kb_id == kb_id).execute()

        # Delete all checkpoints
        SagExtractCheckpoint.delete().where(SagExtractCheckpoint.kb_id == kb_id).execute()

        # Delete all event vectors from doc_store
        _cleanup_doc_store_vectors_for_kb(kb_id, tenant_id)

        logger.info("[SAG] Cleaned up ALL SAG data for kb %s", kb_id)
    except Exception:
        logger.exception("[SAG] Failed to cleanup all SAG data for kb %s", kb_id)


def _cleanup_orphaned_entities(kb_id: str) -> None:
    """Remove entities that have no remaining event associations."""
    from rag.sag.models import SagEntity, SagEventEntity

    # Find entities with no associations
    associated_entity_ids = (
        SagEventEntity.select(SagEventEntity.entity_id)
        .where(SagEventEntity.entity_id.is_null(False))
        .distinct()
    )
    SagEntity.delete().where(
        SagEntity.kb_id == kb_id,
        SagEntity.id.not_in(associated_entity_ids),
    ).execute()


def _cleanup_doc_store_vectors(doc_ids: list[str], kb_id: str, tenant_id: str) -> None:
    """Delete event vectors from doc_store for specific documents."""
    from common import settings
    from rag.nlp import search

    try:
        idxnm = search.index_name(tenant_id)
        if settings.docStoreConn.index_exist(idxnm, kb_id):
            settings.docStoreConn.delete(
                {"doc_id": doc_ids, "sag_kwd": "event"},
                idxnm,
                kb_id,
            )
    except Exception:
        logger.exception("[SAG] Failed to delete event vectors from doc_store for docs %s", doc_ids)


def _cleanup_doc_store_vectors_for_kb(kb_id: str, tenant_id: str) -> None:
    """Delete all event vectors from doc_store for a knowledge base."""
    from common import settings
    from rag.nlp import search

    try:
        idxnm = search.index_name(tenant_id)
        if settings.docStoreConn.index_exist(idxnm, kb_id):
            settings.docStoreConn.delete(
                {"sag_kwd": "event"},
                idxnm,
                kb_id,
            )
    except Exception:
        logger.exception("[SAG] Failed to delete event vectors from doc_store for kb %s", kb_id)
