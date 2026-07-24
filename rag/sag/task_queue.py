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
"""SAG task queue utilities.

Provides helpers to create and enqueue SAG extraction/rebuild tasks
into the RAGFlow Redis task queue, following the same pattern as
queue_per_doc_raptor_task in document_service.
"""

import logging
from datetime import datetime

from api.db.db_models import Task
from api.db.db_utils import bulk_insert_into_db
from common.constants import MAXIMUM_TASK_PAGE_NUMBER
from common.misc_utils import get_uuid
from common import settings
from rag.utils.redis_conn import REDIS_CONN

logger = logging.getLogger(__name__)


def queue_sag_extract_task(doc_id: str, kb_id: str, tenant_id: str, priority: int = 0) -> str:
    """Queue a SAG extraction task for a single document.

    Creates a task record in the DB and pushes it to the Redis queue.
    The task_executor will pick it up and route to the SAG executor.

    Args:
        doc_id: Document ID to extract events/entities from.
        kb_id: Knowledge base ID the document belongs to.
        tenant_id: Tenant ID for queue routing.
        priority: Task priority (default 0).

    Returns:
        The created task ID.
    """
    task_id = get_uuid()
    task = {
        "id": task_id,
        "doc_id": doc_id,
        "from_page": MAXIMUM_TASK_PAGE_NUMBER,
        "to_page": MAXIMUM_TASK_PAGE_NUMBER,
        "task_type": "sag_extract",
        "priority": priority,
        "progress_msg": datetime.now().strftime("%H:%M:%S") + " created task sag_extract",
        "begin_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    bulk_insert_into_db(Task, [task], True)

    # Update knowledgebase sag_task_id
    from api.db.services.knowledgebase_service import KnowledgebaseService

    KnowledgebaseService.update_by_id(kb_id, {"sag_task_id": task_id})

    # Queue to Redis
    task["kb_id"] = kb_id
    task["tenant_id"] = tenant_id
    assert REDIS_CONN.queue_product(
        settings.get_svr_queue_name(priority, "common"),
        message=task,
    ), "Can't access Redis. Please check the Redis' status."

    logger.info("[SAG] Queued sag_extract task %s for doc %s in kb %s", task_id, doc_id, kb_id)
    return task_id


def queue_sag_rebuild_task(kb_id: str, tenant_id: str, doc_ids: list[str], priority: int = 0) -> str:
    """Queue a SAG rebuild task for an entire knowledge base.

    Creates a single rebuild task that will internally iterate over
    all documents in the knowledge base.

    Args:
        kb_id: Knowledge base ID to rebuild.
        tenant_id: Tenant ID for queue routing.
        doc_ids: List of document IDs to re-extract.
        priority: Task priority (default 0).

    Returns:
        The created task ID.
    """
    task_id = get_uuid()
    # Use the first doc_id as the task's doc_id (pattern from graphrag)
    fake_doc_id = doc_ids[0] if doc_ids else kb_id
    task = {
        "id": task_id,
        "doc_id": fake_doc_id,
        "from_page": MAXIMUM_TASK_PAGE_NUMBER,
        "to_page": MAXIMUM_TASK_PAGE_NUMBER,
        "task_type": "sag_rebuild",
        "priority": priority,
        "progress_msg": datetime.now().strftime("%H:%M:%S") + " created task sag_rebuild",
        "begin_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    bulk_insert_into_db(Task, [task], True)

    # Update knowledgebase sag_task_id
    from api.db.services.knowledgebase_service import KnowledgebaseService

    KnowledgebaseService.update_by_id(kb_id, {"sag_task_id": task_id})

    # Queue to Redis with doc_ids for downstream consumer
    task["kb_id"] = kb_id
    task["tenant_id"] = tenant_id
    task["doc_ids"] = doc_ids
    assert REDIS_CONN.queue_product(
        settings.get_svr_queue_name(priority, "common"),
        message=task,
    ), "Can't access Redis. Please check the Redis' status."

    logger.info("[SAG] Queued sag_rebuild task %s for kb %s (%d docs)", task_id, kb_id, len(doc_ids))
    return task_id
