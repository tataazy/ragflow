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
"""SAG Peewee ORM models.

Defines the four core tables:
- sag_events: extracted events (one per chunk)
- sag_entities: deduplicated entities per knowledge base
- sag_event_entity: many-to-many association with weight
- sag_extract_checkpoint: per-document extraction progress for resume
"""

from peewee import (
    BigIntegerField,
    CharField,
    DateTimeField,
    FloatField,
    IntegerField,
    TextField,
)

from api.db.db_models import DB, DataBaseModel, JSONField, LongTextField


class SagEvent(DataBaseModel):
    """A single event extracted from one chunk."""

    id = BigIntegerField(primary_key=True)
    kb_id = CharField(max_length=32, null=False, index=True)
    doc_id = CharField(max_length=32, null=False, index=True)
    chunk_id = CharField(max_length=64, null=False, index=True)
    title = CharField(max_length=255, null=False)
    summary = TextField(null=True)
    content = TextField(null=False)
    category = CharField(max_length=64, null=True)
    start_time = DateTimeField(null=True)
    parent_id = BigIntegerField(null=True)
    rank = IntegerField(default=0)
    event_embedding = LongTextField(null=True, help_text="event vector (redundant, primary in doc_store)")
    status = CharField(max_length=16, null=False, default="completed")

    class Meta:
        db_table = "sag_events"
        indexes = (
            (("kb_id", "doc_id"), False),
            (("kb_id", "category"), False),
        )


class SagEntity(DataBaseModel):
    """A deduplicated entity within a knowledge base."""

    id = BigIntegerField(primary_key=True)
    kb_id = CharField(max_length=32, null=False, index=True)
    entity_name = CharField(max_length=255, null=False)
    entity_type = CharField(max_length=32, null=False)
    description = TextField(null=True)
    heat = IntegerField(default=1)

    class Meta:
        db_table = "sag_entities"
        indexes = (
            (("kb_id", "entity_name", "entity_type"), True),
            (("kb_id", "entity_type"), False),
        )


class SagEventEntity(DataBaseModel):
    """Association between an event and an entity."""

    id = BigIntegerField(primary_key=True)
    event_id = BigIntegerField(null=False)
    entity_id = BigIntegerField(null=False)
    weight = FloatField(default=1.0)
    description = CharField(max_length=512, null=True)

    class Meta:
        db_table = "sag_event_entity"
        indexes = (
            (("event_id", "entity_id"), True),
            (("entity_id",), False),
        )


class SagExtractCheckpoint(DataBaseModel):
    """Per-document extraction progress for pause/resume."""

    id = BigIntegerField(primary_key=True)
    kb_id = CharField(max_length=32, null=False)
    doc_id = CharField(max_length=32, null=False)
    task_id = CharField(max_length=32, null=False)
    processed_chunk_ids = JSONField(null=True, default=list)
    event_ids = JSONField(null=True, default=list)
    event_count = IntegerField(default=0)
    token_usage = IntegerField(default=0)
    status = CharField(max_length=16, null=False, default="running")

    class Meta:
        db_table = "sag_extract_checkpoint"
        indexes = ((("kb_id", "doc_id", "task_id"), True),)
