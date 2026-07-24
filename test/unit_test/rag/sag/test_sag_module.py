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

"""Unit tests for SAG (Structured Association Graph) module."""

import pytest
from unittest.mock import MagicMock, patch


class TestSagConfig:
    """Tests for rag.sag.config module."""

    def test_is_sag_enabled_true(self):
        """Test is_sag_enabled returns True when enabled."""
        from rag.sag.config import is_sag_enabled

        parser_config = {"sag": {"enabled": True}}
        assert is_sag_enabled(parser_config) is True

    def test_is_sag_enabled_false(self):
        """Test is_sag_enabled returns False when disabled."""
        from rag.sag.config import is_sag_enabled

        parser_config = {"sag": {"enabled": False}}
        assert is_sag_enabled(parser_config) is False

    def test_is_sag_enabled_missing(self):
        """Test is_sag_enabled returns False when sag config missing."""
        from rag.sag.config import is_sag_enabled

        assert is_sag_enabled({}) is False
        assert is_sag_enabled({"other": "config"}) is False
        assert is_sag_enabled(None) is False

    def test_validate_kb_sag_config_valid(self):
        """Test validate_kb_sag_config with valid config."""
        from rag.sag.config import validate_kb_sag_config

        config = {
            "enabled": True,
            "extract_concurrency": 4,
            "search_strategy": "multi",
            "search_top_k": 10,
            "hop_num": 1,
        }
        error = validate_kb_sag_config(config)
        assert error is None

    def test_validate_kb_sag_config_invalid_concurrency(self):
        """Test validate_kb_sag_config rejects invalid concurrency."""
        from rag.sag.config import validate_kb_sag_config

        config = {"extract_concurrency": 100}
        error = validate_kb_sag_config(config)
        assert error is not None
        assert "concurrency" in error.lower()

    def test_validate_kb_sag_config_invalid_strategy(self):
        """Test validate_kb_sag_config rejects invalid strategy."""
        from rag.sag.config import validate_kb_sag_config

        config = {"search_strategy": "invalid"}
        error = validate_kb_sag_config(config)
        assert error is not None

    def test_normalize_kb_sag_config(self):
        """Test normalize_kb_sag_config fills defaults."""
        from rag.sag.config import normalize_kb_sag_config, KB_DEFAULTS

        config = {"enabled": True}
        normalized = normalize_kb_sag_config(config)

        assert normalized["enabled"] is True
        assert normalized["extract_concurrency"] == KB_DEFAULTS["extract_concurrency"]
        assert normalized["search_strategy"] == KB_DEFAULTS["search_strategy"]
        assert normalized["search_top_k"] == KB_DEFAULTS["search_top_k"]
        assert normalized["hop_num"] == KB_DEFAULTS["hop_num"]

    def test_get_sag_system_config(self):
        """Test get_sag_system_config returns defaults."""
        from rag.sag.config import get_sag_system_config, _DEFAULTS

        config = get_sag_system_config()
        assert "extract_timeout" in config
        assert "entity_types" in config
        assert config["extract_timeout"] == _DEFAULTS["extract_timeout"]


class TestSagModels:
    """Tests for rag.sag.models module."""

    def test_sag_event_model_fields(self):
        """Test SagEvent model has required fields."""
        from rag.sag.models import SagEvent

        # Check model has expected fields
        field_names = [f.name for f in SagEvent._meta.sorted_fields]
        assert "id" in field_names
        assert "kb_id" in field_names
        assert "doc_id" in field_names
        assert "chunk_id" in field_names
        assert "title" in field_names
        assert "summary" in field_names
        assert "category" in field_names
        assert "status" in field_names

    def test_sag_entity_model_fields(self):
        """Test SagEntity model has required fields."""
        from rag.sag.models import SagEntity

        field_names = [f.name for f in SagEntity._meta.sorted_fields]
        assert "id" in field_names
        assert "kb_id" in field_names
        assert "entity_name" in field_names
        assert "entity_type" in field_names
        assert "heat" in field_names

    def test_sag_event_entity_model_fields(self):
        """Test SagEventEntity model has required fields."""
        from rag.sag.models import SagEventEntity

        field_names = [f.name for f in SagEventEntity._meta.sorted_fields]
        assert "id" in field_names
        assert "event_id" in field_names
        assert "entity_id" in field_names
        assert "weight" in field_names

    def test_sag_extract_checkpoint_model_fields(self):
        """Test SagExtractCheckpoint model has required fields."""
        from rag.sag.models import SagExtractCheckpoint

        field_names = [f.name for f in SagExtractCheckpoint._meta.sorted_fields]
        assert "id" in field_names
        assert "kb_id" in field_names
        assert "doc_id" in field_names
        assert "task_id" in field_names
        assert "status" in field_names


class TestSagCleanup:
    """Tests for rag.sag.cleanup module."""

    @patch("rag.sag.cleanup.SagEvent")
    @patch("rag.sag.cleanup.SagEntity")
    @patch("rag.sag.cleanup.SagEventEntity")
    @patch("rag.sag.cleanup.SagExtractCheckpoint")
    def test_cleanup_sag_data_for_docs(
        self, mock_checkpoint, mock_event_entity, mock_entity, mock_event
    ):
        """Test cleanup_sag_data_for_docs deletes related data."""
        from rag.sag.cleanup import cleanup_sag_data_for_docs

        # Setup mocks
        mock_event.select.return_value.where.return_value.where.return_value = []
        mock_event_entity.delete.return_value.where.return_value.where.return_value.execute.return_value = 0
        mock_event.delete.return_value.where.return_value.where.return_value.execute.return_value = 0
        mock_checkpoint.delete.return_value.where.return_value.where.return_value.execute.return_value = 0

        # Should not raise
        cleanup_sag_data_for_docs(["doc1", "doc2"], "kb1", "tenant1")

    @patch("rag.sag.cleanup.SagEvent")
    @patch("rag.sag.cleanup.SagEntity")
    @patch("rag.sag.cleanup.SagEventEntity")
    @patch("rag.sag.cleanup.SagExtractCheckpoint")
    def test_cleanup_sag_data_for_kb(
        self, mock_checkpoint, mock_event_entity, mock_entity, mock_event
    ):
        """Test cleanup_sag_data_for_kb deletes all KB data."""
        from rag.sag.cleanup import cleanup_sag_data_for_kb

        # Setup mocks
        mock_event.select.return_value.where.return_value = []
        mock_event_entity.delete.return_value.where.return_value.execute.return_value = 0
        mock_event.delete.return_value.where.return_value.execute.return_value = 0
        mock_entity.delete.return_value.where.return_value.execute.return_value = 0
        mock_checkpoint.delete.return_value.where.return_value.execute.return_value = 0

        # Should not raise
        cleanup_sag_data_for_kb("kb1", "tenant1")


class TestSagTaskQueue:
    """Tests for rag.sag.task_queue module."""

    @patch("rag.sag.task_queue.get_uuid")
    @patch("rag.sag.task_queue.settings")
    def test_queue_sag_extract_task(self, mock_settings, mock_get_uuid):
        """Test queue_sag_extract_task creates task."""
        from rag.sag.task_queue import queue_sag_extract_task

        mock_get_uuid.return_value = "test-task-id"
        mock_settings.redis = MagicMock()

        task_id = queue_sag_extract_task("doc1", "kb1", "tenant1")

        assert task_id == "test-task-id"

    @patch("rag.sag.task_queue.get_uuid")
    @patch("rag.sag.task_queue.settings")
    def test_queue_sag_rebuild_task(self, mock_settings, mock_get_uuid):
        """Test queue_sag_rebuild_task creates rebuild task."""
        from rag.sag.task_queue import queue_sag_rebuild_task

        mock_get_uuid.return_value = "test-rebuild-id"
        mock_settings.redis = MagicMock()

        task_id = queue_sag_rebuild_task("kb1", "tenant1", ["doc1", "doc2"])

        assert task_id == "test-rebuild-id"


class TestSagRetriever:
    """Tests for rag.sag.retriever module."""

    def test_retriever_init(self):
        """Test SAGRetriever initialization."""
        from rag.sag.retriever import SAGRetriever

        mock_conn = MagicMock()
        retriever = SAGRetriever(mock_conn)

        assert retriever.doc_store_conn == mock_conn

    @pytest.mark.asyncio
    async def test_vector_retrieval_empty_result(self):
        """Test vector retrieval with no results."""
        from rag.sag.retriever import SAGRetriever

        mock_conn = MagicMock()
        mock_conn.search.return_value = []

        retriever = SAGRetriever(mock_conn)

        # Mock embedding
        with patch.object(retriever, "_embed_query", return_value=[0.1] * 128):
            result = await retriever._vector_retrieval(
                "test query", ["tenant1"], ["kb1"], MagicMock(), top_k=10
            )

        assert result == []


class TestSagExtractor:
    """Tests for rag.sag.extractor module."""

    def test_extract_prompt_format(self):
        """Test extraction prompt is properly formatted."""
        from rag.sag.prompts.extract import build_extraction_prompt

        chunk_text = "This is a test chunk about Apple Inc."
        entity_types = ["organization", "person"]

        prompt = build_extraction_prompt(chunk_text, entity_types)

        assert chunk_text in prompt
        assert "organization" in prompt
        assert "person" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
