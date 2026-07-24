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
"""SAG configuration utilities.

Reads system-level SAG config from service_conf.yaml and provides
validation helpers for knowledgebase-level parser_config.sag blocks.
"""

import logging
from typing import Optional

from common.config_utils import get_base_config

logger = logging.getLogger(__name__)

# System-level defaults (overridable via service_conf.yaml `sag:` section)
_DEFAULTS = {
    "extract_timeout": 120,
    "extract_max_retries": 2,
    "search_source_timeout": 10,
    "search_fallback_vector": True,
    "entity_types": [
        "时间", "地点", "人物", "组织", "群体",
        "主题", "作品", "产品", "动作", "指标", "标签",
    ],
}

# Knowledgebase-level config defaults (parser_config.sag)
KB_DEFAULTS = {
    "enabled": False,
    "extract_model": "",
    "extract_concurrency": 4,
    "chunk_max_tokens": 1000,
    "search_strategy": "multi",
    "search_top_k": 10,
    "hop_num": 1,
}

# Valid search strategies
VALID_STRATEGIES = ("vector", "multi")


def get_sag_system_config() -> dict:
    """Load system-level SAG configuration from service_conf.yaml.

    Returns:
        dict: Merged config with defaults for missing keys.
    """
    conf = get_base_config("sag", {})
    merged = {**_DEFAULTS, **conf}
    return merged


def get_entity_types() -> list[str]:
    """Return the configured entity type list (11 types by default)."""
    return get_sag_system_config().get("entity_types", _DEFAULTS["entity_types"])


def validate_kb_sag_config(sag_config: dict) -> Optional[str]:
    """Validate a knowledgebase-level SAG config block.

    Args:
        sag_config: The `parser_config.sag` dict to validate.

    Returns:
        None if valid, or an error message string describing the first
        validation failure.
    """
    if not isinstance(sag_config, dict):
        return "sag config must be a JSON object"

    concurrency = sag_config.get("extract_concurrency", KB_DEFAULTS["extract_concurrency"])
    if not isinstance(concurrency, int) or concurrency < 1 or concurrency > 20:
        return "extract_concurrency must be an integer in [1, 20]"

    hop_num = sag_config.get("hop_num", KB_DEFAULTS["hop_num"])
    if not isinstance(hop_num, int) or hop_num < 1 or hop_num > 2:
        return "hop_num must be an integer in [1, 2]"

    strategy = sag_config.get("search_strategy", KB_DEFAULTS["search_strategy"])
    if strategy not in VALID_STRATEGIES:
        return f"search_strategy must be one of {VALID_STRATEGIES}"

    top_k = sag_config.get("search_top_k", KB_DEFAULTS["search_top_k"])
    if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
        return "search_top_k must be an integer in [1, 100]"

    chunk_max_tokens = sag_config.get("chunk_max_tokens", KB_DEFAULTS["chunk_max_tokens"])
    if not isinstance(chunk_max_tokens, int) or chunk_max_tokens < 100 or chunk_max_tokens > 8000:
        return "chunk_max_tokens must be an integer in [100, 8000]"

    return None


def normalize_kb_sag_config(sag_config: dict) -> dict:
    """Fill missing keys with defaults and return a normalized config.

    Args:
        sag_config: Partial SAG config from user input.

    Returns:
        Complete config dict with all keys populated.
    """
    normalized = {**KB_DEFAULTS}
    for key in KB_DEFAULTS:
        if key in sag_config:
            normalized[key] = sag_config[key]
    return normalized


def is_sag_enabled(parser_config: dict) -> bool:
    """Check whether SAG is enabled for a knowledge base.

    Args:
        parser_config: The knowledgebase's parser_config dict.

    Returns:
        True if sag.enabled is truthy.
    """
    if not isinstance(parser_config, dict):
        return False
    sag = parser_config.get("sag")
    if not isinstance(sag, dict):
        return False
    return bool(sag.get("enabled", False))
