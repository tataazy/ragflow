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
"""SAG event-entity extraction prompt templates.

Provides the system prompt and user prompt for structured extraction
of events and entities from document chunks via LLM.
"""

# ---------------------------------------------------------------------------
# System prompt: defines the extraction task and output schema
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM_PROMPT = """\
你是一个专业的结构化信息抽取引擎。你的任务是从给定的文本片段中抽取一个核心事件（或知识点）及其关联实体。

## 抽取规则

1. 每个文本片段输出且仅输出 **1 条** 顶级事件。
2. 对于书籍、报告、论文等非新闻文档，"事件"也包括可独立理解的观点、事实、定义、机制、因果关系、论证和结论，不要求必须包含日期、人物动作或新闻事件。
3. 只有目录、页眉页脚、广告、乱码、纯链接，或确实与文档主题无关的片段才可返回空结果。
4. 正文只要包含可复用的知识，就至少保留一个有效的事件。

## 事件字段

- **title**: 事件标题（简短概括，≤ 100 字）
- **summary**: 事件摘要（≤ 500 字，提炼核心信息）
- **content**: 事件完整描述（基于原文，保留关键细节）
- **category**: 事件分类（由你自动归类，如：技术、商业、政策、科学、教育、社会、自然等）
- **start_time**: 事件发生时间（ISO 格式字符串，如 "2024-01-15"；无法确定则为 null）

## 实体字段

每个实体必须严格使用以下格式：
{{"type": "实体类型", "name": "实体名称", "description": "在该事件中的作用说明"}}

禁止把实体类型写成字段名，例如不能输出 {{"location":"中东","name":"中东","description":"地区"}}。

## 可用实体类型（{entity_type_count} 类）

{entity_types_list}

## 输出格式

严格输出 JSON，不要输出任何其他文字。格式如下：

```json
{{
  "title": "事件标题",
  "summary": "事件摘要",
  "content": "事件完整描述",
  "category": "分类",
  "start_time": null,
  "entities": [
    {{"type": "实体类型", "name": "实体名称", "description": "作用说明"}}
  ]
}}
```

如果文本片段确实没有可抽取的内容（目录、页眉页脚、广告、乱码、纯链接），输出：

```json
null
```
"""

# ---------------------------------------------------------------------------
# User prompt template: wraps the chunk content
# ---------------------------------------------------------------------------

EXTRACT_USER_PROMPT = """\
请从以下文本片段中抽取事件和实体：

<chunk>
{chunk_text}
</chunk>
"""


def build_extract_messages(
    chunk_text: str,
    entity_types: list[str],
) -> tuple[str, list[dict]]:
    """Build the system and user messages for extraction.

    Args:
        chunk_text: The chunk text to extract from.
        entity_types: List of valid entity type names.

    Returns:
        Tuple of (system_prompt, history_messages).
        history_messages is a list with one user message.
    """
    # Format entity types list
    types_list = "\n".join(f"- {t}" for t in entity_types)

    system = EXTRACT_SYSTEM_PROMPT.format(
        entity_type_count=len(entity_types),
        entity_types_list=types_list,
    )

    history = [{"role": "user", "content": EXTRACT_USER_PROMPT.format(chunk_text=chunk_text)}]

    return system, history
