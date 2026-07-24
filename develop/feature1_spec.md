# SAG 结构化抽取模块 规格说明书 (Feature Spec)

## 1. 模块目标
在 RAGFlow 文档解析完成、生成 chunk 之后，异步对每个 chunk 执行事件抽取与实体识别，写入 SAG 关系表并生成事件向量索引。

## 2. 非目标
- 不修改原有 chunk 生成逻辑；
- 不阻塞文档解析主流程，抽取失败不影响文档入库；
- 不做全局图维护（无 PageRank、无社区发现、无实体合并）；
- 不处理非文本类 chunk（纯图片、纯表格 chunk 跳过）。

## 3. 输入输出定义
### 输入
- chunk_id: str，RAGFlow 原有文档块唯一标识
- chunk_content: str，文档块文本内容
- doc_meta: dict，文档元数据（doc_id, kb_id, doc_name）
- kb_config: dict，知识库 SAG 配置（extract_model, entity_types 等）

### 输出
- 写入 sag_events、sag_entities、sag_event_entity 三张表
- 事件向量写入 doc_store（用于后续精确检索）
- 返回抽取状态：success / failed / skipped
- 返回 token_usage: int（本次抽取消耗的 token 数）

## 4. 核心逻辑

### 4.1 抽取流程
1. 复用 RAGFlow LLM 接入层（`LLMBundle`），调用指定抽取模型；
2. 抽取 Prompt 遵循 SAG 原论文规范，每个 chunk 输出 1 条事件 + N 个实体；
3. 实体默认分为 11 类：时间、地点、人物、组织、群体、主题、作品、产品、动作、指标、标签；
4. 抽取任务通过异步队列执行，支持失败重试 2 次；
5. 抽取失败仅记录日志，不抛出异常，不影响 chunk 正常入库。

### 4.2 事件字段规范
每个 chunk 抽取产生的事件包含以下字段：
- **title**: 事件标题（简短概括，≤ 100 字）
- **summary**: 事件摘要（≤ 500 字）
- **content**: 事件完整描述（基于 chunk 原文）
- **category**: 事件分类（由 LLM 自动归类）
- **start_time**: 事件发生时间（可为 null，由 LLM 从文本中提取）
- **parent_id**: 父事件 ID（支持事件层级，首期均为 null）
- **rank**: 事件在文档中的顺序（按 chunk 顺序排列）

### 4.3 实体字段规范
- **entity_name**: 实体名称
- **entity_type**: 实体类型（11 类之一）
- **description**: 实体在当前事件中的作用说明

### 4.4 事件-实体关联规范
- **weight**: 关联权重（默认 1.0，表示实体在该事件中的重要程度）
- **description**: 关联描述（实体在事件中的角色）

### 4.5 实体类型校验与容错
- LLM 返回的实体类型必须在 11 类体系内，否则尝试模糊匹配（如 "location" → "地点"）；
- 无法匹配的实体类型记录警告日志并丢弃该实体，不影响事件入库；
- 处理 LLM 常见的格式错误（如把实体类型写成字段名）。

### 4.6 增量抽取
- 仅对新增/变更的 chunk 执行抽取，已处理的 chunk 不重复抽取；
- 文档重新解析时，先删除该文档关联的旧事件/实体/关联记录，再重新抽取；
- 通过 `sag_extract_checkpoint` 表记录每个文档的抽取进度。

> ⚠️ **实施缺陷标注**：“文档重新解析时先删除旧数据”描述正确，但缺乏执行细节：
> - 未指明清理在哪个方法中执行（应在 `reset_document_for_reparse()` 中，chunk 清空后、新任务触发前）；
> - 未说明 `run_sag_extract` 本身不清理旧数据（它用新 task_id 建 checkpoint，processed_set 从空开始），因此必须在触发抽取前完成清理，否则会产生重复事件。

### 4.7 断点续传
- 每个 chunk 抽取成功后立即持久化断点；
- 任务中断后从最近确认的断点继续，不重复已完成的 chunk；
- 断点信息包含：processed_chunk_ids、event_ids、event_count、token_usage。

### 4.8 并发控制
- 单文档内的 chunk 抽取支持并发（默认 4 并发，可配置 1~20）；
- 使用 asyncio.Semaphore 控制并发度；
- 每个 chunk 抽取独立，单个失败不影响其他 chunk。

### 4.9 事件向量索引
- 抽取完成后，将事件的 title + summary 拼接后调用 embedding 模型生成向量；
- 向量写入 doc_store，使用 `sag_kwd: "event"` 字段区分于普通 chunk；
- 向量维度与知识库配置的 embedding 模型保持一致。

## 5. 数据模型约束

### sag_events 表
| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| id | BIGINT | 主键，自增 | |
| kb_id | VARCHAR(32) | 索引，关联 knowledgebase.id | 所属知识库 |
| doc_id | VARCHAR(32) | 索引，关联 document.id | 所属文档 |
| chunk_id | VARCHAR(64) | 索引，关联原有 chunk | 来源 chunk |
| title | VARCHAR(255) | NOT NULL | 事件标题 |
| summary | TEXT | | 事件摘要 |
| content | TEXT | NOT NULL | 事件完整描述 |
| category | VARCHAR(64) | | 事件分类 |
| start_time | DATETIME | NULL | 事件发生时间 |
| parent_id | BIGINT | NULL | 父事件 ID（首期均为 NULL） |
| rank | INT | DEFAULT 0 | 文档内顺序 |
| event_embedding | BLOB | NULL | 事件向量（冗余存储，主存储在 doc_store） |
| status | VARCHAR(16) | DEFAULT 'completed' | completed / deleted |
| create_time | DATETIME | NOT NULL | |
| update_time | DATETIME | NOT NULL | |

索引：
- `idx_sag_events_kb_doc` (kb_id, doc_id)
- `idx_sag_events_chunk` (chunk_id)
- `idx_sag_events_kb_category` (kb_id, category)

### sag_entities 表
| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| id | BIGINT | 主键，自增 | |
| kb_id | VARCHAR(32) | 索引 | 所属知识库 |
| entity_name | VARCHAR(255) | NOT NULL | 实体名称 |
| entity_type | VARCHAR(32) | NOT NULL | 实体类型（11 类之一） |
| description | TEXT | | 实体描述（最近一次抽取的作用说明） |
| heat | INT | DEFAULT 1 | 关联事件数（频次） |
| create_time | DATETIME | NOT NULL | |
| update_time | DATETIME | NOT NULL | |

索引：
- 唯一索引：`uq_sag_entity` (kb_id, entity_name, entity_type)
- `idx_sag_entities_type` (kb_id, entity_type)

### sag_event_entity 表
| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| id | BIGINT | 主键，自增 | |
| event_id | BIGINT | NOT NULL | 关联 sag_events.id |
| entity_id | BIGINT | NOT NULL | 关联 sag_entities.id |
| weight | FLOAT | DEFAULT 1.0 | 关联权重 |
| description | VARCHAR(512) | | 关联描述 |
| create_time | DATETIME | NOT NULL | |

索引：
- 联合唯一索引：`uq_sag_event_entity` (event_id, entity_id)
- `idx_sag_ee_entity` (entity_id)

### sag_extract_checkpoint 表
| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| id | BIGINT | 主键，自增 | |
| kb_id | VARCHAR(32) | NOT NULL | 知识库 ID |
| doc_id | VARCHAR(32) | NOT NULL | 文档 ID |
| task_id | VARCHAR(32) | NOT NULL | 任务 ID |
| processed_chunk_ids | JSON | | 已处理的 chunk ID 列表 |
| event_ids | JSON | | 已产生的事件 ID 列表 |
| event_count | INT | DEFAULT 0 | 已产生事件数 |
| token_usage | INT | DEFAULT 0 | 已消耗 token 数 |
| status | VARCHAR(16) | DEFAULT 'running' | running / paused / completed / failed |
| create_time | DATETIME | NOT NULL | |
| update_time | DATETIME | NOT NULL | |

索引：
- 唯一索引：`uq_sag_checkpoint` (kb_id, doc_id, task_id)

## 6. Prompt 规范

抽取 Prompt 存放于 `rag/sag/prompts/extract.py`，核心约束：
- 每个 chunk 输出且仅输出 1 条顶级事件；
- 事件必须包含 title、summary、content、category、entities 字段；
- 实体必须严格使用 `{"type": "实体类型", "name": "实体名称", "description": "作用说明"}` 格式；
- 对于书籍、报告、论文等非新闻文档，“事件”也包括可独立理解的观点、事实、定义、机制、因果关系；
- 只有目录、页眉页脚、广告、乱码、纯链接才可返回空结果。

## 7. 验收标准 (AC)
1. Given 正常 chunk 内容，When 执行抽取，Then 生成 1 条事件（含 title/summary/content/category）+ 至少 1 个实体，正确写入三张表；
2. Given 空内容 chunk（< 20 字符），When 执行抽取，Then 直接返回 skipped，不产生脏数据；
3. Given LLM 调用超时，When 执行抽取，Then 自动重试 2 次，最终失败记录日志，不影响主流程；
4. Given 重复实体名称+类型，When 执行抽取，Then 复用已有实体 ID，heat +1，不产生重复数据；
5. Given 文档重新解析，When 触发 SAG 抽取，Then 先删除该文档旧数据，再重新抽取；
6. Given 任务中断，When 恢复执行，Then 从断点继续，不重复已完成的 chunk；
7. Given 抽取完成，When 查询 doc_store，Then 事件向量已写入，可通过 `sag_kwd: "event"` 过滤；
8. Given LLM 返回非法实体类型，When 执行抽取，Then 尝试模糊匹配，无法匹配则丢弃实体并记录警告。