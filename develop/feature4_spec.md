# SAG 知识库集成模块 规格说明书 (Feature Spec)

## 1. 模块目标
将 SAG 能力作为知识库级别的可选增强集成到 RAGFlow 知识库管理流程中，包括开关控制、配置下发、数据库字段扩展、以及与现有文档解析流程的衔接。

## 2. 非目标
- 不新增 parser_id（SAG 不是解析方法，而是解析后的增强处理）；
- 不修改现有文档解析流程（chunk 生成逻辑不变）；
- 不实现知识库级别的 SAG 数据迁移。

## 3. 集成方式

### 3.1 开关模型
- SAG 作为知识库 `parser_config` 中的可选配置块，通过 `sag.enabled` 控制；
- 开启 SAG 不影响原有 chunk_method（parser_id）的选择；
- 用户可在创建知识库时开启，也可在已有知识库上后续开启/关闭。

### 3.2 与现有流程的关系
```
文档上传 → 文档解析（原有 parser_id）→ 生成 chunks → 入库
                                                    ↓ (if sag.enabled)
                                              异步 SAG 抽取任务
                                                    ↓
                                              事件/实体/向量写入
```

### 3.3 数据库变更
`knowledgebase` 表新增字段：
- `sag_task_id`: VARCHAR(32), NULL — 当前/最近一次 SAG 构建任务 ID
- `sag_task_finish_at`: DATETIME, NULL — SAG 构建完成时间

`parser_config` JSON 新增 `sag` 块（见 project_spec §5.1）。

## 4. 核心逻辑

### 4.1 创建知识库时开启 SAG
- 创建知识库 API 接受 `parser_config.sag` 配置；
- 若 `sag.enabled = true`，知识库创建成功后不立即触发抽取（此时无文档）；
- 配置校验：extract_concurrency ∈ [1, 20]，hop_num ∈ [1, 2]。

### 4.2 已有知识库开启/关闭 SAG
- 通过更新知识库 API 修改 `parser_config.sag.enabled`；
- 开启时：对已有文档触发全量 SAG 抽取任务（异步）；
- 关闭时：不删除已有 SAG 数据，仅停止检索时使用；
- 重新开启时：检查已有数据，仅对缺失的 chunk 补充抽取。

### 4.3 新文档入库触发
- 文档解析完成（chunk 入库）后，检查所属知识库的 `sag.enabled`；
- 若开启，创建 `sag_extract` 类型任务加入队列；
- 任务粒度为单文档（一个文档一个任务）。

### 4.4 文档删除/重新解析
- 文档删除时：级联删除该文档关联的 sag_events、sag_event_entity 记录，清理孤立实体；
- 文档重新解析时：先删除旧 SAG 数据，再触发新的抽取任务；
- 清理 doc_store 中该文档关联的事件向量（`sag_kwd: "event"` + doc_id 过滤）。

> ⚠️ **实施缺陷标注**：本节只描述了“做什么”，未指明“在哪里做”。实际实现时 `cleanup_sag_data_for_docs` 函数已写好但从未被调用。
> 应补充的接入点清单：
>
> | 现有操作 | 文件 | 方法 | 接入位置 |
> |----------|------|------|----------|
> | 删除文档 | api/db/services/document_service.py | `remove_document()` | chunk 删除 + 知识图谱清理后 |
> | 清空 chunk | api/apps/restful_apis/chunk_api.py | `rm_chunk()` delete_all 分支 | chunk 删除后 |
> | 重新解析 | api/apps/services/document_api_service.py | `reset_document_for_reparse()` | chunk 清空后 |
>
> 另外，本节未覆盖“清空 chunk 但不删除文档”的场景（rm_chunk delete_all），也应触发 SAG 清理。

### 4.5 检索时的开关判断
- 对话/检索请求中，检查知识库 `parser_config.sag.enabled`；
- 仅对已开启 SAG 的知识库调用 SAGRetriever；
- 多知识库检索时，仅对开启 SAG 的知识库执行 SAG 召回。

## 5. API 变更

### 5.1 创建知识库
```
POST /api/v1/datasets
Body 新增可选字段:
{
  "parser_config": {
    "sag": {
      "enabled": true,
      "extract_model": "",
      "extract_concurrency": 4,
      "search_strategy": "multi",
      "search_top_k": 10,
      "hop_num": 1
    }
  }
}
```

### 5.2 更新知识库 SAG 配置
```
PUT /api/v1/datasets/{dataset_id}
Body:
{
  "parser_config": {
    "sag": { "enabled": true, ... }
  }
}
```

### 5.3 获取 SAG 状态
```
GET /api/v1/sag/kb/{kb_id}/status
Response:
{
  "enabled": true,
  "task_id": "xxx",
  "task_status": "completed",  // pending / running / paused / completed / failed
  "event_count": 150,
  "entity_count": 80,
  "last_finish_at": "2024-01-01T00:00:00"
}
```

## 6. 前端交互

### 6.1 知识库创建/编辑页
- 在"高级配置"区域新增"SAG 结构化关系"开关；
- 开启后展示子配置项（抽取模型、并发数、检索策略）；
- 配置项带 tooltip 说明。

### 6.2 知识库详情页
- 开启 SAG 后，新增"知识图谱"Tab（feature3 的入口）；
- 展示 SAG 构建状态（进行中/已完成/失败）；
- 提供"重建图谱"按钮（触发全量重新抽取）。

## 7. 验收标准 (AC)
1. Given 创建知识库时开启 SAG，When 上传文档并解析完成，Then 自动触发 SAG 抽取任务；
2. Given 已有知识库开启 SAG，When 保存配置，Then 对已有文档触发全量抽取；
3. Given 关闭 SAG，When 执行检索，Then SAG 通路不生效，原有检索不受影响；
4. Given 删除文档，When 文档关联的 SAG 数据存在，Then 级联清理事件/实体/向量；
5. Given 重新解析文档，When 触发 SAG 抽取，Then 先清理旧数据再重新抽取；
6. Given 多知识库检索，When 部分知识库开启 SAG 部分未开启，Then 仅对开启的执行 SAG 召回；
7. Given SAG 配置参数非法（如 concurrency=0），When 保存，Then 返回参数校验错误。
