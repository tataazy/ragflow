# SAG 任务管理模块 规格说明书 (Feature Spec)

## 1. 模块目标
实现 SAG 抽取任务的完整生命周期管理，包括任务创建、调度执行、进度追踪、暂停/恢复/取消、失败重试，复用 RAGFlow 现有的 Redis 任务队列基础设施。

## 2. 非目标
- 不实现分布式任务调度（复用 RAGFlow 现有 task_executor 机制）；
- 不实现任务优先级队列（FIFO 即可）；
- 不实现跨知识库的批量任务编排。

## 3. 任务类型与生命周期

### 3.1 任务类型
- `sag_extract`: 单文档事件-实体抽取（核心任务）
- `sag_rebuild`: 知识库级全量重建（删除旧数据 + 重新抽取所有文档）

### 3.2 状态机
```
pending → running → completed
                  → failed (可重试)
                  → paused (可恢复)
                  → cancelled
```

### 3.3 任务粒度
- `sag_extract`: 一个文档一个任务，任务内按 chunk 并发处理；
- `sag_rebuild`: 一个知识库一个任务，内部拆分为多个文档子任务顺序执行。

## 4. 核心逻辑

### 4.1 任务创建
- 文档解析完成后，由 task_service 检查知识库 SAG 配置并创建任务；
- 任务信息写入 RAGFlow 现有 `task` 表（doc_id, task_type='sag_extract'）；
- 同时在 `sag_extract_checkpoint` 表创建断点记录；
- 任务加入 Redis 队列。

### 4.2 任务执行
- 由 `rag/svr/task_executor.py` 消费队列中的 `sag_extract` 任务；
- 执行流程：
  1. 读取断点，确定待处理的 chunk 列表；
  2. 按并发度创建 worker 协程；
  3. 每个 worker 从队列取 chunk → 调用 LLM 抽取 → 写入 DB → 更新断点；
  4. 所有 chunk 处理完毕后，生成事件向量并写入 doc_store；
  5. 更新任务状态为 completed。

### 4.3 进度追踪
- 进度 = processed_chunk_count / total_chunk_count；
- 每处理完一个 chunk 更新 `document.progress` 字段（与原有解析进度共享）；
- 前端通过轮询文档状态获取进度。

### 4.4 暂停/恢复
- 暂停：设置 checkpoint.status = 'paused'，worker 在下一个 chunk 前检查并退出；
- 恢复：重新创建任务（task_type='sag_extract'），从断点继续；
- 暂停时已完成的 chunk 不重复处理。

### 4.5 取消
- 取消：设置 checkpoint.status = 'cancelled'；
- 清理已产生的不完整数据（可选，默认保留）；
- 更新知识库 sag_task_id 为空。

### 4.6 失败重试
- 单 chunk 抽取失败：自动重试 2 次（间隔 5s、15s）；
- 重试仍失败：记录该 chunk 为 failed，继续处理其他 chunk；
- 任务级失败（如 LLM 服务不可用）：任务状态设为 failed，支持手动重试；
- 失败原因记录在 checkpoint.status 和 task.progress_msg 中。

### 4.7 全量重建（sag_rebuild）
- 删除知识库下所有 sag_events、sag_entities、sag_event_entity 记录；
- 清理 doc_store 中该知识库的事件向量；
- 重置所有 sag_extract_checkpoint 记录；
- 按文档顺序逐个创建 sag_extract 子任务；
- 所有子任务完成后，rebuild 任务标记为 completed。

## 5. 与现有任务系统的集成

### 5.1 复用点
- 复用 `task` 表结构（task_type 字段区分）；
- 复用 Redis 队列（`rag/svr/task_executor.py` 的消费逻辑）；
- 复用 `document.progress` 字段展示进度；
- 复用 `knowledgebase.sag_task_id` 追踪知识库级任务状态。

### 5.2 新增点
- task_executor 中新增 `sag_extract` 任务处理分支；
- 新增 `rag/sag/executor.py` 实现具体抽取逻辑；
- 新增 `sag_extract_checkpoint` 表（见 feature1 §5）。

## 6. 验收标准 (AC)
1. Given 文档解析完成且知识库开启 SAG，When 任务入队，Then task_executor 正确消费并执行抽取；
2. Given 抽取进行中，When 查询文档进度，Then 返回正确的百分比进度；
3. Given 抽取进行中，When 触发暂停，Then 当前 chunk 完成后停止，断点已保存；
4. Given 暂停后恢复，When 任务重新执行，Then 从断点继续，不重复已完成 chunk；
5. Given 单 chunk 抽取失败，When 重试 2 次仍失败，Then 跳过该 chunk 继续处理其他；
6. Given LLM 服务完全不可用，When 任务执行，Then 任务状态为 failed，记录错误原因；
7. Given 触发全量重建，When 执行完成，Then 旧数据已清理，所有文档重新抽取；
8. Given 任务取消，When 取消生效，Then worker 停止处理，任务状态为 cancelled。
