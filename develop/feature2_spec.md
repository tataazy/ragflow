# SAG 检索引擎模块 规格说明书 (Feature Spec)

## 1. 模块目标
实现 SAG 「种子召回 → SQL 多跳扩展 → 混合重排 → 结果返回」全流程，封装为标准检索器，接入 RAGFlow 多路召回框架。

## 2. 非目标
- 不实现 PageRank 等图算法；
- 不做全局图遍历，最大跳数限制为 2 跳；
- 不实现搜索答案生成（由 RAGFlow 原有对话链路负责）；
- 不替代原有向量/全文检索，仅作为补充召回通路。

## 3. 输入输出定义
### 输入
- query: str 用户查询
- top_k: int 返回结果数量
- hop_num: int 扩展跳数，默认 1
- kb_ids: list 知识库范围
- strategy: str 检索策略（`vector` / `multi`）
- emb_mdl: EmbeddingModel 向量模型实例

### 输出
- List[Chunk] 格式与原有检索器返回格式完全一致，确保可直接送入重排模块
- 每个 chunk 包含标准字段：chunk_id, content_with_weight, doc_id, docnm_kwd, kb_id, similarity, vector_similarity, term_similarity

## 4. 检索策略

### 4.1 快速模式（strategy = "vector"）
仅执行事件向量召回 + chunk 映射，不调用 LLM：
1. query 向量化后与事件向量做相似度匹配（通过 doc_store 中 `sag_kwd: "event"` 过滤）；
2. 将命中事件映射回原始 chunk；
3. 去重后返回。

### 4.2 精确模式（strategy = "multi"）
并行执行三路召回 + SQL 多跳扩展：
1. **实体召回**：从 query 中抽取实体（调用 LLM），通过 SQL 匹配关联事件；
2. **事件向量召回**：query 向量化后与事件向量做相似度匹配；
3. **词法召回**：从 query 提取关键词，对 chunk 内容做全文匹配；
4. **动态扩展**：基于种子事件的关联实体，通过 SQL JOIN 执行 N 跳扩展；
5. **结果映射**：将最终事件集合映射回原始 chunk，去重后返回。

### 4.3 回退机制
- 精确模式超时/失败/空结果时，自动回退为快速模式；
- 回退行为可通过系统配置 `search_fallback_vector` 关闭；
- 回退时记录警告日志。

## 5. 核心逻辑

### 5.1 种子召回
并行执行三路召回（精确模式）：

**实体召回：**
- 调用 LLM 从 query 中抽取实体关键词（复用 RAGFlow LLMBundle）；
- 通过 SQL 在 `sag_entities` 表中匹配实体名称；
- 通过 `sag_event_entity` JOIN 获取关联事件。

**事件向量召回：**
- query 向量化后在 doc_store 中检索 `sag_kwd: "event"` 的向量记录；
- 返回 top_k * 4 个候选事件（过采样，供后续重排筛选）。

**词法召回：**
- 从 query 提取确定性词汇信号（去停用词、去噪声词）；
- 对 chunk 内容做 LIKE/全文匹配；
- 每个关键词最多返回 2 条结果。

### 5.2 动态多跳扩展
```sql
-- 第 N 跳扩展（通过共享实体连接事件）
SELECT e2.* FROM sag_events e2
JOIN sag_event_entity ee2 ON ee2.event_id = e2.id
JOIN sag_event_entity ee1 ON ee1.entity_id = ee2.entity_id
JOIN sag_events e1 ON e1.id = ee1.event_id
WHERE e1.id IN (:seed_event_ids)
  AND e2.id NOT IN (:seed_event_ids)
  AND e2.kb_id IN (:kb_ids)
  AND e2.status = 'completed'
```
- 最大跳数由 `hop_num` 控制（1~2）；
- 每跳扩展结果数上限为 top_k * 3；
- 不会出现无限循环（已访问事件集合排除）。

### 5.3 混合重排
对合并后的候选 chunk 执行混合重排：

```
combined_score = semantic_score * 0.5
               + rank_score * 0.2
               + lexical_score * 0.3
               + exact_match_bonus (0.15 if 词法精确命中)
```

**相关性门控：**
- 计算语义分下限：`semantic_floor = max(0.35, top_score * 0.68)`；
- 有词法信号时：仅保留词法相关或精确命中的结果；
- 无词法信号时：仅保留语义分 ≥ semantic_floor 的结果；
- 过滤 boilerplate 内容（免责声明、广告等）。

### 5.4 结果映射
- 将最终事件集合通过 `sag_events.chunk_id` 映射回原始 chunk；
- 从 doc_store 获取 chunk 完整内容；
- 去重（同一 chunk 可能被多个事件命中，取最高分）；
- 返回格式与 RAGFlow 原有检索器完全一致。

## 6. 接口规范

### 6.1 检索器类
```python
class SAGRetriever:
    """SAG 事件-实体检索器，接入 RAGFlow 多路召回框架。"""

    async def retrieval(
        self,
        question: str,
        tenant_ids: list[str],
        kb_ids: list[str],
        emb_mdl,
        llm_mdl=None,
        top_k: int = 10,
        hop_num: int = 1,
        strategy: str = "multi",
    ) -> dict:
        """返回格式与 Dealer.retrieval() 一致。"""
```

### 6.2 集成点
- 在 `common/settings.py` 中初始化 `sag_retriever` 实例（与 `kg_retriever` 并列）；
- 在 `dialog_service.py` 和 `dataset_api_service.py` 的检索流程中，当 `use_sag=True` 时调用；
- SAG 检索结果与原有检索结果合并后统一重排。

### 6.3 与 GraphRAG 的合并顺序
```
原有检索结果 (Dealer.retrieval)
  + GraphRAG 结果 (KGSearch.retrieval, if use_kg)
  + SAG 结果 (SAGRetriever.retrieval, if use_sag)
  → 合并去重 → 统一重排 → 返回
```

## 7. 性能约束
- 所有 SQL 查询必须添加索引优化，避免全表扫描；
- 单知识库检索超时默认 10 秒（可配置）；
- 多知识库 fan-out 检索时，单源失败不影响整体结果；
- 精确模式的 LLM 实体抽取失败时，回退为纯向量召回。

## 8. 验收标准 (AC)
1. Given 包含明确实体的查询，When 执行 SAG 检索（multi），Then 返回包含关联实体的多跳 chunk 结果；
2. Given hop_num=2，When 执行检索，Then 最多扩展 2 跳，不会出现无限循环；
3. Given 关闭 SAG 开关，When 执行检索，Then 该检索器不生效，不产生额外开销；
4. Given 100 万级事件表，When 执行单跳检索，Then 响应时间 ≤ 200ms；
5. Given 精确模式 LLM 超时，When 执行检索，Then 自动回退快速模式，仍返回结果；
6. Given SAG + GraphRAG 同时开启，When 执行检索，Then 两路结果正确合并，无重复 chunk；
7. Given strategy="vector"，When 执行检索，Then 不调用 LLM，仅用向量召回，延迟 ≤ 100ms；
8. Given 多知识库检索，When 其中一个知识库失败，Then 其他知识库结果正常返回。