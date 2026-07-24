# RAGFlow-SAG 融合项目 总规格说明书 (Project Spec)

## 1. 项目目标 (Goal)
1. 在 RAGFlow 系统中植入 SAG 结构化关系检索能力、动态超边检索能力等，补充原有 RAG 链路的多跳推理短板；
2. SAG 作为知识库级别的可选增强能力（非 parser_id），通过 `use_sag` 开关控制；开启后在文档解析生成 chunk 之后异步执行事件-实体抽取；
3. 全量复用 RAGFlow 现有技术底座，不引入新的中间件依赖（数据库复用 MySQL，向量存储复用现有 ES/Infinity，LLM 复用现有接入层）；
4. SAG 能力作为可选项，不影响原有文档解析、检索、生成链路的稳定性；
5. 提供事件-实体知识图谱的 2D/3D 可视化能力，支持按知识库维度浏览和探索。

## 2. 非目标 (Non-Goals)
1. 不重构 RAGFlow 原有解析、检索、工作流核心逻辑；
2. 不支持自定义实体类型扩展，首期复用 SAG 默认的 11 类实体体系；
3. 不替换原有混合检索框架，SAG 走独立的召回通路；
4. 不引入 `zleap-sag` 包作为运行时依赖——仅参考其算法逻辑，用 RAGFlow 技术栈重新实现；
5. 不实现 SAG 的 Agent 对话、MCP Server、OpenAI 兼容端点等应用层功能；
6. 不实现 Universe 探索模式（全局知识宇宙），首期仅实现按知识库维度的图谱浏览。

## 3. 与现有 GraphRAG 的边界

| 维度 | RAGFlow GraphRAG（已有） | SAG 融合（本项目） |
|------|--------------------------|--------------------|
| 数据模型 | 三元组（entity → relation → entity）+ community report | 事件-实体超边（chunk → 1 event + N entities） |
| 构建方式 | 离线全局构建，需整体重建 | 增量逐 chunk 构建，新文档即插即用 |
| 检索方式 | 实体向量召回 + N-hop 路径 + 社区报告 | 种子事件召回 + SQL JOIN 动态超边扩展 |
| 触发条件 | `use_kg=True`（对话/检索配置） | `use_sag=True`（知识库级别开关） |
| 代码位置 | `rag/graphrag/` | `rag/sag/` |
| 存储 | 复用 doc_store（ES/Infinity），`knowledge_graph_kwd` 字段区分 | 复用 MySQL（关系表）+ doc_store（向量），`sag_` 前缀表 |
| 可视化 | 已有知识图谱页面 | 新增事件-实体图谱页面（2D/3D） |
| 共存策略 | 两者可同时开启，互不干扰；检索结果合并后统一重排 | 同左 |

## 4. 整体架构约束

### 4.1 后端
- 与 RAGFlow 保持一致：Python 3.13+，Quart 异步框架，Peewee ORM；
- 所有新增代码存放于 `rag/sag/` 目录下，不得散落到原有模块中；
- API 路由注册于 `api/apps/` 下新增 `sag_api.py`，路由前缀 `/api/v1/sag/`；
- 复用 RAGFlow 的认证中间件（`@token_required`），不引入独立认证体系。

> ⚠️ **实施缺陷标注**：“所有新增代码存放于 rag/sag/”约束过于绝对。实际实现中，删除/清空流程的 SAG 清理 hook 必须添加到现有文件（document_service.py、chunk_api.py、document_api_service.py）中。
> 应区分：
> - **新增业务逻辑**（cleanup 函数、executor、retriever）→ 集中于 `rag/sag/`
> - **接入 hook**（在现有删除/清空流程中调用新逻辑）→ 必须在现有文件中添加，并明确标注“以下现有文件需要添加 SAG 相关调用”

### 4.2 数据库
- 复用 RAGFlow MySQL 实例，新增表前缀统一为 `sag_`；
- 禁止修改 RAGFlow 原有核心表结构；
- 知识库表（`knowledgebase`）新增 `sag_task_id`、`sag_task_finish_at` 两个字段用于追踪 SAG 构建任务状态；
- 知识库表 `parser_config` JSON 中新增 `sag` 配置块（见 §5）。

### 4.3 向量存储
- 事件向量（event_embedding）写入 RAGFlow 现有 doc_store（ES/Infinity/OpenSearch）；
- 使用独立索引或在现有索引中通过 `sag_kwd` 字段区分，避免污染原有 chunk 检索；
- 向量维度与知识库配置的 embedding 模型保持一致。

### 4.4 任务调度
- 复用 RAGFlow 基于 Redis 的异步任务队列（`rag/svr/task_executor.py`）；
- SAG 抽取任务类型为 `sag_extract`，与现有 `graphrag` 任务类型并列；
- 支持任务暂停/恢复/取消，断点信息持久化到 MySQL。

### 4.5 LLM 调用
- 复用 RAGFlow 的 LLM 接入层（`LLMBundle`），不得单独封装模型调用；
- 抽取使用的模型由知识库 `parser_config.sag.extract_model` 指定，默认取租户默认 Chat 模型；
- Prompt 模板存放于 `rag/sag/prompts/` 目录，禁止硬编码。

### 4.6 前端
- 与 RAGFlow 保持一致：React + TypeScript + Vite；
- 图谱可视化组件使用 `@antv/g6`（2D）+ `3d-force-graph`（3D），不引入 SAG 原项目的 Next.js 组件；
- SAG 相关页面作为知识库详情页的子 Tab 呈现，不新增顶级路由；
- 所有 SAG UI 组件存放于 `web/src/pages/sag/` 目录下。

## 5. 配置管理

### 5.1 知识库级配置（存储于 `knowledgebase.parser_config.sag`）
```json
{
  "sag": {
    "enabled": false,
    "extract_model": "",
    "extract_concurrency": 4,
    "chunk_max_tokens": 1000,
    "search_strategy": "multi",
    "search_top_k": 10,
    "hop_num": 1
  }
}
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| enabled | bool | false | SAG 总开关 |
| extract_model | string | "" | 抽取用 LLM 模型 ID，空则取租户默认 |
| extract_concurrency | int | 4 | 单文档 chunk 级抽取并发数（1~20） |
| chunk_max_tokens | int | 1000 | 抽取时单 chunk 最大 token 数 |
| search_strategy | string | "multi" | 检索策略：`vector`(快速) / `multi`(精确) |
| search_top_k | int | 10 | SAG 通路返回结果数 |
| hop_num | int | 1 | 多跳扩展跳数（1~2） |

### 5.2 系统级配置（`conf/service_conf.yaml` 新增 `sag` 段）
```yaml
sag:
  extract_timeout: 120        # 单 chunk 抽取超时（秒）
  extract_max_retries: 2      # 抽取失败重试次数
  search_source_timeout: 10   # 单知识库检索超时（秒）
  search_fallback_vector: true # 精确模式失败是否回退快速模式
  entity_types:               # 11 类默认实体体系
    - 时间
    - 地点
    - 人物
    - 组织
    - 群体
    - 主题
    - 作品
    - 产品
    - 动作
    - 指标
    - 标签
```

## 6. 编码强制规范 (MUST / MUST NOT)
### MUST
- 所有新增公共方法必须包含 Google 风格 Docstring；
- 所有数据库操作必须通过 Peewee ORM 实现，禁止裸写 SQL；
- 所有异常必须捕获并封装为 RAGFlow 统一业务异常（`api.utils.api_utils` 中的错误码体系）；
- 所有 SAG 相关操作必须记录结构化日志，前缀为 `[SAG]`；
- 所有新增 API 必须通过 `@token_required` 认证，并校验知识库归属权限；
- 向量操作必须通过 `settings.docStoreConn` 统一接口，不得直接调用 ES/Infinity SDK。

### MUST NOT
- 禁止修改 RAGFlow 原有核心表结构（`knowledgebase` 新增字段除外）；
- 禁止在主线程中执行耗时的事件抽取操作；
- 禁止硬编码 Prompt、模型名称、配置参数；
- 禁止引入 SAG 原项目的冗余依赖（`zleap-sag`、SQLAlchemy、LanceDB 等）；
- 禁止在 SAG 代码中直接 import `rag/graphrag/` 模块，两者通过上层调度合并结果。

## 7. 验收总标准
1. **功能验收**：知识库开启 SAG 后，文档解析可正常完成事件与实体抽取；检索时可触发多跳扩展；图谱页面可正常渲染；
2. **性能验收**：单 100 页 PDF 文档 SAG 抽取耗时 ≤ 原有解析耗时的 1.5 倍；检索延迟增幅 ≤ 30%；
3. **兼容性验收**：关闭 SAG 开关时，系统行为与原有版本完全一致，无功能退化；
4. **共存验收**：SAG 与 GraphRAG 同时开启时，两路召回结果正确合并，无冲突；
5. **代码验收**：单元测试覆盖率 ≥ 80%，符合 RAGFlow 原有代码风格（ruff check 通过）。

> ⚠️ **实施缺陷标注**：验收标准全部是正向场景，缺乏负向/异常/一致性验证。应补充：
> 6. **数据一致性验收**：删除文档后查询 SAG 事件表，该文档事件数为 0；清空 chunk 后图谱下拉不显示该文档；
> 7. **容错验收**：SAG 清理失败时不阻塞主流程（文档仍正常删除）；
> 8. **自愈验收**：历史残留孤儿数据在打开图谱页时自动清理，无需全量 rebuild。

## 8. 模块依赖关系

```
feature4（知识库集成）
  └── feature5（任务管理）
        └── feature1（结构化抽取）
              └── feature2（检索引擎）
                    └── feature3（图谱可视化）
                          └── feature6（API 层）
                                └── feature7（前端集成）
```

开发顺序：feature4 → feature5 → feature1 → feature2 → feature3 → feature6 → feature7

---

## 9. ⚠️ 实施问题回顾与 Spec 缺陷分析

> 本节记录实际开发中暴露的问题，追溯其在 Spec 中的根因，供后续类似需求引以为戒。

### 9.1 问题清单

| # | 问题现象 | 根因 | 涉及 Spec | 缺陷类型 |
|---|----------|------|-----------|----------|
| 1 | 删除文档/清空 chunk 后 SAG 数据残留 | `cleanup_sag_data_for_docs` 已实现但从未接入真实删除流程 | feature4 §4.4 | 接入点缺失 |
| 2 | 重新解析文档后 SAG 事件重复堆积 | `run_sag_extract` 不清理旧数据，新 checkpoint 从空开始 | feature1 §4.6, feature4 §4.4 | 流程闭环缺失 |
| 3 | SAG 图谱下拉显示已删除文档（名称退化为 doc_id） | `list_sag_docs` 接口未验证文档存在性，孤儿事件残留 | feature3 §3.1（未覆盖） | 查询防御缺失 |
| 4 | 表格 chunk 混入大量 OCR 垃圾字符浪费 token | 无通用 chunk 内容清洗环节 | 全部 feature spec（未覆盖） | 数据质量盲区 |

### 9.2 逐项缺陷分析

#### 缺陷 A：只描述"做什么"，不指定"在哪里做"（接入点缺失）

**现象**：feature4 §4.4 明确写了"文档删除时：级联删除该文档关联的 sag_events"，feature1 AC#5 也写了"Given 文档重新解析，Then 先删除旧数据"。但实现时 `cleanup_sag_data_for_docs` 函数写好了却**从未被调用**。

**根因**：Spec 只描述了期望行为，没有指明具体在哪个函数/方法中接入。RAGFlow 的文档删除有多个入口（`remove_document`、`rm_chunk` delete_all、`reset_document_for_reparse`），Spec 没有逐一列举，实现者只关注了 `rag/sag/` 内部逻辑，忽略了需要在**现有模块**中添加 hook。

**教训**：对于需要接入现有流程的需求，Spec 必须明确列出：
- 需要修改的**具体文件和方法名**
- 接入位置（在哪个步骤之后/之前）
- 如果现有流程有多个入口，必须**逐一列举**

#### 缺陷 B：数据生命周期只有"创建"和"全量重建"，缺乏"增量删除"闭环

**现象**：Spec 详细描述了 SAG 数据的创建（抽取）和全量清理（rebuild），但对"单个文档删除"、"清空 chunk"、"重新解析"这些增量操作的数据一致性只有笼统描述，没有完整的场景枚举。

**根因**：Spec 的数据生命周期视角不完整。只关注了正向流（创建→使用→重建），忽略了逆向流（部分删除→数据一致性）和异常流（清理失败→残留兜底）。

**教训**：涉及持久化数据的功能，Spec 必须包含**完整的数据生命周期矩阵**：
```
| 触发操作 | 对 SAG 数据的影响 | 清理范围 | 接入点 |
| 删除文档 | 删除该文档全部 SAG 数据 | events+associations+orphan entities+checkpoint+vectors | remove_document() |
| 清空 chunk | 同上 | 同上 | rm_chunk(delete_all) |
| 重新解析 | 先清理再重新抽取 | 同上 | reset_document_for_reparse() |
| 删除知识库 | 删除 KB 全部 SAG 数据 | 全量 | remove_kb() |
| 关闭 SAG | 保留数据，停止使用 | 无 | 配置更新 |
```

#### 缺陷 C：查询接口缺乏防御性约束

**现象**：`list_sag_docs`（SAG 图谱文档下拉接口）从 `SagEvent` 聚合 doc_id，用 `doc_name_map.get(doc_id, doc_id)` 反查文档名——文档已删除时退化为显示原始 ID 字符串。

**根因**：
1. 该接口是后期新增的，原始 Spec 中根本没有定义（feature3 只定义了 graph/nodes/expand 三个接口）；
2. 即使定义了，Spec 也没有约束"返回的文档引用必须验证存在性"；
3. 缺乏"如果上游清理失败/遗漏，查询层如何兜底"的防御性设计。

**教训**：
- 所有**引用外部实体的查询接口**，Spec 必须约束：引用目标不存在时的行为（过滤/标记/自愈清理）；
- 不能假设上游数据操作 100% 成功，查询层应有**防御性校验**；
- 新增接口时必须回溯：该接口暴露的数据是否可能因其他操作而失效。

#### 缺陷 D：缺乏"反向影响分析"

**现象**：SAG 引入了新的持久化数据（events/entities/vectors），但 Spec 没有分析"现有系统中哪些操作会破坏这些新数据的一致性"。

**根因**：Spec 只从 SAG 模块自身视角出发（正向：如何创建/检索/展示），没有从**现有系统视角反向审视**（逆向：现有的删除/清空/重解析操作对新数据的影响）。

**教训**：引入新的持久化数据或状态时，Spec 必须包含一节**"与现有操作的交互影响分析"**：
- 列举所有会修改相关实体（文档、chunk、知识库）的现有操作
- 逐一分析对新数据的影响
- 明确每个操作是否需要额外处理

#### 缺陷 E：编码约束"代码集中于 rag/sag/"导致跨模块接入被忽视

**现象**：project_spec §4.1 要求"所有新增代码存放于 `rag/sag/` 目录下，不得散落到原有模块中"。这个约束本意是保持代码整洁，但实际效果是让实现者忽略了需要在 `document_service.py`、`chunk_api.py` 等现有文件中添加 hook。

**教训**：代码组织约束不应阻碍必要的跨模块集成。Spec 应区分：
- **新增逻辑**：集中于 `rag/sag/`（如 cleanup 函数本身）
- **接入 hook**：必须在现有流程中添加调用（如 `remove_document` 中调用 cleanup）
- 并明确标注"以下现有文件需要添加 SAG 相关调用"

### 9.3 Spec 中需标注的具体不足

| 文件 | 章节 | 不足描述 | 应补充内容 |
|------|------|----------|------------|
| feature4_spec.md | §4.4 | 只说"文档删除时级联删除"，未指明接入的具体方法 | 列出 `remove_document()`、`rm_chunk(delete_all)`、`reset_document_for_reparse()` 三个接入点 |
| feature4_spec.md | §4.4 | 未覆盖"清空 chunk 但不删文档"的场景 | 补充 delete_all 场景的 SAG 清理要求 |
| feature1_spec.md | §4.6 | "文档重新解析时先删除旧数据"描述正确但缺乏执行细节 | 明确在 reparse 流程的哪个步骤执行清理 |
| feature3_spec.md | §3.1 | 未定义文档列表接口（`/sag/kb/{kb_id}/docs`） | 补充该接口定义及文档存在性验证约束 |
| feature3_spec.md | 全文 | 未考虑孤儿数据在图谱中的表现 | 补充"文档不存在时其事件不展示"的约束 |
| project_spec.md | §4.1 | "代码集中于 rag/sag/"约束过于绝对 | 区分"新增逻辑"和"接入 hook"，允许在现有文件中添加必要调用 |
| project_spec.md | §7 | 验收标准缺乏数据一致性验证 | 补充"删除文档后查询 SAG 数据应为空"等负向验收 |
| 全部 spec | - | 缺乏输入数据质量约束 | 补充 chunk 内容清洗要求（控制字符、OCR 噪声等） |

---

## 10. 📋 Spec 编写规范与教训（后续类似需求必读）

> 基于本项目实施中暴露的问题，总结以下 Spec 编写原则，适用于任何**向现有系统引入新持久化数据/新子系统**的需求。

### 10.1 数据生命周期必须闭环

**原则**：任何新增的持久化数据，Spec 必须覆盖其完整生命周期：

```
创建 → 使用 → 更新 → 部分删除 → 全量删除 → 异常残留兜底
```

**检查清单**：
- [ ] 数据在哪里创建？由什么操作触发？
- [ ] 数据在哪里被读取/使用？
- [ ] 哪些操作会修改数据？
- [ ] 哪些操作会删除数据？（逐一列举现有系统的删除/清空操作）
- [ ] 删除是否完整？是否有级联？
- [ ] 如果删除失败/遗漏，查询层如何兜底？

### 10.2 接入点必须精确到方法级

**原则**：需要接入现有流程的功能，不能只写"XX 时做 YY"，必须指明：

```markdown
### 接入点清单
| 现有操作 | 文件 | 方法 | 接入位置 | 调用内容 |
|----------|------|------|----------|----------|
| 删除文档 | api/db/services/document_service.py | remove_document() | chunk 删除后 | cleanup_sag_data_for_docs([doc.id], ...) |
| 清空 chunk | api/apps/restful_apis/chunk_api.py | rm_chunk() delete_all 分支 | chunk 删除后 | cleanup_sag_data_for_docs([document_id], ...) |
| 重新解析 | api/apps/services/document_api_service.py | reset_document_for_reparse() | chunk 清空后 | cleanup_sag_data_for_docs([doc.id], ...) |
```

### 10.3 必须做反向影响分析

**原则**：引入新数据后，从**现有系统视角**反向审视：

```markdown
### 反向影响分析
以下现有操作会影响 SAG 数据一致性：
1. 文档删除（remove_document）→ 需清理 SAG 数据
2. 清空 chunk（rm_chunk delete_all）→ 需清理 SAG 数据
3. 重新解析（reset_document_for_reparse）→ 需清理后重新抽取
4. 知识库删除（remove_kb）→ 需全量清理
5. 文档禁用/启用 → 是否影响 SAG 检索？需明确
```

### 10.4 查询接口必须有防御性约束

**原则**：所有引用外部实体的查询接口，必须定义"引用目标不存在时"的行为：

```markdown
### 防御性约束
- list_sag_docs 返回的文档必须验证 Document 表中存在
- 检测到孤儿数据（事件引用的文档已不存在）时，触发自愈清理
- 图谱查询不返回已删除文档的事件
```

### 10.5 验收标准必须包含负向场景

**原则**：AC 不能只有正向验证（"做了 A 应该有 B"），必须包含：
- **负向**："删除后查询应为空"
- **边界**："清理失败时不影响主流程"
- **一致性**："操作后各数据源（MySQL + doc_store）保持一致"
- **幂等**："重复执行不产生副作用"

```markdown
### 负向验收标准示例
- Given 删除文档，When 查询 SAG 事件表，Then 该文档事件数为 0
- Given 清空 chunk，When 打开 SAG 图谱文档下拉，Then 不显示该文档
- Given 删除文档时 SAG 清理异常，When 删除操作完成，Then 文档仍被正常删除（清理失败不阻塞主流程）
- Given 历史残留孤儿数据，When 打开 SAG 图谱页，Then 孤儿数据被自愈清理，不显示
```

### 10.6 新增接口必须回溯数据源完整性

**原则**：新增查询接口时，必须回答：
- 该接口暴露的数据从哪里来？
- 这些数据可能因什么操作而失效？
- 失效时接口应如何表现？
- 是否需要在接口层做防御性校验？

### 10.7 代码组织约束不应阻碍必要集成

**原则**：代码集中管理的约束应区分两类：
- **新增业务逻辑**（如 cleanup 函数、executor、retriever）→ 集中于 `rag/sag/`
- **接入 hook**（在现有删除/清空流程中调用新逻辑）→ 必须在现有文件中添加

Spec 中应明确标注："以下现有文件需要添加 SAG 相关调用"，避免被"代码集中"约束误导。

### 10.8 输入数据质量必须约束

**原则**：依赖上游数据（如 chunk 内容）的功能，必须考虑数据质量：
- 上游数据可能包含什么噪声？（OCR 垃圾字符、控制字符、格式错误）
- 是否需要在消费前清洗？
- 清洗逻辑放在哪一层？（通用层 vs 专用层）

---

## 11. 修订记录

| 日期 | 修订内容 | 原因 |
|------|----------|------|
| 初版 | 项目总规格 + 7 个 feature spec | 需求定义 |
| 本次 | 新增 §9 问题回顾、§10 编写规范、§11 修订记录 | 实施中暴露数据生命周期、接入点、防御性设计等缺陷 |