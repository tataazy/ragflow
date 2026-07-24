# SAG API 层模块 规格说明书 (Feature Spec)

## 1. 模块目标
定义 SAG 功能的所有 REST API 端点，包括图谱查询、任务管理、配置管理，统一注册于 RAGFlow 路由体系，复用现有认证和权限中间件。

## 2. 非目标
- 不实现独立的认证体系（复用 RAGFlow `@token_required`）；
- 不实现 WebSocket 实时推送（进度通过轮询获取）；
- 不实现 OpenAPI/Swagger 文档自动生成（RAGFlow 现有方案）。

## 3. 路由设计

所有 SAG API 注册于 `api/apps/sag_api.py`，路由前缀 `/api/v1/sag/`。

### 3.1 图谱查询

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/sag/kb/{kb_id}/graph` | 获取知识库图谱切片 |
| GET | `/api/v1/sag/kb/{kb_id}/nodes/{kind}/{node_id}` | 获取节点详情 |
| POST | `/api/v1/sag/kb/{kb_id}/expand` | 按需展开节点关联 |
| GET | `/api/v1/sag/kb/{kb_id}/entities` | 获取实体列表（支持类型过滤） |
| GET | `/api/v1/sag/kb/{kb_id}/events` | 获取事件列表（支持分页/分类过滤） |

### 3.2 任务管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/sag/kb/{kb_id}/status` | 获取 SAG 构建状态 |
| POST | `/api/v1/sag/kb/{kb_id}/rebuild` | 触发全量重建 |
| POST | `/api/v1/sag/kb/{kb_id}/pause` | 暂停当前任务 |
| POST | `/api/v1/sag/kb/{kb_id}/resume` | 恢复暂停的任务 |
| POST | `/api/v1/sag/kb/{kb_id}/cancel` | 取消当前任务 |

### 3.3 配置管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/sag/kb/{kb_id}/config` | 获取 SAG 配置 |
| PUT | `/api/v1/sag/kb/{kb_id}/config` | 更新 SAG 配置 |

### 3.4 检索（内部调用，非独立端点）

SAG 检索不暴露独立 API，而是集成到现有检索流程中：
- `POST /api/v1/datasets/{id}/search` 中 `use_sag=true` 时触发；
- 对话流程中 `prompt_config.use_sag=true` 时触发。

## 4. 通用规范

### 4.1 认证
- 所有端点使用 `@token_required` 装饰器；
- 从 token 中解析 tenant_id，校验知识库归属。

### 4.2 权限
- 知识库 owner 或 team 成员（按 knowledgebase.permission 判断）；
- 写操作（rebuild/pause/resume/cancel/config）仅 owner 可执行。

### 4.3 错误码
复用 RAGFlow 统一错误码体系：
- 404: 知识库不存在 / 节点不存在
- 403: 无权限
- 400: 参数校验失败
- 409: 任务状态冲突（如已完成的任务不能暂停）
- 500: 内部错误

### 4.4 分页
列表接口统一使用 `page` + `page_size` 参数：
```json
{
  "page": 1,
  "page_size": 20,
  "total": 150,
  "data": [...]
}
```

## 5. 请求/响应示例

### 5.1 获取图谱切片
```
GET /api/v1/sag/kb/{kb_id}/graph?event_limit=200&entity_limit=200

Response 200:
{
  "code": 0,
  "data": {
    "events": [...],
    "entities": [...],
    "associations": [...],
    "total_events": 150,
    "total_entities": 80
  }
}
```

### 5.2 触发重建
```
POST /api/v1/sag/kb/{kb_id}/rebuild

Response 202:
{
  "code": 0,
  "data": {
    "task_id": "xxx",
    "message": "SAG rebuild task created"
  }
}
```

### 5.3 获取状态
```
GET /api/v1/sag/kb/{kb_id}/status

Response 200:
{
  "code": 0,
  "data": {
    "enabled": true,
    "task_id": "xxx",
    "task_status": "running",
    "progress": 0.65,
    "event_count": 100,
    "entity_count": 55,
    "token_usage": 125000,
    "last_finish_at": null
  }
}
```

## 6. 代码结构
```
api/apps/sag_api.py          # 路由注册 + 请求处理
rag/sag/
├── __init__.py
├── service.py               # 业务逻辑层（图谱查询、配置管理）
├── retriever.py             # SAGRetriever 检索器
├── executor.py              # 抽取任务执行器
├── extractor.py             # LLM 抽取逻辑
├── models.py                # Peewee ORM 模型
├── prompts/
│   ├── __init__.py
│   └── extract.py           # 抽取 Prompt 模板
└── utils.py                 # 工具函数
```

## 7. 验收标准 (AC)
1. Given 有效 token 和知识库 ID，When 调用图谱接口，Then 返回正确的图谱数据；
2. Given 无效 token，When 调用任意 SAG API，Then 返回 401；
3. Given 非 owner 用户，When 调用写操作，Then 返回 403；
4. Given 知识库不存在，When 调用 SAG API，Then 返回 404；
5. Given 任务已完成，When 调用暂停接口，Then 返回 409 状态冲突；
6. Given 参数非法（event_limit > 1000），When 调用图谱接口，Then 返回 400 参数错误；
7. Given 所有 API，When 执行完成，Then 响应格式符合 RAGFlow 统一 `{code, data}` 结构。
