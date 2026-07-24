# SAG 功能测试和部署指南

## 概述

本指南说明如何测试和部署包含 SAG（结构化关系检索）功能的 RAGFlow。

## 前置要求

- Docker 和 Docker Compose
- Python 3.13+（本地测试）
- Node.js 20+（前端构建）
- 至少 16GB 内存

## 1. 代码语法检查

### Python 后端

```bash
cd /path/to/ragflow-discover

# 检查 SAG 模块语法
python -m py_compile rag/sag/__init__.py
python -m py_compile rag/sag/config.py
python -m py_compile rag/sag/models.py
python -m py_compile rag/sag/cleanup.py
python -m py_compile rag/sag/task_queue.py
python -m py_compile rag/sag/retriever.py
python -m py_compile api/apps/restful_apis/sag_api.py
```

### 前端

```bash
cd web

# 安装依赖（使用国内镜像）
npm config set registry https://registry.npmmirror.com
npm install

# TypeScript 类型检查
npm run type-check

# 构建
npm run build
```

## 2. Docker 镜像构建（使用国内镜像源）

### 方法一：使用构建参数

```bash
cd /path/to/ragflow-discover

# 使用 NEED_MIRROR=1 启用阿里云/清华镜像
docker build \
    --build-arg NEED_MIRROR=1 \
    -t ragflow-sag:latest \
    -f Dockerfile \
    .
```

### 方法二：使用 Docker Compose

```bash
cd docker

# 修改 .env 文件
# RAGFLOW_IMAGE=ragflow-sag:latest

# 构建并启动
docker compose build --build-arg NEED_MIRROR=1
docker compose --profile cpu up -d
```

### 镜像源说明

Dockerfile 已内置国内镜像支持（`NEED_MIRROR=1`）：
- **APT**: 阿里云 Ubuntu 镜像
- **PyPI**: 阿里云 PyPI 镜像
- **Python**: npmmirror.com Python 构建镜像
- **Git**: Gitee 镜像

## 3. Docker Compose 部署

### 启动服务

```bash
cd docker

# 使用本地构建的镜像
export RAGFLOW_IMAGE=ragflow-sag:latest

# 启动（CPU 模式）
docker compose --profile cpu up -d

# 或 GPU 模式
docker compose --profile gpu up -d
```

### 检查服务状态

```bash
docker compose ps
docker compose logs -f ragflow-cpu
```

### 访问服务

- Web UI: http://localhost
- API: http://localhost:9380

## 4. API 功能验证

### 获取 API Key

1. 访问 http://localhost
2. 注册/登录
3. 进入 "API" 页面获取 API Key

### 测试 SAG API

```bash
# 设置环境变量
export RAGFLOW_URL=http://localhost
export RAGFLOW_API_KEY=your_api_key

# 1. 创建知识库
curl -X POST "$RAGFLOW_URL/api/v1/datasets" \
    -H "Authorization: Bearer $RAGFLOW_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"name": "SAG Test KB", "chunk_method": "naive"}'

# 2. 获取 SAG 配置
curl "$RAGFLOW_URL/api/v1/sag/kb/{kb_id}/config" \
    -H "Authorization: Bearer $RAGFLOW_API_KEY"

# 3. 启用 SAG
curl -X PUT "$RAGFLOW_URL/api/v1/sag/kb/{kb_id}/config" \
    -H "Authorization: Bearer $RAGFLOW_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{"enabled": true, "search_strategy": "multi"}'

# 4. 获取 SAG 状态
curl "$RAGFLOW_URL/api/v1/sag/kb/{kb_id}/status" \
    -H "Authorization: Bearer $RAGFLOW_API_KEY"

# 5. 获取图谱数据
curl "$RAGFLOW_URL/api/v1/sag/kb/{kb_id}/graph?event_limit=100" \
    -H "Authorization: Bearer $RAGFLOW_API_KEY"

# 6. 获取实体列表
curl "$RAGFLOW_URL/api/v1/sag/kb/{kb_id}/entities?page=1&page_size=20" \
    -H "Authorization: Bearer $RAGFLOW_API_KEY"

# 7. 获取事件列表
curl "$RAGFLOW_URL/api/v1/sag/kb/{kb_id}/events?page=1&page_size=20" \
    -H "Authorization: Bearer $RAGFLOW_API_KEY"

# 8. 触发重建
curl -X POST "$RAGFLOW_URL/api/v1/sag/kb/{kb_id}/rebuild" \
    -H "Authorization: Bearer $RAGFLOW_API_KEY"
```

## 5. 前端页面验证

### 访问知识图谱页面

1. 登录 RAGFlow
2. 进入一个已启用 SAG 的知识库
3. 点击左侧导航栏的 "知识图谱" (SAG Graph)
4. 验证：
   - 2D 图谱正常渲染
   - 节点可点击查看详情
   - 工具栏切换视图模式
   - 状态指示器显示正确

### SAG 配置页面

1. 进入知识库设置
2. 找到 SAG 配置区域
3. 验证：
   - 开关可切换
   - 配置项可修改
   - 保存后生效

## 6. 单元测试

### 运行 SAG 模块测试

```bash
cd /path/to/ragflow-discover

# 需要先安装依赖
uv sync --python 3.13 --all-extras

# 运行测试
uv run pytest test/unit_test/rag/sag/ -v
```

## 7. 常见问题

### Docker 构建失败

**问题**: 下载依赖超时

**解决**: 确保使用 `NEED_MIRROR=1` 构建参数

```bash
docker build --build-arg NEED_MIRROR=1 -t ragflow-sag:latest .
```

### 前端构建失败

**问题**: npm install 失败

**解决**: 使用国内镜像

```bash
npm config set registry https://registry.npmmirror.com
npm install
```

### API 返回 404

**问题**: SAG API 端点不存在

**解决**: 确认使用的是包含 SAG 功能的镜像，检查 `api/apps/restful_apis/sag_api.py` 是否存在

### 图谱不显示

**问题**: 知识图谱页面空白

**解决**:
1. 确认知识库已启用 SAG
2. 确认有文档已完成解析
3. 检查 SAG 抽取任务是否完成

## 8. SAG API 端点列表

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/sag/kb/{kb_id}/graph` | 获取图谱切片 |
| GET | `/api/v1/sag/kb/{kb_id}/nodes/{kind}/{id}` | 获取节点详情 |
| POST | `/api/v1/sag/kb/{kb_id}/expand` | 展开节点 |
| GET | `/api/v1/sag/kb/{kb_id}/entities` | 实体列表 |
| GET | `/api/v1/sag/kb/{kb_id}/events` | 事件列表 |
| GET | `/api/v1/sag/kb/{kb_id}/status` | 构建状态 |
| POST | `/api/v1/sag/kb/{kb_id}/rebuild` | 触发重建 |
| POST | `/api/v1/sag/kb/{kb_id}/pause` | 暂停任务 |
| POST | `/api/v1/sag/kb/{kb_id}/resume` | 恢复任务 |
| POST | `/api/v1/sag/kb/{kb_id}/cancel` | 取消任务 |
| GET | `/api/v1/sag/kb/{kb_id}/config` | 获取配置 |
| PUT | `/api/v1/sag/kb/{kb_id}/config` | 更新配置 |
