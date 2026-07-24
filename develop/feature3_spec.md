# SAG 知识图谱可视化模块 规格说明书 (Feature Spec)

## 1. 模块目标
提供按知识库维度的事件-实体知识图谱可视化能力，支持 2D 力导向图和 3D 轨道图两种渲染模式，允许用户浏览事件、实体及其关联关系。

## 2. 非目标
- 不实现全局知识宇宙（Universe）探索模式；
- 不实现图谱编辑（增删改节点/边）；
- 不实现跨知识库的图谱聚合；
- 不实现实时协作编辑。

## 3. 功能范围

### 3.1 图谱数据接口
- 按知识库 ID 获取图谱切片（事件 + 实体 + 关联）；
- 支持分页加载（event_limit, entity_limit 参数）；
- 支持按文档 ID 过滤（仅展示指定文档的事件）；
- 返回格式：`{ events: [], entities: [], associations: [] }`。

> ⚠️ **实施缺陷标注**：
> 1. 本节未定义“文档列表接口”（`GET /sag/kb/{kb_id}/docs`），该接口是前端文档筛选下拉的数据源，实际开发中后期才补充。
> 2. 缺乏防御性约束：文档列表接口必须验证文档在 Document 表中仍存在，孤儿事件（引用已删除文档）应触发自愈清理而非显示原始 doc_id。
> 3. 应补充的接口定义：
>    ```
>    GET /api/v1/sag/kb/{kb_id}/docs
>    Response: { "docs": [{"doc_id": "...", "name": "...", "event_count": N}] }
>    约束：仅返回 Document 表中仍存在的文档；检测到孤儿 doc_id 时调用 cleanup_sag_data_for_docs 自愈清理。
>    ```

### 3.2 2D 力导向图
- 事件节点：圆形，按 category 着色，大小按关联实体数（heat）缩放；
- 实体节点：方形/菱形，按 entity_type 着色；
- 边：事件→实体的 mentions 关系，线宽按 weight 缩放；
- 交互：拖拽、缩放、点击节点展开详情、双击实体展开关联事件；
- 布局：力导向布局（force-directed），支持按 entity_type 分组聚类。

### 3.3 3D 轨道图
- 事件节点沿时间轴（start_time）排列；
- 实体节点围绕关联事件做轨道运动；
- 支持旋转、缩放、平移；
- 点击节点弹出详情面板。

### 3.4 节点详情面板
- 事件详情：title、summary、content、category、start_time、关联实体列表、来源 chunk 原文；
- 实体详情：entity_name、entity_type、description、heat、关联事件列表；
- 点击"查看原文"可跳转到对应 chunk 位置。

### 3.5 时间线视图（可选）
- 按事件 start_time 排列的时间线；
- 支持时间范围筛选；
- 点击时间点展开该时间的事件列表。

## 4. 数据接口规范

### 4.1 获取图谱切片
```
GET /api/v1/sag/kb/{kb_id}/graph
Query Params:
  - event_limit: int (default 200, max 1000)
  - entity_limit: int (default 200, max 1000)
  - doc_ids: str (comma-separated, optional)
Response:
{
  "events": [
    {
      "id": "1",
      "title": "...",
      "summary": "...",
      "category": "...",
      "start_time": "2024-01-01T00:00:00",
      "chunk_id": "...",
      "doc_id": "...",
      "rank": 0,
      "entity_count": 3
    }
  ],
  "entities": [
    {
      "id": "1",
      "name": "...",
      "type": "人物",
      "description": "...",
      "heat": 5
    }
  ],
  "associations": [
    {
      "event_id": "1",
      "entity_id": "1",
      "weight": 1.0,
      "description": "..."
    }
  ],
  "total_events": 150,
  "total_entities": 80
}
```

### 4.2 获取节点详情
```
GET /api/v1/sag/kb/{kb_id}/nodes/{node_kind}/{node_id}
Path Params:
  - node_kind: "event" | "entity"
  - node_id: str
Response: (事件或实体的完整信息 + 关联列表)
```

### 4.3 实体展开（按需加载）
```
POST /api/v1/sag/kb/{kb_id}/expand
Body:
{
  "node_kind": "entity",
  "node_id": "123",
  "limit": 20
}
Response:
{
  "events": [...],
  "associations": [...],
  "has_more": true
}
```

## 5. 前端组件规范

### 5.1 技术选型
- 2D 渲染：`@antv/g6` v5（RAGFlow 已有依赖）；
- 3D 渲染：`3d-force-graph`（新增依赖）；
- 状态管理：复用 RAGFlow 现有 zustand 方案。

### 5.2 组件结构
```
web/src/pages/sag/
├── components/
│   ├── GraphCanvas2D.tsx      # 2D 力导向图
│   ├── GraphCanvas3D.tsx      # 3D 轨道图
│   ├── NodeDetailPanel.tsx    # 节点详情面板
│   ├── GraphToolbar.tsx       # 工具栏（切换2D/3D、筛选、缩放）
│   ├── TimelineView.tsx       # 时间线视图（可选）
│   └── GraphLegend.tsx        # 图例
├── hooks/
│   ├── useGraphData.ts        # 图谱数据获取
│   └── useGraphInteraction.ts # 交互逻辑
├── types.ts                   # 类型定义
└── index.tsx                  # 入口
```

### 5.3 性能约束
- 首屏渲染节点数上限：500（超出时提示用户缩小范围）；
- 2D 图 500 节点时帧率 ≥ 30fps；
- 3D 图 300 节点时帧率 ≥ 24fps；
- 按需加载：初始仅渲染 top-N 事件及其直接关联实体，展开时动态加载。

## 6. 验收标准 (AC)
1. Given 知识库已完成 SAG 抽取，When 打开图谱页面，Then 正确渲染事件-实体关系图；
2. Given 事件数 > 500，When 打开图谱页面，Then 仅渲染前 200 个事件，提示用户缩小范围；
3. Given 点击事件节点，When 查看详情，Then 展示 title/summary/content/关联实体/来源 chunk；
4. Given 双击实体节点，When 展开关联，Then 动态加载该实体关联的事件并渲染新边；
5. Given 切换到 3D 模式，When 渲染完成，Then 可旋转/缩放/平移，节点可点击；
6. Given 知识库未开启 SAG，When 访问图谱页面，Then 展示引导提示，不渲染空图；
7. Given 按文档筛选，When 选择指定文档，Then 仅展示该文档的事件子图。
