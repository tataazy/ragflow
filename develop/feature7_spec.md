# SAG 前端集成模块 规格说明书 (Feature Spec)

## 1. 模块目标
在 RAGFlow 前端（React + TypeScript + Vite）中集成 SAG 功能的用户界面，包括知识库配置入口、图谱可视化页面、构建状态展示、检索结果增强。

## 2. 非目标
- 不新增顶级路由/导航项（SAG 功能嵌入知识库详情页）；
- 不实现独立的 SAG 管理后台；
- 不修改现有对话/检索 UI 的核心交互（仅在结果中增加来源标注）。

## 3. 页面与组件

### 3.1 知识库创建/编辑页 — SAG 配置区
**位置**：知识库设置 → 高级配置区域

**交互**：
- 新增 "SAG 结构化关系" Switch 开关；
- 开启后展开配置面板：
  - 抽取模型选择（下拉，默认"跟随租户默认"）；
  - 抽取并发数（InputNumber，1~20，默认 4）；
  - 检索策略（Radio：快速/精确，默认精确）；
  - 检索 Top-K（InputNumber，1~50，默认 10）；
  - 扩展跳数（Radio：1跳/2跳，默认 1跳）；
- 配置变更即时保存到 `parser_config.sag`。

### 3.2 知识库详情页 — 知识图谱 Tab
**位置**：知识库详情页新增 "知识图谱" Tab（仅 SAG 开启时显示）

**布局**：
```
┌─────────────────────────────────────────────────┐
│ [2D] [3D] [时间线]  │  筛选: [文档▼] [类型▼]  │  ← 工具栏
├─────────────────────────────────────────────────┤
│                                                 │
│              图谱渲染区域                        │  ← 主画布
│                                                 │
├─────────────────────────────────────────────────┤
│  节点详情面板（点击节点时展开）                   │  ← 底部/侧边
└─────────────────────────────────────────────────┘
```

**状态展示**：
- 构建中：显示进度条 + "正在构建知识图谱..."；
- 构建完成：正常渲染图谱；
- 构建失败：显示错误提示 + "重试"按钮；
- 未开启：显示引导页 "开启 SAG 以构建知识图谱"。

### 3.3 检索结果增强
**位置**：对话/搜索的检索结果列表

**增强点**：
- 由 SAG 通路召回的 chunk 标注 "关系召回" 标签；
- 检索结果中展示关联事件标题（如果命中了事件）；
- 不改变原有排序逻辑，仅在 UI 上增加来源标注。

### 3.4 SAG 构建状态指示器
**位置**：知识库列表页 / 知识库详情页顶部

**展示**：
- 小圆点状态指示：绿色(完成) / 蓝色(进行中) / 红色(失败) / 灰色(未开启)；
- hover 展示 tooltip：事件数 / 实体数 / 最近构建时间。

## 4. 组件结构
```
web/src/pages/sag/
├── components/
│   ├── GraphCanvas2D.tsx        # 2D 力导向图（@antv/g6）
│   ├── GraphCanvas3D.tsx        # 3D 轨道图（3d-force-graph）
│   ├── NodeDetailPanel.tsx      # 节点详情面板
│   ├── GraphToolbar.tsx         # 工具栏
│   ├── TimelineView.tsx         # 时间线视图
│   ├── GraphLegend.tsx          # 图例
│   ├── SagConfigPanel.tsx       # SAG 配置面板
│   ├── SagStatusBadge.tsx       # 状态指示器
│   └── SagEmptyState.tsx        # 空状态/引导页
├── hooks/
│   ├── useGraphData.ts          # 图谱数据获取
│   ├── useGraphInteraction.ts   # 交互逻辑
│   ├── useSagConfig.ts          # SAG 配置读写
│   └── useSagStatus.ts          # 构建状态轮询
├── api/
│   └── sag.ts                   # SAG API 调用封装
├── types.ts                     # TypeScript 类型定义
├── constants.ts                 # 常量（颜色映射等）
└── index.tsx                    # 图谱页面入口
```

## 5. 状态管理

### 5.1 图谱状态（组件内 useState/useReducer）
- graphData: { events, entities, associations }
- selectedNode: { kind, id } | null
- viewMode: '2d' | '3d' | 'timeline'
- filters: { doc_ids, entity_types }
- loading, error

### 5.2 配置状态（复用知识库 store）
- 读写 `knowledgebase.parser_config.sag` 字段；
- 通过现有知识库更新 API 保存。

## 6. 新增依赖
| 包名 | 版本 | 用途 |
|------|------|------|
| `3d-force-graph` | ^1.73 | 3D 力导向图渲染 |
| `three` | ^0.170 | 3d-force-graph 的 peer dependency |

注：`@antv/g6` 为 RAGFlow 已有依赖，无需新增。

## 7. 国际化
- 所有 SAG 相关文案支持中英文（复用 RAGFlow i18n 方案）；
- 翻译 key 前缀：`sag.`。

## 8. 验收标准 (AC)
1. Given 知识库编辑页，When 开启 SAG 开关，Then 展示配置面板，保存后配置生效；
2. Given 知识库已开启 SAG 且构建完成，When 打开知识图谱 Tab，Then 正确渲染 2D 图谱；
3. Given 图谱已渲染，When 切换到 3D 模式，Then 正确渲染 3D 图谱，可交互；
4. Given 图谱已渲染，When 点击事件节点，Then 展开详情面板显示完整信息；
5. Given SAG 构建中，When 打开知识图谱 Tab，Then 显示进度条和构建状态；
6. Given 知识库未开启 SAG，When 查看知识库详情，Then 不显示知识图谱 Tab；
7. Given 对话检索结果，When 包含 SAG 召回的 chunk，Then 展示 "关系召回" 标签；
8. Given 前端构建，When 执行 `npm run build`，Then 无 TypeScript 编译错误。
