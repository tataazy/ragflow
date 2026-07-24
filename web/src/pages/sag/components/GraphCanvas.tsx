// SAG event-entity graph rendered with @xyflow/react.
//
// Ported from the original SAG `source-graph` visualization: card-style nodes,
// tree / radial / force layouts, and a d3-force simulation with a collision
// force so that filtered sub-graphs spread out instead of collapsing into an
// unreadable blob. Zoom is clamped (fitView minZoom) so a small filtered result
// never shrinks to invisibility.
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  Position,
  ReactFlow,
  useNodesInitialized,
  useNodesState,
  useReactFlow,
  type Edge,
  type Node,
  type NodeProps,
  type NodeTypes,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import {
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from 'd3-force';
import {
  ListTree,
  Orbit,
  RotateCcw,
  Share2,
  Sparkles,
  Users,
  type LucideIcon,
} from 'lucide-react';
import * as React from 'react';
import { useTranslation } from 'react-i18next';
import { cn } from '@/lib/utils';
import { DEFAULT_ENTITY_COLOR, ENTITY_TYPE_COLORS } from '../constants';
import type { SagGraphData, SelectedNode } from '../types';

export type GraphLayout = 'force' | 'radial' | 'tree';
export type GraphPoint = { x: number; y: number };
export type GraphKind = 'event' | 'entity';
type GraphSide = 'top' | 'right' | 'bottom' | 'left';

interface EventEntityRelation {
  id: string;
  eventId: string;
  entityId: string;
}

interface EventEntityGraphSlice {
  events: SagGraphData['events'];
  entities: SagGraphData['entities'];
  relations: EventEntityRelation[];
}

interface GraphNodeData extends Record<string, unknown> {
  kind: GraphKind;
  originalId: string;
  label: string;
  subtitle?: string;
  entityColor?: string;
  isSelected?: boolean;
}

export const GRAPH_EDGE_TYPE: Record<GraphLayout, 'straight' | 'smoothstep'> = {
  radial: 'straight',
  tree: 'smoothstep',
  force: 'straight',
};

const LAYOUT_STORAGE_KEY = 'sag:graph-layout';

export function eventEntityNodeId(kind: GraphKind, id: string) {
  return `${kind}:${id}`;
}

/** Normalize the API graph into the event-entity relations actually rendered. */
function sliceEventEntityGraph(data: SagGraphData): EventEntityGraphSlice {
  const eventIds = new Set(data.events.map((event) => event.id));
  const entityIds = new Set(data.entities.map((entity) => entity.id));
  const seen = new Set<string>();
  const relations: EventEntityRelation[] = [];

  data.associations.forEach((assoc) => {
    if (!eventIds.has(assoc.event_id) || !entityIds.has(assoc.entity_id)) return;
    const key = `${assoc.event_id}:${assoc.entity_id}`;
    if (seen.has(key)) return;
    seen.add(key);
    relations.push({ id: `mention:${key}`, eventId: assoc.event_id, entityId: assoc.entity_id });
  });

  return { events: data.events, entities: data.entities, relations };
}

const KIND_META: Record<
  GraphKind,
  { titleKey: 'event' | 'entity'; width: number; className: string; header: string }
> = {
  event: {
    titleKey: 'event',
    width: 196,
    className: 'border-amber-500/40',
    header: 'bg-amber-500/12 text-amber-700 dark:text-amber-300',
  },
  entity: {
    titleKey: 'entity',
    width: 148,
    className: 'border-dashed',
    header: 'bg-violet-500/10 text-violet-700 dark:text-violet-300',
  },
};

function GraphNode({ data }: NodeProps) {
  const { t } = useTranslation();
  const node = data as GraphNodeData;
  const meta = KIND_META[node.kind];
  const isEntity = node.kind === 'entity';
  const borderColor = isEntity
    ? node.entityColor || DEFAULT_ENTITY_COLOR
    : undefined;
  return (
    <div
      className={cn(
        'relative cursor-grab overflow-hidden rounded-lg border bg-card shadow-sm transition-[box-shadow,border-color] hover:shadow-md active:cursor-grabbing',
        meta.className,
        node.isSelected && 'ring-2 ring-primary/60 ring-offset-1',
      )}
      style={{ width: meta.width, ...(borderColor ? { borderColor } : {}) }}
    >
      <GraphHandles />
      <div className={cn('flex items-center gap-1.5 px-2 py-1 text-[10px] font-medium', meta.header)}>
        {isEntity ? (
          <span
            className="size-2 shrink-0 rounded-full"
            style={{ backgroundColor: node.entityColor || DEFAULT_ENTITY_COLOR }}
          />
        ) : (
          <Sparkles className="size-3 shrink-0" />
        )}
        {t(`sag.graph.${meta.titleKey}`)}
        {isEntity && <Users className="ml-auto size-3 shrink-0 opacity-60" />}
      </div>
      <div className="px-2.5 py-2">
        <div
          className="line-clamp-2 text-xs font-medium leading-snug text-foreground"
          title={node.label}
        >
          {node.label}
        </div>
        {node.subtitle && (
          <div className="mt-1 truncate text-[10px] text-muted-foreground" title={node.subtitle}>
            {node.subtitle}
          </div>
        )}
      </div>
    </div>
  );
}

const nodeTypes: NodeTypes = { sagNode: GraphNode };

function makeNodes(
  slice: EventEntityGraphSlice,
  positions: Map<string, GraphPoint>,
  selectedNode: SelectedNode | null,
  translateType: (type: string) => string,
): Node[] {
  const fallback = { x: 0, y: 0 };
  return [
    ...slice.events.map((event) => ({
      id: eventEntityNodeId('event', event.id),
      type: 'sagNode',
      position: positions.get(eventEntityNodeId('event', event.id)) ?? fallback,
      data: {
        kind: 'event',
        originalId: event.id,
        label: event.title,
        subtitle: event.category || undefined,
        isSelected:
          selectedNode?.kind === 'event' && selectedNode.id === event.id,
      } satisfies GraphNodeData,
    })),
    ...slice.entities.map((entity) => ({
      id: eventEntityNodeId('entity', entity.id),
      type: 'sagNode',
      position: positions.get(eventEntityNodeId('entity', entity.id)) ?? fallback,
      data: {
        kind: 'entity',
        originalId: entity.id,
        label: entity.name,
        subtitle: entity.type ? translateType(entity.type) : undefined,
        entityColor: ENTITY_TYPE_COLORS[entity.type] || DEFAULT_ENTITY_COLOR,
        isSelected:
          selectedNode?.kind === 'entity' && selectedNode.id === entity.id,
      } satisfies GraphNodeData,
    })),
  ];
}

const SIDES: Array<{ side: GraphSide; position: Position }> = [
  { side: 'top', position: Position.Top },
  { side: 'right', position: Position.Right },
  { side: 'bottom', position: Position.Bottom },
  { side: 'left', position: Position.Left },
];

export function GraphHandles() {
  return (
    <>
      {SIDES.map(({ side, position }) => (
        <React.Fragment key={side}>
          <Handle
            id={`target-${side}`}
            type="target"
            position={position}
            isConnectable={false}
            className="!size-2 !border-0 !bg-transparent !opacity-0"
          />
          <Handle
            id={`source-${side}`}
            type="source"
            position={position}
            isConnectable={false}
            className="!size-2 !border-0 !bg-transparent !opacity-0"
          />
        </React.Fragment>
      ))}
    </>
  );
}

function sideFromVector(from: GraphPoint, to: GraphPoint): GraphSide {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  if (Math.abs(dx) >= Math.abs(dy)) return dx >= 0 ? 'right' : 'left';
  return dy >= 0 ? 'bottom' : 'top';
}

function oppositeSide(side: GraphSide): GraphSide {
  if (side === 'top') return 'bottom';
  if (side === 'bottom') return 'top';
  if (side === 'left') return 'right';
  return 'left';
}

export function graphEdgeHandles(from: GraphPoint, to: GraphPoint) {
  const side = sideFromVector(from, to);
  return {
    sourceHandle: `source-${side}`,
    targetHandle: `target-${oppositeSide(side)}`,
  };
}

function makeEdges(
  slice: EventEntityGraphSlice,
  positions: Map<string, GraphPoint>,
  layout: GraphLayout,
): Edge[] {
  return slice.relations.map((relation) => {
    const source = eventEntityNodeId('event', relation.eventId);
    const target = eventEntityNodeId('entity', relation.entityId);
    const from = positions.get(source) ?? { x: 0, y: 0 };
    const to = positions.get(target) ?? { x: 0, y: 0 };
    return {
      id: relation.id,
      source,
      target,
      ...graphEdgeHandles(from, to),
      type: GRAPH_EDGE_TYPE[layout],
      interactionWidth: 14,
      style: {
        stroke: 'hsl(263 55% 58% / 0.32)',
        strokeWidth: 1.25,
      },
    };
  });
}

function linkedEventPositions(
  slice: EventEntityGraphSlice,
  positions: Map<string, GraphPoint>,
  entityId: string,
) {
  return slice.relations
    .filter((relation) => relation.entityId === entityId)
    .map((relation) => positions.get(eventEntityNodeId('event', relation.eventId)))
    .filter((position): position is GraphPoint => Boolean(position));
}

function buildTreePositions(slice: EventEntityGraphSlice, locale: string) {
  const positions = new Map<string, GraphPoint>();
  const eventGap = 260;
  const eventStart = -((slice.events.length - 1) * eventGap) / 2;
  slice.events.forEach((event, index) => {
    positions.set(eventEntityNodeId('event', event.id), {
      x: eventStart + index * eventGap,
      y: 0,
    });
  });

  const entities = slice.entities
    .map((entity) => {
      const linked = linkedEventPositions(slice, positions, entity.id);
      return {
        entity,
        desiredX: linked.length
          ? linked.reduce((sum, position) => sum + position.x, 0) / linked.length
          : 0,
      };
    })
    .sort(
      (a, b) =>
        a.desiredX - b.desiredX || a.entity.name.localeCompare(b.entity.name, locale),
    );
  let previousX = -Infinity;
  entities.forEach(({ entity, desiredX }) => {
    const x = Math.max(desiredX, previousX + 174);
    previousX = x;
    positions.set(eventEntityNodeId('entity', entity.id), { x, y: 310 });
  });
  if (entities.length > 0) {
    const first =
      positions.get(eventEntityNodeId('entity', entities[0].entity.id))?.x ?? 0;
    const last =
      positions.get(
        eventEntityNodeId('entity', entities[entities.length - 1].entity.id),
      )?.x ?? 0;
    const offset = (first + last) / 2;
    entities.forEach(({ entity }) => {
      const id = eventEntityNodeId('entity', entity.id);
      const position = positions.get(id);
      if (position) positions.set(id, { ...position, x: position.x - offset });
    });
  }
  return positions;
}

function normalizeAngle(angle: number) {
  const tau = Math.PI * 2;
  return ((angle % tau) + tau) % tau;
}

function buildRadialPositions(slice: EventEntityGraphSlice, locale: string) {
  const tau = Math.PI * 2;
  const positions = new Map<string, GraphPoint>();
  const eventCount = slice.events.length;
  const eventRadius =
    eventCount <= 1 ? 0 : Math.max(250, (eventCount * 220 * 1.16) / tau);
  const eventAngles = new Map<string, number>();

  slice.events.forEach((event, index) => {
    const angle = -Math.PI / 2 + (index * tau) / Math.max(eventCount, 1);
    eventAngles.set(event.id, angle);
    positions.set(eventEntityNodeId('event', event.id), {
      x: Math.cos(angle) * eventRadius,
      y: Math.sin(angle) * eventRadius,
    });
  });

  const entities = slice.entities
    .map((entity, index) => {
      const angles = slice.relations
        .filter((relation) => relation.entityId === entity.id)
        .map((relation) => eventAngles.get(relation.eventId))
        .filter((angle): angle is number => angle != null);
      const desired = angles.length
        ? Math.atan2(
            angles.reduce((sum, angle) => sum + Math.sin(angle), 0),
            angles.reduce((sum, angle) => sum + Math.cos(angle), 0),
          )
        : -Math.PI / 2 + (index * tau) / Math.max(slice.entities.length, 1);
      return { entity, angle: normalizeAngle(desired) };
    })
    .sort(
      (a, b) => a.angle - b.angle || a.entity.name.localeCompare(b.entity.name, locale),
    );

  const minGap = Math.min(0.18, (tau * 0.88) / Math.max(entities.length, 1));
  let previousAngle = -Infinity;
  entities.forEach((item) => {
    item.angle = Math.max(item.angle, previousAngle + minGap);
    previousAngle = item.angle;
  });
  if (
    entities.length &&
    entities[entities.length - 1].angle - entities[0].angle > tau - minGap
  ) {
    const start = entities[0].angle;
    entities.forEach((item, index) => {
      item.angle = start + (index * tau) / entities.length;
    });
  }

  const entityRadius = Math.max(360, eventRadius + 390);
  entities.forEach(({ entity, angle }) => {
    positions.set(eventEntityNodeId('entity', entity.id), {
      x: Math.cos(angle) * entityRadius,
      y: Math.sin(angle) * entityRadius,
    });
  });
  return positions;
}

function collisionRadius(kind: GraphKind) {
  return kind === 'event' ? 126 : 80;
}

function buildNetwork(
  slice: EventEntityGraphSlice,
  layout: GraphLayout,
  locale: string,
  selectedNode: SelectedNode | null,
  translateType: (type: string) => string,
): { nodes: Node[]; edges: Edge[] } {
  if (layout === 'tree') {
    const positions = buildTreePositions(slice, locale);
    return {
      nodes: makeNodes(slice, positions, selectedNode, translateType),
      edges: makeEdges(slice, positions, layout),
    };
  }

  const radialPositions = buildRadialPositions(slice, locale);
  const radialNodes = makeNodes(slice, radialPositions, selectedNode, translateType);
  if (layout === 'radial') {
    return { nodes: radialNodes, edges: makeEdges(slice, radialPositions, layout) };
  }

  type SimNode = SimulationNodeDatum & { id: string; kind: GraphKind };
  const degree = new Map<string, number>();
  slice.relations.forEach((relation) => {
    const event = eventEntityNodeId('event', relation.eventId);
    const entity = eventEntityNodeId('entity', relation.entityId);
    degree.set(event, (degree.get(event) ?? 0) + 1);
    degree.set(entity, (degree.get(entity) ?? 0) + 1);
  });
  const simNodes: SimNode[] = radialNodes.map((node) => ({
    id: node.id,
    kind: (node.data as GraphNodeData).kind,
    x: node.position.x * 0.48,
    y: node.position.y * 0.48,
  }));
  const seedEdges = makeEdges(slice, radialPositions, 'force');
  const simLinks: SimulationLinkDatum<SimNode>[] = seedEdges.map((edge) => ({
    source: edge.source,
    target: edge.target,
  }));
  // The force simulation is quadratic and becomes the dominant UI cost on
  // larger graphs. The deterministic radial layout stays readable and avoids a
  // long main-thread stall once the working set crosses this threshold.
  if (simNodes.length > 280) {
    return { nodes: radialNodes, edges: makeEdges(slice, radialPositions, layout) };
  }

  const simulation = forceSimulation<SimNode>(simNodes)
    .force(
      'link',
      forceLink<SimNode, SimulationLinkDatum<SimNode>>(simLinks)
        .id((node) => node.id)
        .distance((link) => {
          const source =
            typeof link.source === 'object' ? link.source.id : String(link.source);
          const target =
            typeof link.target === 'object' ? link.target.id : String(link.target);
          return (
            164 +
            Math.min(64, Math.max(degree.get(source) ?? 1, degree.get(target) ?? 1) * 3.5)
          );
        })
        .strength(0.46),
    )
    .force(
      'charge',
      forceManyBody<SimNode>()
        .strength((node) => (node.kind === 'event' ? -430 : -135))
        .distanceMax(1500),
    )
    .force(
      'collide',
      forceCollide<SimNode>((node) => collisionRadius(node.kind))
        .strength(0.96)
        .iterations(4),
    )
    .force(
      'x',
      forceX<SimNode>(0).strength((node) => (node.kind === 'event' ? 0.028 : 0.018)),
    )
    .force(
      'y',
      forceY<SimNode>(0).strength((node) => (node.kind === 'event' ? 0.028 : 0.018)),
    )
    .stop();
  for (let index = 0; index < 340; index += 1) simulation.tick();
  const positions = new Map(
    simNodes.map((node) => [node.id, { x: node.x ?? 0, y: node.y ?? 0 }]),
  );
  return {
    nodes: makeNodes(slice, positions, selectedNode, translateType),
    edges: makeEdges(slice, positions, layout),
  };
}

function FitViewOnChange({
  nodes,
  edges,
  refreshKey,
  padding,
  minZoom,
}: {
  nodes: Node[];
  edges: Edge[];
  refreshKey: unknown;
  padding: number;
  minZoom: number;
}) {
  const { fitView } = useReactFlow();
  const initialized = useNodesInitialized();
  React.useEffect(() => {
    if (!initialized || nodes.length === 0) return;
    let frame = 0;
    const timers: number[] = [];
    const fit = (duration = 260) => {
      fitView({ padding, duration, minZoom, maxZoom: 1.05 });
    };
    frame = window.requestAnimationFrame(() => {
      fit(0);
      timers.push(window.setTimeout(() => fit(), 120));
      timers.push(window.setTimeout(() => fit(), 360));
    });
    return () => {
      window.cancelAnimationFrame(frame);
      timers.forEach((timer) => window.clearTimeout(timer));
    };
  }, [edges, fitView, initialized, minZoom, nodes, padding, refreshKey]);
  return null;
}

const LAYOUT_LABEL_KEY: Record<GraphLayout, string> = {
  force: 'sag.graph.forceLayout',
  radial: 'sag.graph.radialLayout',
  tree: 'sag.graph.treeLayout',
};

const LAYOUT_ICON: Record<GraphLayout, LucideIcon> = {
  force: Share2,
  radial: Orbit,
  tree: ListTree,
};

interface GraphCanvasProps {
  data: SagGraphData | undefined;
  selectedNode: SelectedNode | null;
  onNodeClick: (kind: GraphKind, id: string) => void;
  className?: string;
}

const GraphCanvas = ({
  data,
  selectedNode,
  onNodeClick,
  className,
}: GraphCanvasProps) => {
  const { t, i18n } = useTranslation();
  const locale = i18n.language || 'zh';
  const [layout, setLayout] = React.useState<GraphLayout>('force');
  const [hoveredNodeId, setHoveredNodeId] = React.useState<string | null>(null);
  const [positionVersion, setPositionVersion] = React.useState(0);

  React.useEffect(() => {
    const saved = window.localStorage.getItem(LAYOUT_STORAGE_KEY);
    if (saved === 'force' || saved === 'radial' || saved === 'tree') {
      setLayout(saved);
    }
  }, []);
  const changeLayout = (next: GraphLayout) => {
    setLayout(next);
    window.localStorage.setItem(LAYOUT_STORAGE_KEY, next);
  };

  const slice = React.useMemo(
    () => (data ? sliceEventEntityGraph(data) : { events: [], entities: [], relations: [] }),
    [data],
  );

  const translateType = React.useCallback(
    (type: string) => t(`sag.entityType.${type}`, type),
    [t],
  );

  const network = React.useMemo(
    () => buildNetwork(slice, layout, locale, selectedNode, translateType),
    [slice, layout, locale, selectedNode, translateType],
  );

  const [flowNodes, setFlowNodes, onNodesChange] = useNodesState(network.nodes);

  React.useEffect(() => {
    setFlowNodes(network.nodes);
    setHoveredNodeId(null);
  }, [network.nodes, setFlowNodes]);

  const connectedNodeIds = React.useMemo(() => {
    if (!hoveredNodeId) return null;
    const connected = new Set([hoveredNodeId]);
    network.edges.forEach((edge) => {
      if (edge.source === hoveredNodeId) connected.add(edge.target);
      if (edge.target === hoveredNodeId) connected.add(edge.source);
    });
    return connected;
  }, [network.edges, hoveredNodeId]);

  const renderedNodes = React.useMemo(() => {
    if (!connectedNodeIds) return flowNodes;
    return flowNodes.map((node) => ({
      ...node,
      style: { ...node.style, opacity: connectedNodeIds.has(node.id) ? 1 : 0.22 },
    }));
  }, [connectedNodeIds, flowNodes]);

  const renderedEdges = React.useMemo(() => {
    const positions = new Map(flowNodes.map((node) => [node.id, node.position]));
    return network.edges.map((edge) => {
      const from = positions.get(edge.source) ?? { x: 0, y: 0 };
      const to = positions.get(edge.target) ?? { x: 0, y: 0 };
      const highlighted = Boolean(
        hoveredNodeId && (edge.source === hoveredNodeId || edge.target === hoveredNodeId),
      );
      const muted = Boolean(hoveredNodeId && !highlighted);
      const strokeWidth =
        typeof edge.style?.strokeWidth === 'number' ? edge.style.strokeWidth : 1.25;
      return {
        ...edge,
        ...graphEdgeHandles(from, to),
        zIndex: highlighted ? 2 : edge.zIndex,
        style: {
          ...edge.style,
          opacity: muted ? 0.1 : 1,
          strokeWidth: highlighted ? strokeWidth + 0.85 : strokeWidth,
        },
      };
    });
  }, [network.edges, flowNodes, hoveredNodeId]);

  const resetNodePositions = React.useCallback(() => {
    setFlowNodes(network.nodes);
    setHoveredNodeId(null);
    setPositionVersion((value) => value + 1);
  }, [network.nodes, setFlowNodes]);

  const LayoutIcon = LAYOUT_ICON[layout];
  const hasNodes = network.nodes.length > 0;

  return (
    <div className={cn('absolute inset-0 overflow-hidden', className)}>
      {/* Legend */}
      <div className="pointer-events-none absolute left-3 top-3 z-10 rounded-lg border bg-card/95 px-2.5 py-2 shadow-sm backdrop-blur-sm">
        <div className="flex flex-wrap gap-x-3 gap-y-1.5">
          <span className="inline-flex items-center gap-1.5 text-[10px] text-muted-foreground">
            <span className="size-2 rounded-full bg-amber-500" />
            {t('sag.graph.event')}
          </span>
          <span className="inline-flex items-center gap-1.5 text-[10px] text-muted-foreground">
            <span className="size-2 rounded-full border border-dashed border-violet-500 bg-violet-500/20" />
            {t('sag.graph.entity')}
          </span>
        </div>
        <div className="mt-1.5 text-[10px] text-muted-foreground">
          {t('sag.graph.stats', {
            events: slice.events.length,
            entities: slice.entities.length,
            relations: slice.relations.length,
          })}
        </div>
      </div>

      {/* Toolbar: layout toggle + reset */}
      <div className="absolute right-3 top-3 z-20 flex items-center gap-1.5">
        <div className="flex items-center rounded-md border bg-card/95 shadow-sm backdrop-blur-sm">
          {(['force', 'radial', 'tree'] as GraphLayout[]).map((value) => {
            const Icon = LAYOUT_ICON[value];
            return (
              <button
                key={value}
                type="button"
                onClick={() => changeLayout(value)}
                aria-label={t(LAYOUT_LABEL_KEY[value])}
                title={t(LAYOUT_LABEL_KEY[value])}
                className={cn(
                  'grid size-8 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground',
                  layout === value && 'bg-muted text-foreground',
                )}
              >
                <Icon className="size-4" />
              </button>
            );
          })}
        </div>
        <button
          type="button"
          onClick={resetNodePositions}
          disabled={!hasNodes}
          aria-label={t('sag.graph.resetPositions')}
          title={t('sag.graph.resetPositions')}
          className="grid size-8 place-items-center rounded-md border bg-card/95 text-muted-foreground shadow-sm backdrop-blur-sm transition-colors hover:bg-muted hover:text-foreground disabled:pointer-events-none disabled:opacity-40"
        >
          <RotateCcw className="size-4" />
        </button>
      </div>

      <ReactFlow
        nodes={renderedNodes}
        edges={renderedEdges}
        nodeTypes={nodeTypes}
        nodeOrigin={[0.5, 0.5]}
        fitView
        fitViewOptions={{ padding: 0.18, minZoom: 0.12, maxZoom: 1.05 }}
        minZoom={0.08}
        maxZoom={1.8}
        proOptions={{ hideAttribution: true }}
        nodesDraggable
        onNodesChange={onNodesChange}
        nodesConnectable={false}
        elementsSelectable
        onlyRenderVisibleElements
        onNodeClick={(_event, node) => {
          const nodeData = node.data as GraphNodeData;
          if (nodeData.kind && nodeData.originalId) {
            onNodeClick(nodeData.kind, nodeData.originalId);
          }
        }}
        onNodeMouseEnter={(_event, node) => setHoveredNodeId(node.id)}
        onNodeMouseLeave={() => setHoveredNodeId(null)}
        onPaneClick={() => setHoveredNodeId(null)}
        aria-label={t('sag.graph.aria')}
      >
        <FitViewOnChange
          nodes={network.nodes}
          edges={network.edges}
          refreshKey={`${layout}-${positionVersion}-${slice.relations.length}`}
          padding={0.18}
          minZoom={0.12}
        />
        <Background variant={BackgroundVariant.Dots} gap={22} size={1} className="!bg-transparent" />
        <Controls showInteractive={false} className="!shadow-sm" />
      </ReactFlow>

      {/* Current layout indicator */}
      <div className="pointer-events-none absolute bottom-3 right-3 z-10 flex items-center gap-1 rounded-md border bg-card/90 px-2 py-1 text-[10px] text-muted-foreground shadow-sm backdrop-blur-sm">
        <LayoutIcon className="size-3" />
        {t(LAYOUT_LABEL_KEY[layout])}
      </div>
    </div>
  );
};

export default GraphCanvas;
