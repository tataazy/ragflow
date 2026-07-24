// SAG Knowledge Graph page entry
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useKnowledgeBaseId } from '@/hooks/use-knowledge-request';
import { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router';
import { Routes } from '@/routes';
import { RefreshCw } from 'lucide-react';
import GraphCanvas from './components/GraphCanvas';
import GraphToolbar from './components/GraphToolbar';
import NodeDetailPanel from './components/NodeDetailPanel';
import SagEmptyState from './components/SagEmptyState';
import SagStatusBadge from './components/SagStatusBadge';
import {
  useSagGraphData,
  useSagNodeDetail,
  useSagStatus,
  useSagTaskControl,
  useSagDocs,
  useGraphInteraction,
} from './hooks';
import type { SagGraphData } from './types';

const SagGraphPage = () => {
  const { t } = useTranslation();
  const kbId = useKnowledgeBaseId();
  const navigate = useNavigate();

  // State
  const {
    viewMode,
    setViewMode,
    selectedNode,
    selectNode,
    clearSelection,
    filters,
    updateFilters,
  } = useGraphInteraction();

  // Data hooks
  const { data: status, loading: statusLoading } = useSagStatus(kbId);
  const { docs } = useSagDocs(kbId);
  const { data: graphData, loading: graphLoading, refetch } = useSagGraphData(
    kbId,
    {
      doc_ids: filters.doc_ids.length > 0 ? filters.doc_ids.join(',') : undefined,
      entity_types:
        filters.entity_types.length > 0
          ? filters.entity_types.join(',')
          : undefined,
    },
  );
  const { data: nodeDetail, loading: nodeLoading } = useSagNodeDetail(
    kbId,
    selectedNode,
  );
  const { rebuild, isRebuilding } = useSagTaskControl(kbId);

  // Check if we should show empty state
  const showEmptyState = useMemo(() => {
    if (!status) return false;
    if (!status.enabled) return true;
    if (status.task_status === 'running') return true;
    if (status.task_status === 'failed') return true;
    if (status.event_count === 0 && status.entity_count === 0) return true;
    return false;
  }, [status]);

  const handleNodeClick = useCallback(
    (kind: 'event' | 'entity', id: string) => {
      selectNode(kind, id);
    },
    [selectNode],
  );

  const handleEnable = useCallback(() => {
    // Navigate to dataset settings to enable SAG
    navigate(`${Routes.DatasetBase}${Routes.DataSetSetting}/${kbId}`);
  }, [navigate, kbId]);

  const handleRebuild = useCallback(async () => {
    try {
      await rebuild();
    } catch (error) {
      console.error('Rebuild failed:', error);
    }
  }, [rebuild]);

  const handleEntityTypeFilter = useCallback(
    (type: string) => {
      updateFilters({
        entity_types: type === 'all' ? [] : [type],
      });
    },
    [updateFilters],
  );

  const handleDocFilter = useCallback(
    (docId: string) => {
      updateFilters({
        doc_ids: docId === 'all' ? [] : [docId],
      });
    },
    [updateFilters],
  );

  return (
    <div className="flex h-full flex-col p-4">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold">{t('sag.knowledgeGraph')}</h1>
          <SagStatusBadge status={status} />
        </div>
        {status?.enabled && !showEmptyState && (
          <Button
            variant="outline"
            onClick={handleRebuild}
            disabled={isRebuilding}
          >
            <RefreshCw className={`mr-2 h-4 w-4 ${isRebuilding ? 'animate-spin' : ''}`} />
            {t('sag.rebuild')}
          </Button>
        )}
      </div>

      {/* Main content */}
      <Card className="relative flex-1 overflow-hidden">
        {showEmptyState ? (
          <div className="flex h-full items-center justify-center">
            <SagEmptyState
              status={status}
              onEnable={handleEnable}
              onRebuild={handleRebuild}
              isRebuilding={isRebuilding}
            />
          </div>
        ) : (
          <div className="flex h-full flex-col">
            {/* Toolbar */}
            <GraphToolbar
              viewMode={viewMode}
              onViewModeChange={setViewMode}
              entityTypeFilter={
                filters.entity_types.length > 0 ? filters.entity_types[0] : 'all'
              }
              onEntityTypeFilterChange={handleEntityTypeFilter}
              entityTypes={graphData?.entity_types}
              docs={docs}
              docFilter={
                filters.doc_ids.length > 0 ? filters.doc_ids[0] : 'all'
              }
              onDocFilterChange={handleDocFilter}
              onRefresh={() => refetch()}
            />

            {/* Graph area */}
            <div className="relative flex-1">
              {graphLoading ? (
                <div className="flex h-full items-center justify-center">
                  <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                </div>
              ) : viewMode === '2d' ? (
                <GraphCanvas
                  data={graphData as SagGraphData}
                  selectedNode={selectedNode}
                  onNodeClick={handleNodeClick}
                />
              ) : viewMode === '3d' ? (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  {t('sag.comingSoon3D')}
                </div>
              ) : (
                <div className="flex h-full items-center justify-center text-muted-foreground">
                  {t('sag.comingSoonTimeline')}
                </div>
              )}

              {/* Node detail panel */}
              <NodeDetailPanel
                node={nodeDetail}
                selectedNode={selectedNode}
                loading={nodeLoading}
                onClose={clearSelection}
                onNavigateToNode={handleNodeClick}
              />
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};

export default SagGraphPage;
