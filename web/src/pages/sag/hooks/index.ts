// SAG hooks
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useCallback, useState } from 'react';
import type {
  SagConfig,
  SagGraphData,
  SagNodeDetail,
  SagStatus,
  SagDoc,
  SelectedNode,
  ViewMode,
  GraphFilters,
} from '../types';
import {
  fetchSagGraph,
  fetchSagNodeDetail,
  fetchSagStatus,
  fetchSagConfig,
  updateSagConfig,
  triggerSagRebuild,
  pauseSagTask,
  resumeSagTask,
  cancelSagTask,
  fetchSagDocs,
  type SagGraphParams,
} from '../api/sag';

export const sagQueryKeys = {
  graph: (kbId: string, params?: SagGraphParams) =>
    ['sag', 'graph', kbId, params] as const,
  nodeDetail: (kbId: string, kind: string, nodeId: string) =>
    ['sag', 'node', kbId, kind, nodeId] as const,
  status: (kbId: string) => ['sag', 'status', kbId] as const,
  config: (kbId: string) => ['sag', 'config', kbId] as const,
  docs: (kbId: string) => ['sag', 'docs', kbId] as const,
};

// Graph data hook
export function useSagGraphData(kbId: string, params?: SagGraphParams) {
  const { data, isFetching: loading, error, refetch } = useQuery({
    queryKey: sagQueryKeys.graph(kbId, params),
    queryFn: async () => {
      const res = await fetchSagGraph(kbId, params);
      return res.data?.data as SagGraphData;
    },
    enabled: !!kbId,
    staleTime: 30000,
  });

  return { data, loading, error, refetch };
}

// Node detail hook
export function useSagNodeDetail(
  kbId: string,
  selectedNode: SelectedNode | null,
) {
  const { data, isFetching: loading } = useQuery({
    queryKey: selectedNode
      ? sagQueryKeys.nodeDetail(kbId, selectedNode.kind, selectedNode.id)
      : ['sag', 'node', 'none'],
    queryFn: async () => {
      if (!selectedNode) return null;
      const res = await fetchSagNodeDetail(kbId, selectedNode.kind, selectedNode.id);
      return res.data?.data as SagNodeDetail;
    },
    enabled: !!kbId && !!selectedNode,
  });

  return { data, loading };
}

// Status hook with polling
export function useSagStatus(kbId: string, enabled: boolean = true) {
  const { data, isFetching: loading, refetch } = useQuery({
    queryKey: sagQueryKeys.status(kbId),
    queryFn: async () => {
      const res = await fetchSagStatus(kbId);
      return res.data?.data as SagStatus;
    },
    enabled: !!kbId && enabled,
    refetchInterval: (query) => {
      const status = query.state.data?.task_status;
      // Poll every 3 seconds when running, otherwise stop
      return status === 'running' ? 3000 : false;
    },
  });

  return { data, loading, refetch };
}

// Config hook
export function useSagConfig(kbId: string) {
  const queryClient = useQueryClient();

  const { data, isFetching: loading } = useQuery({
    queryKey: sagQueryKeys.config(kbId),
    queryFn: async () => {
      const res = await fetchSagConfig(kbId);
      return res.data?.data as SagConfig;
    },
    enabled: !!kbId,
  });

  const updateMutation = useMutation({
    mutationFn: (config: Partial<SagConfig>) => updateSagConfig(kbId, config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: sagQueryKeys.config(kbId) });
    },
  });

  return {
    data,
    loading,
    updateConfig: updateMutation.mutateAsync,
    isUpdating: updateMutation.isPending,
  };
}

// Task control hooks
export function useSagTaskControl(kbId: string) {
  const queryClient = useQueryClient();

  const invalidateStatus = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: sagQueryKeys.status(kbId) });
    queryClient.invalidateQueries({ queryKey: ['sag', 'graph', kbId] });
  }, [queryClient, kbId]);

  const rebuildMutation = useMutation({
    mutationFn: () => triggerSagRebuild(kbId),
    onSuccess: invalidateStatus,
  });

  const pauseMutation = useMutation({
    mutationFn: () => pauseSagTask(kbId),
    onSuccess: invalidateStatus,
  });

  const resumeMutation = useMutation({
    mutationFn: () => resumeSagTask(kbId),
    onSuccess: invalidateStatus,
  });

  const cancelMutation = useMutation({
    mutationFn: () => cancelSagTask(kbId),
    onSuccess: invalidateStatus,
  });

  return {
    rebuild: rebuildMutation.mutateAsync,
    pause: pauseMutation.mutateAsync,
    resume: resumeMutation.mutateAsync,
    cancel: cancelMutation.mutateAsync,
    isRebuilding: rebuildMutation.isPending,
    isPausing: pauseMutation.isPending,
    isResuming: resumeMutation.isPending,
    isCancelling: cancelMutation.isPending,
  };
}

// Document list hook (for grouping/filtering by document)
export function useSagDocs(kbId: string) {
  const { data, isFetching: loading } = useQuery({
    queryKey: sagQueryKeys.docs(kbId),
    queryFn: async () => {
      const res = await fetchSagDocs(kbId);
      return (res.data?.data?.docs ?? []) as SagDoc[];
    },
    enabled: !!kbId,
    staleTime: 60000,
  });

  return { docs: data ?? [], loading };
}

// Graph interaction state hook
export function useGraphInteraction() {
  const [viewMode, setViewMode] = useState<ViewMode>('2d');
  const [selectedNode, setSelectedNode] = useState<SelectedNode | null>(null);
  const [filters, setFilters] = useState<GraphFilters>({
    doc_ids: [],
    entity_types: [],
  });

  const selectNode = useCallback((kind: 'event' | 'entity', id: string) => {
    setSelectedNode({ kind, id });
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedNode(null);
  }, []);

  const updateFilters = useCallback((newFilters: Partial<GraphFilters>) => {
    setFilters((prev) => ({ ...prev, ...newFilters }));
  }, []);

  return {
    viewMode,
    setViewMode,
    selectedNode,
    selectNode,
    clearSelection,
    filters,
    updateFilters,
  };
}
