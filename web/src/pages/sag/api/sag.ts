// SAG API service
import api from '@/utils/api';
import request from '@/utils/request';
import type {
  SagConfig,
  SagExpandResponse,
  SagGraphData,
  SagListResponse,
  SagEntity,
  SagEvent,
  SagNodeDetail,
  SagStatus,
  SagDocsResponse,
} from '../types';

export interface SagGraphParams {
  event_limit?: number;
  entity_limit?: number;
  doc_ids?: string;
  entity_types?: string;
}

export interface SagListParams {
  page?: number;
  page_size?: number;
  entity_type?: string;
  category?: string;
  doc_id?: string;
}

export interface SagExpandParams {
  node_kind: 'event' | 'entity';
  node_id: string;
  limit?: number;
}

// Graph APIs
export function fetchSagGraph(kbId: string, params?: SagGraphParams) {
  return request.get<SagGraphData>(api.sagGraph(kbId), { params });
}

export function fetchSagNodeDetail(
  kbId: string,
  kind: 'event' | 'entity',
  nodeId: string,
) {
  return request.get<SagNodeDetail>(api.sagNodeDetail(kbId, kind, nodeId));
}

export function expandSagNode(kbId: string, params: SagExpandParams) {
  return request.post<SagExpandResponse>(api.sagExpand(kbId), params);
}

// List APIs
export function fetchSagEntities(kbId: string, params?: SagListParams) {
  return request.get<SagListResponse<SagEntity>>(api.sagEntities(kbId), {
    params,
  });
}

export function fetchSagEvents(kbId: string, params?: SagListParams) {
  return request.get<SagListResponse<SagEvent>>(api.sagEvents(kbId), {
    params,
  });
}

// Status APIs
export function fetchSagStatus(kbId: string) {
  return request.get<SagStatus>(api.sagStatus(kbId));
}

export function triggerSagRebuild(kbId: string) {
  return request.post<{ task_id: string; message: string; doc_count: number }>(
    api.sagRebuild(kbId),
  );
}

export function pauseSagTask(kbId: string) {
  return request.post<{ message: string; task_id: string }>(api.sagPause(kbId));
}

export function resumeSagTask(kbId: string) {
  return request.post<{ message: string; task_id: string }>(
    api.sagResume(kbId),
  );
}

export function cancelSagTask(kbId: string) {
  return request.post<{ message: string; task_id: string }>(api.sagCancel(kbId));
}

// Config APIs
export function fetchSagConfig(kbId: string) {
  return request.get<SagConfig>(api.sagConfig(kbId));
}

export function updateSagConfig(kbId: string, config: Partial<SagConfig>) {
  return request.put<SagConfig>(api.sagConfig(kbId), config);
}

// Document grouping API
export function fetchSagDocs(kbId: string) {
  return request.get<SagDocsResponse>(api.sagDocs(kbId));
}
