// SAG (Structured Association Graph) TypeScript types

export interface SagEvent {
  id: string;
  title: string;
  summary: string;
  category: string;
  start_time: string | null;
  chunk_id: string;
  doc_id: string;
  rank: number;
  entity_count?: number;
}

export interface SagEntity {
  id: string;
  name: string;
  type: string;
  description: string;
  heat: number;
}

export interface SagAssociation {
  event_id: string;
  entity_id: string;
  weight: number;
  description: string;
}

export interface SagGraphData {
  events: SagEvent[];
  entities: SagEntity[];
  associations: SagAssociation[];
  total_events: number;
  total_entities: number;
  entity_types?: string[];
  sag_enabled: boolean;
}

export interface SagNodeDetail {
  id: string;
  title?: string;
  name?: string;
  summary?: string;
  content?: string;
  category?: string;
  type?: string;
  description?: string;
  start_time?: string | null;
  chunk_id?: string;
  doc_id?: string;
  rank?: number;
  heat?: number;
  status?: string;
  entities?: SagEntityWithWeight[];
  events?: SagEventWithWeight[];
}

export interface SagEntityWithWeight extends SagEntity {
  weight: number;
  association_description: string;
}

export interface SagEventWithWeight extends SagEvent {
  weight: number;
  association_description: string;
}

export interface SagStatus {
  enabled: boolean;
  task_id: string;
  task_status: 'idle' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  event_count: number;
  entity_count: number;
  token_usage: number;
}

export interface SagConfig {
  enabled: boolean;
  extract_model: string;
  extract_concurrency: number;
  chunk_max_tokens: number;
  search_strategy: 'vector' | 'multi';
  search_top_k: number;
  hop_num: number;
}

export interface SagExpandResponse {
  events?: SagEvent[];
  entities?: SagEntity[];
  associations: SagAssociation[];
  has_more: boolean;
  total: number;
}

export interface SagListResponse<T> {
  page: number;
  page_size: number;
  total: number;
  data: T[];
}

export interface SagDoc {
  doc_id: string;
  name: string;
  event_count: number;
}

export interface SagDocsResponse {
  docs: SagDoc[];
}

export type ViewMode = '2d' | '3d' | 'timeline';

export type NodeKind = 'event' | 'entity';

export interface SelectedNode {
  kind: NodeKind;
  id: string;
}

export interface GraphFilters {
  doc_ids: string[];
  entity_types: string[];
  category?: string;
}
