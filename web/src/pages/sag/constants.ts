// SAG constants

// Entity type colors keyed by the backend entity type names (Chinese by
// default, see rag/sag/config.py entity_types). Unknown/custom types fall
// back to DEFAULT_ENTITY_COLOR.
export const ENTITY_TYPE_COLORS: Record<string, string> = {
  时间: '#6DC8EC',
  地点: '#F6BD16',
  人物: '#5B8FF9',
  组织: '#5AD8A6',
  群体: '#9270CA',
  主题: '#FF9D4D',
  作品: '#FF99C3',
  产品: '#269A99',
  动作: '#E8684A',
  指标: '#BDD2FD',
  标签: '#A0DC2C',
  other: '#BFBFBF',
};

export const DEFAULT_ENTITY_COLOR = '#BFBFBF';

export const EVENT_NODE_COLOR = '#E8684A';
export const ENTITY_NODE_COLOR = '#5B8FF9';

export const SEARCH_STRATEGIES = [
  { value: 'vector', label: 'sag.strategyFast' },
  { value: 'multi', label: 'sag.strategyPrecise' },
] as const;

export const HOP_OPTIONS = [
  { value: 1, label: 'sag.hop1' },
  { value: 2, label: 'sag.hop2' },
] as const;

export const DEFAULT_SAG_CONFIG = {
  enabled: false,
  extract_model: '',
  extract_concurrency: 4,
  chunk_max_tokens: 1000,
  search_strategy: 'multi' as const,
  search_top_k: 10,
  hop_num: 1,
};

export const STATUS_COLORS: Record<string, string> = {
  idle: '#BFBFBF',
  running: '#1890FF',
  paused: '#FAAD14',
  completed: '#52C41A',
  failed: '#FF4D4F',
  cancelled: '#BFBFBF',
};

export const MAX_EVENT_LIMIT = 1000;
export const MAX_ENTITY_LIMIT = 1000;
export const DEFAULT_EVENT_LIMIT = 200;
export const DEFAULT_ENTITY_LIMIT = 200;
