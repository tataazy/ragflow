// Graph toolbar component
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Box, Clock, FileText, Square } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { SagDoc, ViewMode } from '../types';

interface GraphToolbarProps {
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  entityTypeFilter?: string;
  onEntityTypeFilterChange?: (type: string) => void;
  entityTypes?: string[];
  docs?: SagDoc[];
  docFilter?: string;
  onDocFilterChange?: (docId: string) => void;
  onRefresh?: () => void;
}

const GraphToolbar = ({
  viewMode,
  onViewModeChange,
  entityTypeFilter,
  onEntityTypeFilterChange,
  entityTypes,
  docs,
  docFilter,
  onDocFilterChange,
  onRefresh,
}: GraphToolbarProps) => {
  const { t } = useTranslation();

  return (
    <div className="flex items-center justify-between border-b px-4 py-2">
      {/* View mode toggle */}
      <div className="flex items-center gap-1">
        <Button
          variant={viewMode === '2d' ? 'default' : 'ghost'}
          size="sm"
          onClick={() => onViewModeChange('2d')}
        >
          <Square className="mr-1 h-4 w-4" />
          2D
        </Button>
        <Button
          variant={viewMode === '3d' ? 'default' : 'ghost'}
          size="sm"
          onClick={() => onViewModeChange('3d')}
        >
          <Box className="mr-1 h-4 w-4" />
          3D
        </Button>
        <Button
          variant={viewMode === 'timeline' ? 'default' : 'ghost'}
          size="sm"
          onClick={() => onViewModeChange('timeline')}
        >
          <Clock className="mr-1 h-4 w-4" />
          {t('sag.timeline')}
        </Button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-2">
        {onDocFilterChange && docs && docs.length > 0 && (
          <Select
            value={docFilter || 'all'}
            onValueChange={onDocFilterChange}
          >
            <SelectTrigger className="w-[220px]">
              <FileText className="mr-1 h-4 w-4" />
              <SelectValue placeholder={t('sag.allDocuments')} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">{t('sag.allDocuments')}</SelectItem>
              {docs.map((doc) => (
                <SelectItem key={doc.doc_id} value={doc.doc_id}>
                  {doc.name} ({doc.event_count})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        {onEntityTypeFilterChange && entityTypes && entityTypes.length > 0 && (
          <Select
            value={entityTypeFilter || 'all'}
            onValueChange={onEntityTypeFilterChange}
          >
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder={t('sag.filterByType')} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">{t('sag.allTypes')}</SelectItem>
              {entityTypes.map((type) => (
                <SelectItem key={type} value={type}>
                  {t(`sag.entityType.${type}`, type)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        {onRefresh && (
          <Button variant="outline" size="sm" onClick={onRefresh}>
            {t('common.refresh')}
          </Button>
        )}
      </div>
    </div>
  );
};

export default GraphToolbar;
