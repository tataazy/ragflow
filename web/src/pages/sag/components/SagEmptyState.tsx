// SAG empty state / guide page component
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Network, Play, RefreshCw } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { SagStatus } from '../types';

interface SagEmptyStateProps {
  status: SagStatus | undefined;
  onEnable?: () => void;
  onRebuild?: () => void;
  isRebuilding?: boolean;
}

const SagEmptyState = ({
  status,
  onEnable,
  onRebuild,
  isRebuilding,
}: SagEmptyStateProps) => {
  const { t } = useTranslation();

  // Not enabled
  if (!status?.enabled) {
    return (
      <Card className="flex flex-col items-center justify-center p-12 text-center">
        <Network className="mb-4 h-16 w-16 text-muted-foreground" />
        <h3 className="mb-2 text-lg font-medium">{t('sag.notEnabled')}</h3>
        <p className="mb-6 max-w-md text-muted-foreground">
          {t('sag.notEnabledDescription')}
        </p>
        {onEnable && (
          <Button onClick={onEnable}>{t('sag.enableSag')}</Button>
        )}
      </Card>
    );
  }

  // Building
  if (status.task_status === 'running') {
    return (
      <Card className="flex flex-col items-center justify-center p-12 text-center">
        <div className="mb-4 h-16 w-16 animate-spin rounded-full border-4 border-primary border-t-transparent" />
        <h3 className="mb-2 text-lg font-medium">{t('sag.building')}</h3>
        <p className="mb-4 text-muted-foreground">
          {t('sag.buildingDescription')}
        </p>
        <div className="w-64">
          <div className="h-2 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-primary transition-all"
              style={{ width: `${Math.round(status.progress * 100)}%` }}
            />
          </div>
          <p className="mt-2 text-sm text-muted-foreground">
            {Math.round(status.progress * 100)}%
          </p>
        </div>
      </Card>
    );
  }

  // Failed
  if (status.task_status === 'failed') {
    return (
      <Card className="flex flex-col items-center justify-center p-12 text-center">
        <RefreshCw className="mb-4 h-16 w-16 text-destructive" />
        <h3 className="mb-2 text-lg font-medium">{t('sag.buildFailed')}</h3>
        <p className="mb-6 max-w-md text-muted-foreground">
          {t('sag.buildFailedDescription')}
        </p>
        {onRebuild && (
          <Button onClick={onRebuild} disabled={isRebuilding}>
            <RefreshCw className="mr-2 h-4 w-4" />
            {t('sag.retry')}
          </Button>
        )}
      </Card>
    );
  }

  // Empty graph (enabled but no data)
  if (status.event_count === 0 && status.entity_count === 0) {
    return (
      <Card className="flex flex-col items-center justify-center p-12 text-center">
        <Network className="mb-4 h-16 w-16 text-muted-foreground" />
        <h3 className="mb-2 text-lg font-medium">{t('sag.noGraphData')}</h3>
        <p className="mb-6 max-w-md text-muted-foreground">
          {t('sag.noGraphDataDescription')}
        </p>
        {onRebuild && (
          <Button onClick={onRebuild} disabled={isRebuilding}>
            <Play className="mr-2 h-4 w-4" />
            {t('sag.startBuild')}
          </Button>
        )}
      </Card>
    );
  }

  return null;
};

export default SagEmptyState;
