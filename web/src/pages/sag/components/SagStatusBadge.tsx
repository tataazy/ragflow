// SAG status badge component
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { useTranslation } from 'react-i18next';
import type { SagStatus } from '../types';
import { STATUS_COLORS } from '../constants';

interface SagStatusBadgeProps {
  status: SagStatus | undefined;
  showTooltip?: boolean;
}

const SagStatusBadge = ({ status, showTooltip = true }: SagStatusBadgeProps) => {
  const { t } = useTranslation();

  if (!status) return null;

  const color = STATUS_COLORS[status.task_status] || STATUS_COLORS.idle;
  const statusText = t(`sag.status.${status.task_status}`, status.task_status);

  const badge = (
    <span
      className="inline-block h-2.5 w-2.5 rounded-full"
      style={{ backgroundColor: color }}
    />
  );

  if (!showTooltip) return badge;

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>{badge}</TooltipTrigger>
        <TooltipContent>
          <div className="space-y-1 text-xs">
            <p className="font-medium">{statusText}</p>
            {status.task_status === 'running' && (
              <p>{t('sag.progress')}: {Math.round(status.progress * 100)}%</p>
            )}
            <p>
              {t('sag.events')}: {status.event_count}
            </p>
            <p>
              {t('sag.entities')}: {status.entity_count}
            </p>
            {status.token_usage > 0 && (
              <p>
                {t('sag.tokenUsage')}: {status.token_usage.toLocaleString()}
              </p>
            )}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

export default SagStatusBadge;
