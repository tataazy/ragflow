import { Switch } from '@/components/ui/switch';
import { useFormContext } from 'react-hook-form';
import { useTranslation } from 'react-i18next';

export default function SagFormFields() {
  const { t } = useTranslation();
  const { register, watch, setValue } = useFormContext();

  const sagEnabled = watch('parser_config.sag.enabled');

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-base font-medium text-text-primary">
            SAG (Structured Attribution Graph)
          </div>
          <div className="text-sm text-text-secondary mt-1">
            Enable SAG to extract entities and events from documents for knowledge graph visualization
          </div>
        </div>
        <Switch
          checked={sagEnabled}
          onCheckedChange={(checked) => {
            setValue('parser_config.sag.enabled', checked);
          }}
        />
      </div>

      {sagEnabled && (
        <div className="pl-4 border-l-2 border-border-subtle space-y-3">
          <div className="text-sm text-text-secondary">
            After enabling, upload and parse documents first, then go to the SAG page to trigger extraction.
          </div>
        </div>
      )}
    </div>
  );
}
