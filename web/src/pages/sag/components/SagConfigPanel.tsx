// SAG config panel component
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { useTranslation } from 'react-i18next';
import { useState, useEffect } from 'react';
import type { SagConfig } from '../types';
import { DEFAULT_SAG_CONFIG, SEARCH_STRATEGIES, HOP_OPTIONS } from '../constants';

interface SagConfigPanelProps {
  config: SagConfig | undefined;
  loading: boolean;
  onSave: (config: Partial<SagConfig>) => Promise<void>;
  isSaving: boolean;
}

const SagConfigPanel = ({
  config,
  loading,
  onSave,
  isSaving,
}: SagConfigPanelProps) => {
  const { t } = useTranslation();
  const [formConfig, setFormConfig] = useState<SagConfig>(DEFAULT_SAG_CONFIG);

  useEffect(() => {
    if (config) {
      setFormConfig(config);
    }
  }, [config]);

  const handleSave = async () => {
    await onSave(formConfig);
  };

  if (loading) {
    return (
      <Card className="p-6">
        <div className="flex items-center justify-center">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-6">
      <div className="space-y-6">
        {/* Enable switch */}
        <div className="flex items-center justify-between">
          <div>
            <Label className="text-base font-medium">
              {t('sag.enableSag')}
            </Label>
            <p className="text-sm text-muted-foreground">
              {t('sag.enableSagDescription')}
            </p>
          </div>
          <Switch
            checked={formConfig.enabled}
            onCheckedChange={(checked) =>
              setFormConfig((prev) => ({ ...prev, enabled: checked }))
            }
          />
        </div>

        {formConfig.enabled && (
          <>
            {/* Extract model */}
            <div className="space-y-2">
              <Label>{t('sag.extractModel')}</Label>
              <Input
                value={formConfig.extract_model}
                onChange={(e) =>
                  setFormConfig((prev) => ({
                    ...prev,
                    extract_model: e.target.value,
                  }))
                }
                placeholder={t('sag.extractModelPlaceholder')}
              />
              <p className="text-xs text-muted-foreground">
                {t('sag.extractModelHint')}
              </p>
            </div>

            {/* Extract concurrency */}
            <div className="space-y-2">
              <Label>{t('sag.extractConcurrency')}</Label>
              <Input
                type="number"
                min={1}
                max={20}
                value={formConfig.extract_concurrency}
                onChange={(e) =>
                  setFormConfig((prev) => ({
                    ...prev,
                    extract_concurrency: parseInt(e.target.value) || 4,
                  }))
                }
              />
            </div>

            {/* Search strategy */}
            <div className="space-y-2">
              <Label>{t('sag.searchStrategy')}</Label>
              <RadioGroup
                value={formConfig.search_strategy}
                onValueChange={(value) =>
                  setFormConfig((prev) => ({
                    ...prev,
                    search_strategy: value as 'vector' | 'multi',
                  }))
                }
              >
                {SEARCH_STRATEGIES.map((strategy) => (
                  <div key={strategy.value} className="flex items-center space-x-2">
                    <RadioGroupItem value={strategy.value} id={strategy.value} />
                    <Label htmlFor={strategy.value} className="font-normal">
                      {t(strategy.label)}
                    </Label>
                  </div>
                ))}
              </RadioGroup>
            </div>

            {/* Search top-k */}
            <div className="space-y-2">
              <Label>{t('sag.searchTopK')}</Label>
              <Input
                type="number"
                min={1}
                max={50}
                value={formConfig.search_top_k}
                onChange={(e) =>
                  setFormConfig((prev) => ({
                    ...prev,
                    search_top_k: parseInt(e.target.value) || 10,
                  }))
                }
              />
            </div>

            {/* Hop number */}
            <div className="space-y-2">
              <Label>{t('sag.hopNum')}</Label>
              <RadioGroup
                value={String(formConfig.hop_num)}
                onValueChange={(value) =>
                  setFormConfig((prev) => ({
                    ...prev,
                    hop_num: parseInt(value) || 1,
                  }))
                }
              >
                {HOP_OPTIONS.map((option) => (
                  <div key={option.value} className="flex items-center space-x-2">
                    <RadioGroupItem
                      value={String(option.value)}
                      id={`hop-${option.value}`}
                    />
                    <Label htmlFor={`hop-${option.value}`} className="font-normal">
                      {t(option.label)}
                    </Label>
                  </div>
                ))}
              </RadioGroup>
            </div>
          </>
        )}

        {/* Save button */}
        <div className="flex justify-end">
          <Button onClick={handleSave} disabled={isSaving}>
            {isSaving ? t('common.saving') : t('common.save')}
          </Button>
        </div>
      </div>
    </Card>
  );
};

export default SagConfigPanel;
