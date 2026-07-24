// Node detail panel component
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { X, Calendar, FileText, Tag, Link2 } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import type { SagNodeDetail, SelectedNode } from '../types';
import { ENTITY_TYPE_COLORS, DEFAULT_ENTITY_COLOR } from '../constants';

interface NodeDetailPanelProps {
  node: SagNodeDetail | null | undefined;
  selectedNode: SelectedNode | null;
  loading: boolean;
  onClose: () => void;
  onNavigateToNode?: (kind: 'event' | 'entity', id: string) => void;
}

const NodeDetailPanel = ({
  node,
  selectedNode,
  loading,
  onClose,
  onNavigateToNode,
}: NodeDetailPanelProps) => {
  const { t } = useTranslation();

  if (!selectedNode) return null;

  const isEvent = selectedNode.kind === 'event';

  return (
    <Card className="absolute bottom-0 left-0 right-0 z-10 max-h-[40%] overflow-hidden border-t shadow-lg">
      <div className="flex items-center justify-between border-b px-4 py-2">
        <div className="flex items-center gap-2">
          <Badge variant={isEvent ? 'destructive' : 'default'}>
            {isEvent ? t('sag.event') : t('sag.entity')}
          </Badge>
          <span className="font-medium">
            {isEvent ? node?.title : node?.name}
          </span>
        </div>
        <Button variant="ghost" size="sm" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>

      <ScrollArea className="max-h-[calc(40vh-48px)] p-4">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          </div>
        ) : node ? (
          <div className="space-y-4">
            {/* Basic info */}
            <div className="grid grid-cols-2 gap-4">
              {isEvent ? (
                <>
                  {node.category && (
                    <div className="flex items-center gap-2">
                      <Tag className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">
                        {t('sag.category')}:
                      </span>
                      <Badge variant="outline">{node.category}</Badge>
                    </div>
                  )}
                  {node.start_time && (
                    <div className="flex items-center gap-2">
                      <Calendar className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">
                        {t('sag.time')}:
                      </span>
                      <span className="text-sm">
                        {new Date(node.start_time).toLocaleDateString()}
                      </span>
                    </div>
                  )}
                  {node.doc_id && (
                    <div className="flex items-center gap-2">
                      <FileText className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">
                        {t('sag.document')}:
                      </span>
                      <span className="text-sm truncate max-w-[150px]">
                        {node.doc_id}
                      </span>
                    </div>
                  )}
                  {node.rank !== undefined && (
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-muted-foreground">
                        {t('sag.rank')}:
                      </span>
                      <span className="text-sm">{node.rank}</span>
                    </div>
                  )}
                </>
              ) : (
                <>
                  {node.type && (
                    <div className="flex items-center gap-2">
                      <Tag className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm text-muted-foreground">
                        {t('sag.type')}:
                      </span>
                      <Badge
                        style={{
                          backgroundColor:
                            ENTITY_TYPE_COLORS[node.type] || DEFAULT_ENTITY_COLOR,
                          color: 'white',
                        }}
                      >
                        {node.type}
                      </Badge>
                    </div>
                  )}
                  {node.heat !== undefined && (
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-muted-foreground">
                        {t('sag.heat')}:
                      </span>
                      <span className="text-sm">{node.heat}</span>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Summary/Description */}
            {(node.summary || node.description) && (
              <div>
                <h4 className="mb-1 text-sm font-medium text-muted-foreground">
                  {isEvent ? t('sag.summary') : t('sag.description')}
                </h4>
                <p className="text-sm">{node.summary || node.description}</p>
              </div>
            )}

            {/* Content (for events) */}
            {isEvent && node.content && (
              <div>
                <h4 className="mb-1 text-sm font-medium text-muted-foreground">
                  {t('sag.content')}
                </h4>
                <p className="text-sm whitespace-pre-wrap">{node.content}</p>
              </div>
            )}

            {/* Associated entities (for events) */}
            {isEvent && node.entities && node.entities.length > 0 && (
              <div>
                <h4 className="mb-2 flex items-center gap-1 text-sm font-medium text-muted-foreground">
                  <Link2 className="h-4 w-4" />
                  {t('sag.associatedEntities')} ({node.entities.length})
                </h4>
                <div className="flex flex-wrap gap-2">
                  {node.entities.map((entity) => (
                    <button
                      key={entity.id}
                      onClick={() => onNavigateToNode?.('entity', entity.id)}
                      className="inline-flex items-center gap-1 rounded-full border px-2 py-1 text-xs hover:bg-muted"
                    >
                      <span
                        className="h-2 w-2 rounded-full"
                        style={{
                          backgroundColor:
                            ENTITY_TYPE_COLORS[entity.type] || DEFAULT_ENTITY_COLOR,
                        }}
                      />
                      {entity.name}
                      <span className="text-muted-foreground">
                        ({entity.weight})
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Associated events (for entities) */}
            {!isEvent && node.events && node.events.length > 0 && (
              <div>
                <h4 className="mb-2 flex items-center gap-1 text-sm font-medium text-muted-foreground">
                  <Link2 className="h-4 w-4" />
                  {t('sag.associatedEvents')} ({node.events.length})
                </h4>
                <div className="space-y-2">
                  {node.events.map((event) => (
                    <button
                      key={event.id}
                      onClick={() => onNavigateToNode?.('event', event.id)}
                      className="block w-full rounded border p-2 text-left text-sm hover:bg-muted"
                    >
                      <span className="font-medium">{event.title}</span>
                      {event.category && (
                        <Badge variant="outline" className="ml-2">
                          {event.category}
                        </Badge>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <p className="text-center text-muted-foreground">
            {t('sag.noData')}
          </p>
        )}
      </ScrollArea>
    </Card>
  );
};

export default NodeDetailPanel;
