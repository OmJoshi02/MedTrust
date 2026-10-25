import { useEffect, useRef, useState } from 'react';
import { RefreshCw, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { formatBytes } from '@/lib/format';
import { ACCEPT_ATTRIBUTE } from '@/lib/validation';

interface ImagePreviewProps {
  file: File;
  onRemove: () => void;
  onReplace: (file: File) => void;
  disabled?: boolean;
}

/** Shows the selected image with its details and remove / replace controls. */
export function ImagePreview({ file, onRemove, onReplace, disabled = false }: ImagePreviewProps) {
  const [url, setUrl] = useState<string>();
  const replaceRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  return (
    <div className="overflow-hidden rounded-lg border border-line bg-surface">
      <div className="flex min-h-[20rem] items-center justify-center bg-dark p-3">
        {url && (
          <img
            src={url}
            alt={`Preview of the selected image: ${file.name}`}
            className="max-h-[26rem] w-auto rounded object-contain"
          />
        )}
      </div>
      <div className="flex flex-col gap-3 border-t border-line p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium" title={file.name}>
            {file.name}
          </p>
          <p className="tabular mt-0.5 text-xs text-muted">
            {formatBytes(file.size)} &middot; {file.type || 'unknown type'}
          </p>
        </div>
        <div className="flex shrink-0 gap-2">
          <input
            ref={replaceRef}
            type="file"
            accept={ACCEPT_ATTRIBUTE}
            className="sr-only"
            tabIndex={-1}
            aria-hidden="true"
            onChange={(event) => {
              const next = event.target.files?.[0];
              if (next) onReplace(next);
              event.target.value = '';
            }}
          />
          <Button
            variant="secondary"
            size="sm"
            disabled={disabled}
            onClick={() => replaceRef.current?.click()}
            leadingIcon={<RefreshCw aria-hidden="true" className="h-3.5 w-3.5" />}
          >
            Replace
          </Button>
          <Button
            variant="secondary"
            size="sm"
            disabled={disabled}
            onClick={onRemove}
            leadingIcon={<Trash2 aria-hidden="true" className="h-3.5 w-3.5" />}
          >
            Remove
          </Button>
        </div>
      </div>
    </div>
  );
}
