import { useRef, useState, type DragEvent, type KeyboardEvent } from 'react';
import { ImagePlus } from 'lucide-react';
import { cn } from '@/lib/cn';
import { formatBytes } from '@/lib/format';
import { ACCEPT_ATTRIBUTE } from '@/lib/validation';

interface ImageDropzoneProps {
  onFileSelected: (file: File) => void;
  maxBytes: number;
  disabled?: boolean;
  /** Message id used by the parent to describe validation errors. */
  describedBy?: string;
  hasError?: boolean;
}

/** Drag-and-drop and browse control for a single image. Validation lives with the parent. */
export function ImageDropzone({ onFileSelected, maxBytes, disabled = false, describedBy, hasError }: ImageDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  // Counts nested dragenter/dragleave events so the highlight does not flicker.
  const depth = useRef(0);

  const open = () => {
    if (!disabled) inputRef.current?.click();
  };

  const onDragEnter = (event: DragEvent) => {
    event.preventDefault();
    if (disabled) return;
    depth.current += 1;
    setDragging(true);
  };

  const onDragLeave = (event: DragEvent) => {
    event.preventDefault();
    depth.current = Math.max(0, depth.current - 1);
    if (depth.current === 0) setDragging(false);
  };

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    depth.current = 0;
    setDragging(false);
    if (disabled) return;
    const file = event.dataTransfer.files[0];
    if (file) onFileSelected(file);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      open();
    }
  };

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled || undefined}
      aria-describedby={describedBy}
      aria-invalid={hasError || undefined}
      aria-label="Upload a chest X-ray image. Press Enter to browse files, or drop a file here."
      onClick={open}
      onKeyDown={onKeyDown}
      onDragEnter={onDragEnter}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      className={cn(
        'group relative flex min-h-[20rem] cursor-pointer flex-col items-center justify-center rounded-lg border border-dashed px-6 py-12 text-center transition-colors duration-150',
        dragging ? 'border-ink bg-subtle' : 'border-ink/25 bg-surface/60 hover:border-ink/60 hover:bg-surface',
        hasError && 'border-danger/60',
        disabled && 'cursor-not-allowed opacity-60',
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTRIBUTE}
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        disabled={disabled}
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onFileSelected(file);
          // Allow re-selecting the same file after removing it.
          event.target.value = '';
        }}
        data-testid="file-input"
      />
      <span
        className={cn(
          'mb-5 inline-flex h-12 w-12 items-center justify-center rounded-full border border-line bg-canvas text-ink transition-transform duration-200',
          dragging && 'scale-105',
        )}
      >
        <ImagePlus aria-hidden="true" className="h-5 w-5" />
      </span>
      <p className="text-base font-medium">{dragging ? 'Release to select this image' : 'Drag and drop a chest X-ray here'}</p>
      <p className="mt-1.5 text-sm text-muted">or</p>
      <span className="mt-2 inline-flex h-10 items-center rounded-md border border-line bg-surface px-4 text-sm font-medium transition-colors group-hover:border-ink/40">
        Browse files
      </span>
      <p className="mt-6 text-xs text-muted">
        PNG, JPG, JPEG or WebP &middot; up to {formatBytes(maxBytes)}
      </p>
    </div>
  );
}
