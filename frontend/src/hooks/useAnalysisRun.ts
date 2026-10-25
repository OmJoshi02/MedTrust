import { useCallback, useEffect, useRef, useState } from 'react';
import { env } from '@/config/env';
import { analysisService, describeError, isApiError, type ApiErrorKind } from '@/services';
import type { AnalysisReport } from '@/types/analysis';
import { validateImage } from '@/lib/validation';

export type RunPhase = 'idle' | 'running' | 'success' | 'error';

/**
 * Visual stages only. Unless the backend reports real stage information, these
 * advance on a timer to show that work is in progress; they are not measurements.
 */
export const STAGES = [
  { id: 'validate', label: 'Image validation' },
  { id: 'predict', label: 'Disease prediction' },
  { id: 'reliability', label: 'Reliability assessment' },
  { id: 'report', label: 'Report preparation' },
] as const;

const STAGE_INTERVAL_MS = 2600;

export interface RunError {
  message: string;
  kind: ApiErrorKind | 'validation';
}

export interface AnalysisRun {
  phase: RunPhase;
  /** Index of the stage currently shown as active (visual only). */
  stageIndex: number;
  /** Real upload progress, 0–100, or null before the upload has reported. */
  uploadPercent: number | null;
  report: AnalysisReport | null;
  error: RunError | null;
  /** File-selection error (type, size, content). */
  fileError: string | null;
  file: File | null;
  selectFile: (file: File) => Promise<void>;
  clearFile: () => void;
  submit: () => Promise<void>;
  cancel: () => void;
  reset: () => void;
}

export function useAnalysisRun(): AnalysisRun {
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [phase, setPhase] = useState<RunPhase>('idle');
  const [stageIndex, setStageIndex] = useState(0);
  const [uploadPercent, setUploadPercent] = useState<number | null>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [error, setError] = useState<RunError | null>(null);

  // Guards against double submission even before React re-renders.
  const inFlight = useRef(false);
  const controllerRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const mounted = useRef(true);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      controllerRef.current?.abort();
      stopTimer();
    };
  }, [stopTimer]);

  const selectFile = useCallback(
    async (candidate: File) => {
      if (inFlight.current) return;
      const problem = await validateImage(candidate, { maxBytes: env.maxUploadMb * 1024 * 1024 });
      if (!mounted.current) return;
      if (problem) {
        setFileError(problem);
        return;
      }
      setFileError(null);
      setFile(candidate);
      setReport(null);
      setError(null);
      setPhase('idle');
    },
    [],
  );

  const clearFile = useCallback(() => {
    if (inFlight.current) return;
    setFile(null);
    setFileError(null);
    setReport(null);
    setError(null);
    setPhase('idle');
  }, []);

  const submit = useCallback(async () => {
    // Never analyse anything but a file the user explicitly selected.
    if (!file || inFlight.current) return;
    inFlight.current = true;

    const controller = new AbortController();
    controllerRef.current = controller;

    setPhase('running');
    setError(null);
    setReport(null);
    setStageIndex(0);
    setUploadPercent(null);

    stopTimer();
    timerRef.current = setInterval(() => {
      // Stop on the final stage until the response arrives.
      setStageIndex((current) => Math.min(current + 1, STAGES.length - 1));
    }, STAGE_INTERVAL_MS);

    try {
      const result = await analysisService.create(file, {
        signal: controller.signal,
        onUploadProgress: (percent) => {
          if (mounted.current) setUploadPercent(percent);
        },
      });
      if (!mounted.current) return;
      setReport(result);
      setStageIndex(STAGES.length);
      setPhase('success');
    } catch (caught) {
      if (!mounted.current) return;
      if (isApiError(caught) && caught.kind === 'aborted') {
        setPhase('idle');
        return;
      }
      setError({
        message: describeError(caught),
        kind: isApiError(caught) ? caught.kind : 'unknown',
      });
      setPhase('error');
    } finally {
      stopTimer();
      inFlight.current = false;
      controllerRef.current = null;
    }
  }, [file, stopTimer]);

  const cancel = useCallback(() => {
    controllerRef.current?.abort();
  }, []);

  const reset = useCallback(() => {
    if (inFlight.current) return;
    setFile(null);
    setFileError(null);
    setReport(null);
    setError(null);
    setPhase('idle');
    setStageIndex(0);
    setUploadPercent(null);
  }, []);

  return {
    phase,
    stageIndex,
    uploadPercent,
    report,
    error,
    fileError,
    file,
    selectFile,
    clearFile,
    submit,
    cancel,
    reset,
  };
}
