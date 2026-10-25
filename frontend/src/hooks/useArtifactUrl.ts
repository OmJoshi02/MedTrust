import { useEffect, useState } from 'react';
import { analysisService } from '@/services';

interface ArtifactState {
  url?: string;
  error?: unknown;
  loading: boolean;
}

/**
 * Fetches a report artifact (image) through the API client, so the request
 * carries the session cookie and works across origins, and exposes it as an
 * object URL that is revoked when no longer needed.
 */
export function useArtifactUrl(analysisId: string | undefined, artifactName: string | undefined) {
  const enabled = Boolean(analysisId && artifactName);
  const [state, setState] = useState<ArtifactState>({ loading: enabled });

  useEffect(() => {
    if (!analysisId || !artifactName) {
      setState({ loading: false });
      return;
    }
    const controller = new AbortController();
    let objectUrl: string | undefined;
    setState({ loading: true });

    analysisService
      .getArtifact(analysisId, artifactName, controller.signal)
      .then((blob) => {
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        setState({ url: objectUrl, loading: false });
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setState({ error, loading: false });
      });

    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [analysisId, artifactName]);

  return state;
}
