import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiClient, parseErrorBody } from '@/services/apiClient';
import { ApiError } from '@/services/errors';
import { normalizeList, normalizeReport, normalizeUser } from '@/services/normalizers';
import { DISEASES } from '@/types/analysis';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

describe('ApiClient', () => {
  const client = new ApiClient('http://api.test', 1000);

  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const fetchMock = () => fetch as unknown as ReturnType<typeof vi.fn>;

  it('sends credentials and parses JSON', async () => {
    fetchMock().mockResolvedValue(jsonResponse({ ok: true }));
    const result = await client.request<{ ok: boolean }>('/x', { query: { a: 1, b: undefined, c: '' } });
    expect(result).toEqual({ ok: true });
    const [url, init] = fetchMock().mock.calls[0] as [string, RequestInit];
    expect(url).toBe('http://api.test/x?a=1');
    expect(init.credentials).toBe('include');
  });

  it('never puts tokens or passwords in storage', async () => {
    fetchMock().mockResolvedValue(jsonResponse({ ok: true }));
    await client.request('/auth/login', { method: 'POST', body: { email: 'a@b.co', password: 'secret-pw' } });
    expect(JSON.stringify({ ...window.localStorage })).not.toContain('secret-pw');
    expect(JSON.stringify({ ...window.sessionStorage })).not.toContain('secret-pw');
  });

  it.each([
    [401, 'unauthorized'],
    [403, 'forbidden'],
    [404, 'not_found'],
    [409, 'conflict'],
    [413, 'payload_too_large'],
    [415, 'unsupported_media'],
    [422, 'validation'],
    [429, 'rate_limited'],
    [500, 'server'],
    [503, 'server'],
  ] as const)('maps HTTP %i to "%s"', async (status, kind) => {
    fetchMock().mockResolvedValue(jsonResponse({ detail: 'nope' }, status));
    await expect(client.request('/x')).rejects.toMatchObject({ name: 'ApiError', kind, status, message: 'nope' });
  });

  it('reports a network failure', async () => {
    fetchMock().mockRejectedValue(new TypeError('Failed to fetch'));
    await expect(client.request('/x')).rejects.toMatchObject({ kind: 'network' });
  });

  it('reports a timeout distinctly from a caller cancellation', async () => {
    fetchMock().mockImplementation(
      (_url: string, init: RequestInit) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
        }),
    );
    await expect(new ApiClient('http://api.test', 20).request('/x')).rejects.toMatchObject({ kind: 'timeout' });

    const controller = new AbortController();
    const pending = client.request('/x', { signal: controller.signal });
    controller.abort();
    await expect(pending).rejects.toMatchObject({ kind: 'aborted' });
  });

  it('rejects a non-JSON success body as a bad response', async () => {
    fetchMock().mockResolvedValue(new Response('<html>oops</html>', { status: 200 }));
    await expect(client.request('/x')).rejects.toMatchObject({ kind: 'bad_response' });
  });

  it('notifies the unauthorized handler unless suppressed', async () => {
    const handler = vi.fn();
    client.setUnauthorizedHandler(handler);
    fetchMock().mockImplementation(async () => jsonResponse({ detail: 'expired' }, 401));

    await expect(client.request('/x')).rejects.toBeInstanceOf(ApiError);
    expect(handler).toHaveBeenCalledTimes(1);

    await expect(client.request('/x', { suppressUnauthorizedHandler: true })).rejects.toBeInstanceOf(ApiError);
    expect(handler).toHaveBeenCalledTimes(1);
    client.setUnauthorizedHandler(null);
  });

  it('parses FastAPI validation errors into per-field messages', () => {
    const parsed = parseErrorBody({
      detail: [{ loc: ['body', 'email'], msg: 'value is not a valid email address', type: 'value_error' }],
    });
    expect(parsed.message).toBe('value is not a valid email address');
    expect(parsed.fieldErrors).toEqual({ email: 'value is not a valid email address' });
  });
});

describe('response normalisation', () => {
  it('always returns the 14 diseases in canonical order, even for a sparse response', () => {
    const report = normalizeReport({ id: 'a1', status: 'completed' });
    expect(report.predictions.map((p) => p.disease)).toEqual([...DISEASES]);
    expect(report.predictions.every((p) => p.rawProbability === null && p.calibratedProbability === null)).toBe(true);
  });

  it('leaves every optional section undefined rather than inventing values', () => {
    const report = normalizeReport({ id: 'a1', status: 'completed', predictions: { Effusion: 0.4 } });
    expect(report.quality).toBeUndefined();
    expect(report.uncertainty).toBeUndefined();
    expect(report.ood).toBeUndefined();
    expect(report.calibration).toBeUndefined();
    expect(report.explanation).toBeUndefined();
    expect(report.isDemo).toBe(false);
  });

  it('accepts predictions as an array, an object, or parallel maps', () => {
    const asArray = normalizeReport({
      id: 'x',
      predictions: [
        { disease: 'Pleural Thickening', raw_probability: 0.3, calibrated_probability: 0.2 },
        { name: 'Mass', raw: 0.1 },
      ],
    });
    expect(asArray.predictions[12]).toMatchObject({ disease: 'Pleural_Thickening', rawProbability: 0.3, calibratedProbability: 0.2 });
    expect(asArray.predictions[4]).toMatchObject({ disease: 'Mass', rawProbability: 0.1, calibratedProbability: null });

    const parallel = normalizeReport({
      id: 'y',
      raw_probabilities: { Hernia: 0.7 },
      calibrated_probabilities: { Hernia: 0.5 },
      thresholds: { Hernia: 0.4 },
    });
    expect(parallel.predictions[13]).toMatchObject({ rawProbability: 0.7, calibratedProbability: 0.5, threshold: 0.4 });
  });

  it('ignores out-of-range probabilities and says so', () => {
    const report = normalizeReport({ id: 'z', predictions: { Effusion: 1.7, Mass: 0.2 } });
    expect(report.predictions[2]?.rawProbability).toBeNull();
    expect(report.predictions[4]?.rawProbability).toBe(0.2);
    expect(report.warnings.join(' ')).toMatch(/outside the range 0–1/);
  });

  it('parses supplied sections and only those', () => {
    const report = normalizeReport({
      id: 'full',
      status: 'success',
      image_quality: { overall_status: 'pass', metrics: { sharpness: 0.8, contrast: { value: 0.6, status: 'warn' } } },
      uncertainty: { method: 'MC dropout', mean_uncertainty: 0.12 },
      ood: { is_ood: true, score: 0.9 },
      calibration: { ece: 0.03, reliability: [{ confidence: 0.5, accuracy: 0.45 }, { confidence: 9, accuracy: 2 }] },
      explanation: { artifacts: [{ name: 'gradcam_mass', disease: 'Mass' }], stability: 0.8 },
    });
    expect(report.status).toBe('completed');
    expect(report.quality?.overallStatus).toBe('good');
    expect(report.quality?.metrics.map((m) => m.key)).toEqual(['sharpness', 'contrast']);
    expect(report.quality?.metrics[1]?.status).toBe('caution');
    expect(report.uncertainty?.metrics[0]).toMatchObject({ key: 'mean_uncertainty', value: 0.12 });
    expect(report.ood).toMatchObject({ status: 'out_of_distribution', score: 0.9 });
    expect(report.calibration?.ece).toBe(0.03);
    expect(report.calibration?.brier).toBeUndefined();
    expect(report.calibration?.bins).toHaveLength(1); // the invalid bin is dropped
    expect(report.explanation?.artifacts[0]).toMatchObject({ name: 'gradcam_mass', disease: 'Mass' });
    expect(report.explanation?.faithfulness).toBeUndefined();
  });

  it('rejects a payload with no analysis id', () => {
    expect(() => normalizeReport({ status: 'completed' })).toThrowError(ApiError);
    expect(() => normalizeReport('nope')).toThrowError(ApiError);
  });

  it('normalises list responses, including a missing total', () => {
    const list = normalizeList(
      { items: [{ id: '1', filename: 'a.png', status: 'done', predictions: { Mass: 0.8 } }, { nonsense: true }] },
      { page: 1, pageSize: 10 },
    );
    expect(list.items).toHaveLength(1);
    expect(list.items[0]?.topFinding).toMatchObject({ disease: 'Mass', probability: 0.8, calibrated: false });
    expect(list.total).toBeNull();
    expect(list.hasMore).toBe(false);
  });

  it('normalises users and rejects payloads without an email', () => {
    expect(normalizeUser({ user: { email: 'a@b.co', full_name: 'Ada', is_active: true } })).toMatchObject({
      email: 'a@b.co',
      fullName: 'Ada',
      status: 'active',
      isDemo: false,
    });
    expect(normalizeUser({ id: 1 })).toBeNull();
  });
});
