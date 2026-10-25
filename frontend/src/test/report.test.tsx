import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { demoAnalysisApi } from '@/demo';
import { ApiError } from '@/services/errors';
import { normalizeReport } from '@/services/normalizers';
import { renderApp } from './utils';

describe('report page', () => {
  it('renders every section of a full sample report and labels it as sample data', async () => {
    renderApp('/report/demo-sample-001');

    expect(await screen.findByRole('heading', { level: 1, name: 'sample_pa_view_01.png' })).toBeInTheDocument();
    expect(screen.getByText(/sample data — not model output/i)).toBeInTheDocument();
    expect(screen.getByText(/you are viewing sample data/i)).toBeInTheDocument();

    for (const title of [
      'Uploaded X-ray',
      'Disease prediction results',
      'Image-quality assessment',
      'Uncertainty assessment',
      'Out-of-distribution assessment',
      'Calibration',
      'Explainability',
      'Limitations and disclaimer',
    ]) {
      expect(screen.getByRole('heading', { level: 2, name: title })).toBeInTheDocument();
    }

    expect(screen.getByText(/not a confirmed diagnosis/i)).toBeInTheDocument();
    expect(screen.getAllByText(/model explanations, not proof of pathology/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/not a probability of disease/i)).toBeInTheDocument();
  });

  it('lists the 14 diseases in canonical order with raw and calibrated columns', async () => {
    renderApp('/report/demo-sample-001');
    const table = await screen.findByRole('table');
    const rows = within(table).getAllByRole('row').slice(1);
    expect(rows).toHaveLength(14);
    expect(within(rows[0] as HTMLElement).getByRole('rowheader')).toHaveTextContent('Atelectasis');
    expect(within(rows[12] as HTMLElement).getByRole('rowheader')).toHaveTextContent('Pleural Thickening');
    expect(within(rows[13] as HTMLElement).getByRole('rowheader')).toHaveTextContent('Hernia');
    expect(within(table).getAllByRole('columnheader').map((c) => c.textContent)).toEqual(
      expect.arrayContaining(['Calibrated', 'Raw']),
    );
  });

  it('lets the reader switch between calibrated and raw views', async () => {
    const user = userEvent.setup();
    renderApp('/report/demo-sample-001');
    await screen.findByRole('table');

    await user.click(screen.getByRole('radio', { name: 'Raw' }));
    const headers = within(screen.getByRole('table')).getAllByRole('columnheader');
    expect(headers.map((h) => h.textContent)).not.toContain('Calibrated');
  });

  it('shows explicit "not evaluated" messages when optional data is missing', async () => {
    vi.spyOn(demoAnalysisApi, 'getReport').mockResolvedValue(
      normalizeReport({
        id: 'sparse-1',
        status: 'completed',
        filename: 'sparse.png',
        raw_probabilities: { Effusion: 0.4, Mass: 0.1 },
      }),
    );
    renderApp('/report/sparse-1');

    expect(await screen.findByRole('heading', { level: 1, name: 'sparse.png' })).toBeInTheDocument();
    expect(screen.getByText(/image-quality assessment was not returned/i)).toBeInTheDocument();
    expect(screen.getByText(/uncertainty estimates were not returned/i)).toBeInTheDocument();
    expect(screen.getByText(/out-of-distribution assessment was not returned/i)).toBeInTheDocument();
    expect(screen.getByText(/calibration information was not returned/i)).toBeInTheDocument();
    expect(screen.getByText(/explainability results were not returned/i)).toBeInTheDocument();

    // Missing calibrated values are labelled, never invented.
    expect(screen.getAllByText('Not provided').length).toBeGreaterThan(0);
    // Real (non-demo) data must not carry the sample-data label.
    expect(within(screen.getByRole('main')).queryByText(/sample data/i)).not.toBeInTheDocument();
    // No fabricated metrics.
    expect(screen.queryByText(/reliability diagram/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/expected calibration error/i)).not.toBeInTheDocument();
  });

  it('degrades gracefully for the deliberately sparse sample report', async () => {
    renderApp('/report/demo-sample-004');
    expect(await screen.findByRole('heading', { level: 1, name: 'sample_incomplete_result.png' })).toBeInTheDocument();
    expect(screen.getByText(/uncertainty estimates were not returned/i)).toBeInTheDocument();
    expect(screen.getByText(/explainability results were not returned/i)).toBeInTheDocument();
  });

  it('shows stability and faithfulness only when supplied', async () => {
    renderApp('/report/demo-sample-001');
    expect(await screen.findByRole('heading', { name: 'Explanation stability' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Explanation faithfulness' })).toBeInTheDocument();
    expect(screen.queryByText(/not calculated for this analysis/i)).not.toBeInTheDocument();
  });

  it('shows a failed analysis clearly', async () => {
    vi.spyOn(demoAnalysisApi, 'getReport').mockResolvedValue(
      normalizeReport({ id: 'f1', status: 'failed', error: 'Inference crashed.' }),
    );
    renderApp('/report/f1');
    expect(await screen.findByText(/this analysis did not complete/i)).toBeInTheDocument();
    expect(screen.getByText('Inference crashed.')).toBeInTheDocument();
  });

  it('treats an unknown or foreign report id as not found', async () => {
    renderApp('/report/someone-elses-report');
    expect(await screen.findByRole('heading', { name: /this report could not be found/i })).toBeInTheDocument();
  });

  it('treats a forbidden response exactly like not found, so ids cannot be probed', async () => {
    vi.spyOn(demoAnalysisApi, 'getReport').mockRejectedValue(
      new ApiError({ kind: 'forbidden', status: 403, message: 'Forbidden' }),
    );
    renderApp('/report/other-user');
    expect(await screen.findByRole('heading', { name: /this report could not be found/i })).toBeInTheDocument();
  });

  it('offers a retry when the report cannot be loaded', async () => {
    const user = userEvent.setup();
    const spy = vi
      .spyOn(demoAnalysisApi, 'getReport')
      .mockRejectedValueOnce(new ApiError({ kind: 'network', message: 'The MedTrust service could not be reached.' }))
      .mockResolvedValueOnce(normalizeReport({ id: 'r1', status: 'completed', filename: 'back.png' }));
    renderApp('/report/r1');

    expect(await screen.findByText(/the report could not be loaded/i)).toBeInTheDocument();
    expect(screen.getByText(/could not be reached/i)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /try again/i }));
    expect(await screen.findByRole('heading', { level: 1, name: 'back.png' })).toBeInTheDocument();
    expect(spy).toHaveBeenCalledTimes(2);
  });
});
