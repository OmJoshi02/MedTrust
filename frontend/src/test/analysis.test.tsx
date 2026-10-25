import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { demoAnalysisApi } from '@/demo';
import { buildSampleReport } from '@/demo/fixtures';
import { ApiError } from '@/services/errors';
import type { AnalysisReport } from '@/types/analysis';
import { makeImageFile, renderApp } from './utils';

function sample(overrides: Partial<AnalysisReport> = {}): AnalysisReport {
  return {
    ...buildSampleReport({
      id: 'demo-test-1',
      filename: 'chest.png',
      createdAt: '2026-01-02T03:04:05.000Z',
      seed: 7,
    }),
    ...overrides,
  };
}

async function selectFile() {
  const input = (await screen.findByTestId('file-input')) as HTMLInputElement;
  fireEvent.change(input, { target: { files: [makeImageFile('chest.png')] } });
  await screen.findByRole('img', { name: /preview of the selected image/i });
}

describe('analysis submission', () => {
  it('submits the selected file and shows the returned results with a link to the report', async () => {
    const create = vi.spyOn(demoAnalysisApi, 'create').mockResolvedValue(sample());
    renderApp('/analyze');
    await selectFile();

    fireEvent.click(screen.getByRole('button', { name: 'Analyze Image' }));

    expect(await screen.findByText('Analysis complete')).toBeInTheDocument();
    expect(create).toHaveBeenCalledTimes(1);
    const submitted = create.mock.calls[0]?.[0] as File;
    expect(submitted.name).toBe('chest.png');

    // Demo results must be labelled as sample data, never as model output.
    expect(screen.getByText(/sample results — not model output/i)).toBeInTheDocument();

    const results = screen.getByRole('region', { name: /review the results/i });
    expect(within(results).getByRole('table')).toBeInTheDocument();
    expect(within(results).getAllByRole('row')).toHaveLength(6); // header + 5 technical quality indicators
    expect(within(results).getByText('Sharpness')).toBeInTheDocument();
    expect(within(results).getByText('Contrast')).toBeInTheDocument();

    expect(within(results).getByRole('link', { name: /open full report/i })).toHaveAttribute(
      'href',
      '/report/demo-test-1',
    );
  });

  it('prevents duplicate submissions while an analysis is running', async () => {
    let finish: (report: AnalysisReport) => void = () => {};
    const create = vi.spyOn(demoAnalysisApi, 'create').mockImplementation(
      () =>
        new Promise<AnalysisReport>((resolve) => {
          finish = resolve;
        }),
    );
    renderApp('/analyze');
    await selectFile();

    const button = screen.getByRole('button', { name: 'Analyze Image' });
    fireEvent.click(button);
    fireEvent.click(button);
    fireEvent.click(button);

    expect(await screen.findByRole('button', { name: /analysing/i })).toBeDisabled();
    expect(create).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
    expect(screen.getByText(/not reported by the analysis service/i)).toBeInTheDocument();

    finish(sample());
    expect(await screen.findByText('Analysis complete')).toBeInTheDocument();
  });

  it('shows a network failure and never substitutes sample results', async () => {
    vi.spyOn(demoAnalysisApi, 'create').mockRejectedValue(
      new ApiError({ kind: 'network', message: 'The MedTrust service could not be reached.' }),
    );
    renderApp('/analyze');
    await selectFile();
    fireEvent.click(screen.getByRole('button', { name: 'Analyze Image' }));

    expect(await screen.findByText(/could not reach the analysis service/i)).toBeInTheDocument();
    expect(screen.queryByText('Analysis complete')).not.toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
  });

  it.each([
    ['timeout', /the analysis timed out/i],
    ['server', /analysis service reported an error/i],
    ['unsupported_media', /unsupported file format/i],
    ['payload_too_large', /the file is too large/i],
  ] as const)('handles a %s failure', async (kind, title) => {
    vi.spyOn(demoAnalysisApi, 'create').mockRejectedValue(new ApiError({ kind, message: 'Detail from server.' }));
    renderApp('/analyze');
    await selectFile();
    fireEvent.click(screen.getByRole('button', { name: 'Analyze Image' }));

    expect(await screen.findByText(title)).toBeInTheDocument();
    expect(screen.getByText('Detail from server.')).toBeInTheDocument();
  });

  it('can retry after a failure', async () => {
    const create = vi
      .spyOn(demoAnalysisApi, 'create')
      .mockRejectedValueOnce(new ApiError({ kind: 'server', message: 'Boom' }))
      .mockResolvedValueOnce(sample());
    renderApp('/analyze');
    await selectFile();

    fireEvent.click(screen.getByRole('button', { name: 'Analyze Image' }));
    fireEvent.click(await screen.findByRole('button', { name: /try again/i }));

    expect(await screen.findByText('Analysis complete')).toBeInTheDocument();
    expect(create).toHaveBeenCalledTimes(2);
  });

  it('reports a failed analysis returned by the backend instead of showing results', async () => {
    vi.spyOn(demoAnalysisApi, 'create').mockResolvedValue(
      sample({ status: 'failed', error: 'Model could not read this image.' }),
    );
    renderApp('/analyze');
    await selectFile();
    fireEvent.click(screen.getByRole('button', { name: 'Analyze Image' }));

    expect(await screen.findByText(/the analysis did not complete/i)).toBeInTheDocument();
    expect(screen.getByText('Model could not read this image.')).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole('table')).not.toBeInTheDocument());
  });

  it('marks sections the backend did not return as not evaluated', async () => {
    vi.spyOn(demoAnalysisApi, 'create').mockResolvedValue(
      sample({
        predictions: [],
        quality: undefined,
        uncertainty: undefined,
        ood: undefined,
        calibration: undefined,
        explanation: undefined,
      }),
    );
    renderApp('/analyze');
    await selectFile();
    fireEvent.click(screen.getByRole('button', { name: 'Analyze Image' }));

    await screen.findByText('Analysis complete');
    expect(screen.getAllByText('Not evaluated').length).toBe(5);
  });
});
