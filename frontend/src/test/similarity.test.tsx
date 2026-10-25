import { render, screen, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { normalizeReport } from '@/services/normalizers';
import { SimilarityPanel } from '@/components/analysis/SimilarityPanel';
import type { SimilarityAssessment } from '@/types/analysis';

describe('Similarity Normalizer and Component Tests', () => {
  it('does not fall back to concordanceRate when primary_concordance_rate is missing', () => {
    const rawPayload = {
      id: 'case-test-1',
      status: 'completed',
      similarity: {
        total_references: 22,
        concordance_rate: 100.0,
        concordant_cases_count: 5,
        primary_concordance_rate: null,
        any_finding_overlap_rate: null,
        matches: [
          {
            rank: 1,
            filename: '00000013_010.png',
            similarity_score: 70.17,
            concordance: { status: 'CONCORDANT', matches_primary: false },
          },
        ],
      },
    };

    const report = normalizeReport(rawPayload);
    const sim = report.similarity;
    expect(sim).toBeDefined();
    // Must NOT fall back to 100.0%
    expect(sim?.primary_concordance_rate).toBeUndefined();
    expect(sim?.any_finding_overlap_rate).toBeUndefined();
  });

  it('correctly parses and preserves 0.0% primary and any-finding concordance rates', () => {
    const rawPayload = {
      id: 'case-test-zero',
      status: 'completed',
      similarity: {
        total_references: 22,
        concordance_rate: 0.0,
        concordant_cases_count: 0,
        primary_concordance_rate: 0.0,
        primary_concordant_count: 0,
        any_finding_overlap_rate: 0.0,
        any_finding_overlap_count: 0,
        matches: [
          {
            rank: 1,
            filename: '00000001_000.png',
            similarity_score: 65.0,
            concordance: { status: 'DIFFERENTIAL', matches_primary: false, has_overlap: false },
          },
        ],
      },
    };

    const report = normalizeReport(rawPayload);
    const sim = report.similarity;
    expect(sim?.primary_concordance_rate).toBe(0.0);
    expect(sim?.primary_concordant_count).toBe(0);
    expect(sim?.any_finding_overlap_rate).toBe(0.0);
    expect(sim?.any_finding_overlap_count).toBe(0);

    render(<SimilarityPanel similarity={sim} />);

    // 0.0% must NOT be rendered as N/A (it appears in both Primary Concordance and Any-Finding Overlap)
    expect(screen.getAllByText('0.0%')).toHaveLength(2);
  });

  it('renders Case 42 metrics correctly with 20.0% primary and 100.0% overlap', () => {
    const case42Sim: SimilarityAssessment = {
      status: 'SUCCESS',
      top_k: 5,
      total_references: 22,
      top_match: {
        rank: 1,
        filename: '00000013_010.png',
        similarity_score: 70.17,
        image_url: '/reference_images/00000013_010.png',
      },
      concordance_rate: 20.0,
      concordant_cases_count: 1,
      primary_concordance_rate: 20.0,
      primary_concordant_count: 1,
      any_finding_overlap_rate: 100.0,
      any_finding_overlap_count: 5,
      primary_query_finding: 'Consolidation',
      concordance_summary: '1 of 5 (20.0%) retrieved reference cases corroborate primary Consolidation finding; 5 of 5 (100.0%) share any detected finding.',
      matches: [
        {
          rank: 1,
          filename: '00000013_010.png',
          similarity_score: 70.17,
          image_url: '/reference_images/00000013_010.png',
          metadata: { finding_labels: 'Effusion|Pneumonia|Pneumothorax' },
          concordance: {
            status: 'SECONDARY_OVERLAP',
            label: 'Secondary Overlap',
            matches_primary: false,
            has_overlap: true,
            is_concordant: true,
            shared_findings: ['Effusion', 'Pneumonia', 'Pneumothorax'],
          },
        },
        {
          rank: 2,
          filename: '00000020_001.png',
          similarity_score: 67.64,
          image_url: '/reference_images/00000020_001.png',
          metadata: { finding_labels: 'Fibrosis|Infiltration' },
          concordance: {
            status: 'SECONDARY_OVERLAP',
            label: 'Secondary Overlap',
            matches_primary: false,
            has_overlap: true,
            is_concordant: true,
            shared_findings: ['Infiltration'],
          },
        },
        {
          rank: 3,
          filename: '00000013_023.png',
          similarity_score: 61.8,
          image_url: '/reference_images/00000013_023.png',
          metadata: { finding_labels: 'Infiltration|Mass|Pneumothorax' },
          concordance: {
            status: 'SECONDARY_OVERLAP',
            label: 'Secondary Overlap',
            matches_primary: false,
            has_overlap: true,
            is_concordant: true,
            shared_findings: ['Infiltration', 'Mass', 'Pneumothorax'],
          },
        },
        {
          rank: 4,
          filename: '00000032_012.png',
          similarity_score: 61.71,
          image_url: '/reference_images/00000032_012.png',
          metadata: { finding_labels: 'Atelectasis|Consolidation|Edema|Pneumonia' },
          concordance: {
            status: 'PRIMARY_CONCORDANT',
            label: 'Primary Concordant',
            matches_primary: true,
            has_overlap: true,
            is_concordant: true,
            shared_findings: ['Consolidation'],
          },
        },
        {
          rank: 5,
          filename: '00000013_030.png',
          similarity_score: 61.4,
          image_url: '/reference_images/00000013_030.png',
          metadata: { finding_labels: 'Atelectasis|Cardiomegaly|Emphysema|Mass|Pneumothorax' },
          concordance: {
            status: 'SECONDARY_OVERLAP',
            label: 'Secondary Overlap',
            matches_primary: false,
            has_overlap: true,
            is_concordant: true,
            shared_findings: ['Atelectasis', 'Mass', 'Pneumothorax', 'Emphysema'],
          },
        },
      ],
      disclaimer: 'Test disclaimer',
      summary: 'Retrieved Top-5 similar reference radiographs.',
    };

    render(<SimilarityPanel similarity={case42Sim} />);

    // Top Match Alignment (appears in summary tile)
    expect(screen.getByText(/70\.2%/)).toBeInTheDocument();

    // Primary Concordance (C_d) - appears in summary tile and Consolidation evidence card
    expect(screen.getAllByText('20.0%').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('1 / 5')).toBeInTheDocument();

    // Any-Finding Overlap (O)
    expect(screen.getByText('100.0%')).toBeInTheDocument();
    expect(screen.getByText('5 / 5')).toBeInTheDocument();

    // Assessment Banner
    expect(
      screen.getByText(/1 of 5 \(20.0%\) retrieved reference cases corroborate primary Consolidation finding/i)
    ).toBeInTheDocument();
  });

  it('renders prediction-driven evidence for multiple predicted findings without global 14-disease table', () => {
    const multiFindingSim: SimilarityAssessment = {
      status: 'SUCCESS',
      top_k: 5,
      total_references: 22,
      matches: [
        {
          rank: 1,
          filename: '00000013_010.png',
          similarity_score: 85.0,
          findings: ['Cardiomegaly', 'Effusion'],
          image_url: '/ref1.png',
          metadata: { patient_age: 55, patient_gender: 'M', view_position: 'PA' },
        },
        {
          rank: 2,
          filename: '00000020_001.png',
          similarity_score: 80.0,
          findings: ['Cardiomegaly'],
          image_url: '/ref2.png',
          metadata: { patient_age: 62, patient_gender: 'F', view_position: 'AP' },
        },
        {
          rank: 3,
          filename: '00000013_023.png',
          similarity_score: 75.0,
          findings: ['Effusion'],
          image_url: '/ref3.png',
          metadata: { patient_age: 48, patient_gender: 'M', view_position: 'PA' },
        },
      ],
      predicted_findings: ['Cardiomegaly', 'Effusion'],
      predicted_findings_evidence: [
        {
          predicted_finding: 'Cardiomegaly',
          matching_cases: 2,
          retrieved_cases: 3,
          matching_rate: 66.7,
          weighted_support: 68.8,
          dataset_prevalence: 2.48,
          is_matched: true,
          explanation: '2 of 3 retrieved reference cases (66.7% matching rate, 68.8% weighted support) corroborate predicted Cardiomegaly.',
          matching_references: [
            {
              rank: 1,
              filename: '00000013_010.png',
              similarity_score: 85.0,
              findings: ['Cardiomegaly', 'Effusion'],
              image_url: '/ref1.png',
              metadata: { patient_age: 55, patient_gender: 'M', view_position: 'PA' },
            },
            {
              rank: 2,
              filename: '00000020_001.png',
              similarity_score: 80.0,
              findings: ['Cardiomegaly'],
              image_url: '/ref2.png',
              metadata: { patient_age: 62, patient_gender: 'F', view_position: 'AP' },
            },
          ],
        },
        {
          predicted_finding: 'Effusion',
          matching_cases: 2,
          retrieved_cases: 3,
          matching_rate: 66.7,
          weighted_support: 66.7,
          dataset_prevalence: 11.88,
          is_matched: true,
          explanation: '2 of 3 retrieved reference cases (66.7% matching rate, 66.7% weighted support) corroborate predicted Effusion.',
          matching_references: [
            {
              rank: 1,
              filename: '00000013_010.png',
              similarity_score: 85.0,
              findings: ['Cardiomegaly', 'Effusion'],
              image_url: '/ref1.png',
              metadata: { patient_age: 55, patient_gender: 'M', view_position: 'PA' },
            },
            {
              rank: 3,
              filename: '00000013_023.png',
              similarity_score: 75.0,
              findings: ['Effusion'],
              image_url: '/ref3.png',
              metadata: { patient_age: 48, patient_gender: 'M', view_position: 'PA' },
            },
          ],
        },
      ],
      disclaimer: 'Similarity scores represent image/feature similarity only.',
      summary: 'Retrieved Top-3 similar reference radiographs.',
    };

    render(<SimilarityPanel similarity={multiFindingSim} />);

    // Verify both predicted findings exist with their streamlined progress bars and percentages
    expect(screen.getAllByText('Cardiomegaly').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Effusion').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('2 of 3 matches').length).toBe(2);

    // Verify progress bars are rendered for both findings with their matching percentages
    const progressBars = screen.getAllByRole('progressbar');
    expect(progressBars.length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText('66.7%').length).toBeGreaterThanOrEqual(2);

    // Verify global 14-disease prevalence table is NOT rendered
    expect(screen.queryByText(/Section A: NIH Dataset Statistics/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Section B: Retrieved-Case Label Support/i)).not.toBeInTheDocument();
  });

  it('clearly displays zero matching cases when no retrieved references match the predicted finding', () => {
    const zeroMatchSim: SimilarityAssessment = {
      status: 'SUCCESS',
      top_k: 5,
      total_references: 22,
      matches: [
        {
          rank: 1,
          filename: '00000001_000.png',
          similarity_score: 82.5,
          findings: ['Cardiomegaly'],
          image_url: '/ref1.png',
        },
        {
          rank: 2,
          filename: '00000002_000.png',
          similarity_score: 79.1,
          findings: ['Effusion'],
          image_url: '/ref2.png',
        },
      ],
      predicted_findings: ['Pneumonia'],
      predicted_findings_evidence: [
        {
          predicted_finding: 'Pneumonia',
          matching_cases: 0,
          retrieved_cases: 2,
          matching_rate: 0.0,
          weighted_support: 0.0,
          dataset_prevalence: 1.28,
          is_matched: false,
          explanation: '0 of 2 retrieved reference cases (0.0% matching rate, 0.0% weighted support) match predicted Pneumonia. No reference support found in top matches.',
          matching_references: [],
        },
      ],
      disclaimer: 'Similarity disclaimer',
      summary: 'Retrieved Top-2 similar reference radiographs.',
    };

    render(<SimilarityPanel similarity={zeroMatchSim} />);

    // Verify finding name and 0-matches badge
    expect(screen.getByText('Pneumonia')).toBeInTheDocument();
    expect(screen.getByText('0 of 2 matches')).toBeInTheDocument();

    // Verify progress bar rendered at 0%
    const progressBar = screen.getByRole('progressbar', {
      name: /Matching retrieval rate for Pneumonia: 0\.0%/i,
    });
    expect(progressBar).toBeInTheDocument();
    expect(progressBar).toHaveAttribute('aria-valuenow', '0');
    expect(screen.getAllByText('0.0%').length).toBeGreaterThanOrEqual(1);
  });

  it('renders accessible Top-K segmented control and triggers onKChange when switched', () => {
    const onKChangeMock = vi.fn();
    const mockSim: SimilarityAssessment = {
      status: 'SUCCESS',
      top_k: 5,
      total_references: 2500,
      candidates_searched: 2500,
      matches: [
        { rank: 1, filename: '00000001_000.png', similarity_score: 91.2, findings: ['Cardiomegaly'], image_url: '/ref1.png' },
        { rank: 2, filename: '00000002_000.png', similarity_score: 87.4, findings: ['Cardiomegaly'], image_url: '/ref2.png' },
        { rank: 3, filename: '00000003_000.png', similarity_score: 82.1, findings: ['Effusion'], image_url: '/ref3.png' },
        { rank: 4, filename: '00000004_000.png', similarity_score: 79.5, findings: ['Cardiomegaly'], image_url: '/ref4.png' },
        { rank: 5, filename: '00000005_000.png', similarity_score: 76.3, findings: ['Effusion'], image_url: '/ref5.png' },
      ],
      primary_query_finding: 'Cardiomegaly',
      primary_concordance_rate: 60.0,
      primary_concordant_count: 3,
      any_finding_overlap_rate: 100.0,
      any_finding_overlap_count: 5,
      disclaimer: 'Observational comparison only.',
      summary: 'Retrieved Top-5 similar reference radiographs.',
    };

    render(<SimilarityPanel similarity={mockSim} onKChange={onKChangeMock} />);

    // Segmented control radiogroup and radio items
    const radioGroup = screen.getByRole('radiogroup', { name: /Top-K retrieval depth/i });
    expect(radioGroup).toBeInTheDocument();

    const top5Button = screen.getByRole('radio', { name: /Top-5/i });
    const top10Button = screen.getByRole('radio', { name: /Top-10/i });
    const top20Button = screen.getByRole('radio', { name: /Top-20/i });

    expect(top5Button).toHaveAttribute('aria-checked', 'true');
    expect(top10Button).toHaveAttribute('aria-checked', 'false');
    expect(top20Button).toHaveAttribute('aria-checked', 'false');

    // Click Top-10
    fireEvent.click(top10Button);
    expect(top10Button).toHaveAttribute('aria-checked', 'true');
    expect(top5Button).toHaveAttribute('aria-checked', 'false');
    expect(onKChangeMock).toHaveBeenCalledWith(10);
  });

  it('renders horizontal evidence progress bar with appropriate ARIA attributes', () => {
    const mockSim: SimilarityAssessment = {
      status: 'SUCCESS',
      top_k: 5,
      total_references: 2500,
      matches: [
        { rank: 1, filename: '00000001_000.png', similarity_score: 85.0, findings: ['Cardiomegaly'], image_url: '/ref1.png' },
      ],
      predicted_findings: ['Cardiomegaly'],
      predicted_findings_evidence: [
        {
          predicted_finding: 'Cardiomegaly',
          matching_cases: 1,
          retrieved_cases: 1,
          matching_rate: 100.0,
          weighted_support: 100.0,
          dataset_prevalence: 2.48,
          is_matched: true,
          explanation: '1 of 1 retrieved reference cases (100.0% matching rate) corroborate predicted Cardiomegaly.',
          matching_references: [
            { rank: 1, filename: '00000001_000.png', similarity_score: 85.0, findings: ['Cardiomegaly'], image_url: '/ref1.png' },
          ],
        },
      ],
      disclaimer: 'Observational comparison only.',
      summary: 'Retrieved Top-1 similar reference radiograph.',
    };

    render(<SimilarityPanel similarity={mockSim} />);

    const progressBar = screen.getByRole('progressbar', {
      name: /Matching retrieval rate for Cardiomegaly: 100\.0%/i,
    });
    expect(progressBar).toBeInTheDocument();
    expect(progressBar).toHaveAttribute('aria-valuenow', '100');
    expect(progressBar).toHaveAttribute('aria-valuemin', '0');
    expect(progressBar).toHaveAttribute('aria-valuemax', '100');
    expect(progressBar).toHaveAttribute('aria-valuetext', '100.0%');
  });

  it('renders methodological explanation section and clinical disclaimer', () => {
    const mockSim: SimilarityAssessment = {
      status: 'SUCCESS',
      top_k: 5,
      total_references: 2500,
      matches: [
        { rank: 1, filename: '00000001_000.png', similarity_score: 85.0, findings: ['Cardiomegaly'], image_url: '/ref1.png' },
      ],
      primary_query_finding: 'Cardiomegaly',
      disclaimer: 'Observational comparison only.',
      summary: 'Retrieved Top-1 similar reference radiograph.',
    };

    render(<SimilarityPanel similarity={mockSim} />);

    expect(screen.getByText(/How to Interpret Case-Based Similarity Metrics/i)).toBeInTheDocument();
    expect(screen.getByText('Feature Cosine Similarity')).toBeInTheDocument();
    expect(screen.getByText('Primary Concordance Rate (C_d)')).toBeInTheDocument();
    expect(screen.getByText('Multilabel Any-Finding Overlap (O)')).toBeInTheDocument();
    expect(screen.getByText('Retrieval Match Rate (R_d)')).toBeInTheDocument();
    expect(screen.getByText('Cosine-Weighted Support (V_d)')).toBeInTheDocument();
    expect(screen.getByText('Baseline Population Prevalence (P_d)')).toBeInTheDocument();
    expect(screen.getByText(/Case-based similarity retrieval provides observational context/i)).toBeInTheDocument();
  });
});

