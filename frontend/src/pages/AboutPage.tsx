import type { ReactNode } from 'react';
import { PageContainer } from '@/components/layout/PageContainer';
import { ButtonLink } from '@/components/ui/Button';
import { env } from '@/config/env';
import { useDocumentTitle } from '@/hooks/useDocumentTitle';
import { DISEASES, formatDiseaseName, type DiseaseName } from '@/types/analysis';

const DISEASE_NOTES: Record<DiseaseName, string> = {
  Atelectasis: 'Partial or complete collapse of lung tissue.',
  Cardiomegaly: 'An enlarged cardiac silhouette.',
  Effusion: 'Fluid collecting in the pleural space around the lung.',
  Infiltration: 'A non-specific term for material denser than air within lung tissue.',
  Mass: 'A larger focal lung lesion, conventionally above 3 cm.',
  Nodule: 'A small, rounded lung lesion, conventionally up to 3 cm.',
  Pneumonia: 'Infection that inflames the air sacs of the lung.',
  Pneumothorax: 'Air in the pleural space, which can collapse the lung.',
  Consolidation: 'A region of lung filled with fluid or material instead of air.',
  Edema: 'Excess fluid within lung tissue.',
  Emphysema: 'Damage to the air sacs that leaves the lung over-inflated.',
  Fibrosis: 'Scarring and stiffening of lung tissue.',
  Pleural_Thickening: 'Thickening of the pleura, the lining of the lung.',
  Hernia: 'Abdominal contents protruding through the diaphragm.',
};

const SECTIONS = [
  { id: 'overview', label: 'Overview' },
  { id: 'problem', label: 'Problem and motivation' },
  { id: 'objectives', label: 'Objectives' },
  { id: 'workflow', label: 'System workflow' },
  { id: 'multilabel', label: 'Multi-label classification' },
  { id: 'categories', label: 'The 14 categories' },
  { id: 'quality', label: 'Image quality' },
  { id: 'uncertainty', label: 'Prediction uncertainty' },
  { id: 'ood', label: 'Out-of-distribution detection' },
  { id: 'calibration', label: 'Calibration' },
  { id: 'explainability', label: 'Explainability' },
  { id: 'limitations', label: 'Limitations and intended use' },
  { id: 'team', label: 'Acknowledgement and team' },
] as const;

function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} className="scroll-mt-28 border-t border-line pt-8">
      <h2 id={`${id}-title`} className="text-xl font-semibold tracking-tight sm:text-2xl">
        {title}
      </h2>
      <div className="mt-4 max-w-prose space-y-4 text-[0.9375rem] leading-relaxed text-charcoal">{children}</div>
    </section>
  );
}

export function AboutPage() {
  useDocumentTitle('About');

  return (
    <PageContainer className="py-12 sm:py-20">
      <header className="mb-14 max-w-3xl">
        <p className="eyebrow mb-3">About the project</p>
        <h1 className="h-page">Trust-aware explainable AI for chest X-ray disease detection</h1>
        <p className="lead mt-5">
          MedTrust is a university research project that studies how to present a chest X-ray prediction together with
          the evidence needed to judge its reliability.
        </p>
      </header>

      <div className="grid gap-12 lg:grid-cols-[14rem_minmax(0,1fr)] lg:gap-16">
        <nav aria-label="On this page" className="hidden lg:block">
          <ul className="sticky top-28 space-y-2.5 border-l border-line text-sm">
            {SECTIONS.map((section) => (
              <li key={section.id}>
                <a
                  href={`#${section.id}`}
                  className="-ml-px block border-l border-transparent py-0.5 pl-4 text-muted transition-colors hover:border-ink hover:text-ink"
                >
                  {section.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>

        <div className="min-w-0 space-y-14">
          <Section id="overview" title="Project overview">
            <p>
              MedTrust brings together several components that are usually studied separately: multi-label disease
              detection, image-quality assessment, uncertainty estimation, out-of-distribution detection, probability
              calibration and visual explanation. This web interface presents their outputs for a single uploaded
              chest radiograph in one report.
            </p>
            <p>
              The interface is a front end only. It communicates with an analysis service through a configurable
              REST API and does not contain any model.
            </p>
          </Section>

          <Section id="problem" title="Problem statement and motivation">
            <p>
              Deep learning models can score highly on benchmark datasets and still behave unpredictably on an
              individual image, for example when the image is blurred, mis-positioned or unlike the training data. A
              bare probability gives no sign of this.
            </p>
            <p>
              Models are also often over-confident: a reported 90% may not correspond to being right nine times in ten.
              For a system to be useful to researchers and, eventually, to clinicians, it must show when its output
              deserves less trust.
            </p>
          </Section>

          <Section id="objectives" title="Objectives">
            <ul className="list-disc space-y-2 pl-5">
              <li>Estimate the probability of each of 14 thoracic findings from a single chest radiograph.</li>
              <li>Report whether the image is of adequate quality and resembles the data the model was developed on.</li>
              <li>Quantify prediction uncertainty and calibrate probabilities so they can be read more honestly.</li>
              <li>Provide visual explanations and, where calculated, measures of how stable and faithful they are.</li>
              <li>Present all of this in one transparent report that states clearly when something was not evaluated.</li>
            </ul>
          </Section>

          <Section id="workflow" title="System workflow">
            <ol className="list-decimal space-y-2 pl-5">
              <li>You select a chest X-ray image and submit it from the analysis page.</li>
              <li>The front end sends the image to the configured analysis service.</li>
              <li>
                The service returns predictions with image-quality, uncertainty, out-of-distribution, calibration and
                explanation results.
              </li>
              <li>The interface shows a preview, and the full report presents each result with a short explanation.</li>
            </ol>
          </Section>

          <Section id="multilabel" title="Multi-label chest X-ray classification">
            <p>
              A single radiograph can show several conditions at the same time, such as an effusion together with
              atelectasis. Multi-label classification therefore predicts an independent probability for every category
              rather than choosing one label, and the probabilities do not need to sum to 100%.
            </p>
          </Section>

          <section id="categories" aria-labelledby="categories-title" className="scroll-mt-28 border-t border-line pt-8">
            <h2 id="categories-title" className="text-xl font-semibold tracking-tight sm:text-2xl">
              The 14 disease categories
            </h2>
            <p className="mt-4 max-w-prose text-[0.9375rem] leading-relaxed text-charcoal">
              Every list and table in MedTrust uses this fixed order. The notes below are general descriptions, not
              statements about any particular image.
            </p>
            <ol className="mt-6 grid gap-x-10 border-b border-line sm:grid-cols-2">
              {DISEASES.map((disease, index) => (
                <li key={disease} className="grid grid-cols-[2rem_1fr] gap-x-2 border-t border-line/70 py-3 text-sm">
                  <span className="tabular text-xs text-muted">{String(index + 1).padStart(2, '0')}</span>
                  <span>
                    <span className="font-medium">{formatDiseaseName(disease)}</span>
                    <span className="mt-0.5 block text-[0.8125rem] leading-snug text-muted">{DISEASE_NOTES[disease]}</span>
                  </span>
                </li>
              ))}
            </ol>
          </section>

          <Section id="quality" title="Image-quality assessment">
            <p>
              Measures such as sharpness, contrast, brightness and noise describe whether an image carries enough
              information to be analysed reliably. Poor quality does not make a prediction wrong, but it is a reason
              for more caution.
            </p>
          </Section>

          <Section id="uncertainty" title="Prediction uncertainty">
            <p>
              Uncertainty describes how settled a model&rsquo;s output is. It can reflect noise in the data
              (aleatoric) or limits in what the model has learned (epistemic). A high value means the output could
              easily have been different; a low value does not mean the prediction is correct.
            </p>
          </Section>

          <Section id="ood" title="Out-of-distribution detection">
            <p>
              An out-of-distribution (OOD) detector estimates whether an image resembles the data the model was
              developed on. An OOD score is a measure of unfamiliarity, not a probability of disease, and images that
              score as unfamiliar warrant extra caution.
            </p>
          </Section>

          <Section id="calibration" title="Calibration">
            <p>
              A model is well calibrated when its stated probabilities match observed frequencies. Calibration is
              commonly summarised with Expected Calibration Error (ECE), the Brier score and negative log-likelihood,
              and visualised as a reliability diagram. MedTrust shows raw and calibrated probabilities side by side
              when both are available.
            </p>
          </Section>

          <Section id="explainability" title="Explainability">
            <p>
              Gradient-based visual explanations such as Grad-CAM highlight regions that influenced a model output.
              They are tools for inspecting model behaviour and can highlight areas unrelated to disease. Stability
              (consistency under small input changes) and faithfulness (whether the highlighted regions actually drive
              the output) indicate how far an explanation can be relied on, and are shown only when calculated.
            </p>
          </Section>

          <Section id="limitations" title="Limitations and intended use">
            <ul className="list-disc space-y-2 pl-5">
              <li>MedTrust is intended for research and demonstration only.</li>
              <li>It is not a medical device and has not been clinically validated or approved by any regulator.</li>
              <li>Outputs are statistical estimates and must not be used for diagnosis, triage or treatment.</li>
              <li>Results may be unreliable for images unlike the development data, regardless of reported scores.</li>
              <li>Explanations are not evidence that a finding is present.</li>
              <li>Do not upload identifiable patient information.</li>
            </ul>
          </Section>

          <Section id="team" title="Acknowledgement and team">
            <p>
              MedTrust is a research project
              {env.institution ? ` carried out at ${env.institution}` : ''}. It builds on published research in
              uncertainty estimation, calibration and explainable AI, and on publicly available chest radiograph
              datasets and open-source software.
            </p>
            {env.team.length > 0 ? (
              <ul className="divide-y divide-line border-y border-line">
                {env.team.map((member) => (
                  <li key={member.name} className="flex flex-wrap items-baseline justify-between gap-2 py-3 text-sm">
                    <span className="font-medium text-ink">{member.name}</span>
                    {member.role && <span className="text-muted">{member.role}</span>}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="rounded-md border border-dashed border-line bg-subtle/40 px-4 py-3 text-sm text-muted">
                Team details have not been configured. Set <code className="font-mono text-xs">VITE_PROJECT_TEAM</code> and{' '}
                <code className="font-mono text-xs">VITE_INSTITUTION</code> in your <code className="font-mono text-xs">.env</code> file
                to show them here.
              </p>
            )}
            {env.contactEmail && (
              <p>
                Contact:{' '}
                <a className="font-medium text-ink underline underline-offset-4" href={`mailto:${env.contactEmail}`}>
                  {env.contactEmail}
                </a>
              </p>
            )}
          </Section>

          <div className="border-t border-ink pt-8">
            <ButtonLink to="/analyze">Analyze an X-Ray</ButtonLink>
          </div>
        </div>
      </div>
    </PageContainer>
  );
}
