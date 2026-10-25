import { Link } from 'react-router-dom';
import { env } from '@/config/env';
import { Logo } from '@/components/ui/Logo';
import { ApiStatusIndicator } from '@/components/feedback/ApiStatusIndicator';
import { useAuth } from '@/hooks/useAuth';

export function Footer() {
  const { status, user } = useAuth();
  const signedIn = status === 'authenticated' && user !== null;
  const year = new Date().getFullYear();

  return (
    <footer className="no-print on-dark mt-24 bg-dark text-surface/70">
      <div className="page-container py-14">
        <div className="grid gap-12 md:grid-cols-[1.4fr_1fr_1fr]">
          <div>
            <Logo tone="light" />
            <p className="mt-4 max-w-sm text-sm leading-relaxed">
              Trust-aware, explainable multi-label chest X-ray analysis. A research project exploring how to
              evaluate a prediction&rsquo;s reliability alongside the prediction itself.
            </p>
            <div className="mt-6">
              <ApiStatusIndicator className="text-surface/60" />
            </div>
          </div>

          <nav aria-label="Footer">
            <h2 className="eyebrow !text-surface/50">Explore</h2>
            <ul className="mt-4 space-y-2.5 text-sm">
              <li>
                <Link className="transition-colors hover:text-surface" to="/">
                  Home
                </Link>
              </li>
              <li>
                <Link className="transition-colors hover:text-surface" to="/analyze">
                  Analyze X-Ray
                </Link>
              </li>
              <li>
                <Link className="transition-colors hover:text-surface" to="/about">
                  About
                </Link>
              </li>
              {signedIn ? (
                <>
                  <li>
                    <Link className="transition-colors hover:text-surface" to="/history">
                      Recent Analyses
                    </Link>
                  </li>
                  <li>
                    <Link className="transition-colors hover:text-surface" to="/profile">
                      Profile
                    </Link>
                  </li>
                </>
              ) : (
                <>
                  <li>
                    <Link className="transition-colors hover:text-surface" to="/login">
                      Log In
                    </Link>
                  </li>
                  <li>
                    <Link className="transition-colors hover:text-surface" to="/signup">
                      Sign Up
                    </Link>
                  </li>
                </>
              )}
            </ul>
          </nav>

          <div>
            <h2 className="eyebrow !text-surface/50">Project</h2>
            <dl className="mt-4 space-y-2.5 text-sm">
              <div>
                <dt className="sr-only">Type</dt>
                <dd>University research project</dd>
              </div>
              {env.institution && (
                <div>
                  <dt className="sr-only">Institution</dt>
                  <dd>{env.institution}</dd>
                </div>
              )}
              {env.contactEmail && (
                <div>
                  <dt className="sr-only">Contact</dt>
                  <dd>
                    <a className="underline-offset-4 transition-colors hover:text-surface hover:underline" href={`mailto:${env.contactEmail}`}>
                      {env.contactEmail}
                    </a>
                  </dd>
                </div>
              )}
              <div>
                <dt className="sr-only">Status</dt>
                <dd>Not clinically validated</dd>
              </div>
            </dl>
          </div>
        </div>

        <div className="mt-12 border-t border-white/10 pt-6 text-xs leading-relaxed text-surface/50">
          <p>
            <strong className="font-medium text-surface/70">Research use only.</strong> MedTrust is not a medical
            device and is not intended for diagnosis, treatment or clinical decision-making. Outputs must not replace
            the judgement of a qualified clinician.
          </p>
          <p className="mt-3">&copy; {year} MedTrust research project.</p>
        </div>
      </div>
    </footer>
  );
}
