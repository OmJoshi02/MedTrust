import { useState, type FormEvent } from 'react';
import { Link, Navigate, useNavigate } from 'react-router-dom';
import { Alert } from '@/components/ui/Alert';
import { Button } from '@/components/ui/Button';
import { PasswordField } from '@/components/ui/PasswordField';
import { TextField } from '@/components/ui/TextField';
import { AuthLayout } from '@/components/layout/AuthLayout';
import { env } from '@/config/env';
import { useAuth } from '@/hooks/useAuth';
import { useDocumentTitle } from '@/hooks/useDocumentTitle';
import { validateEmail } from '@/lib/validation';
import { describeError, isApiError } from '@/services';

interface FieldErrors {
  email?: string | null;
  password?: string | null;
}

export function LoginPage() {
  useDocumentTitle('Log In');
  const { status, login, sessionExpired, clearSessionExpired } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [showRecovery, setShowRecovery] = useState(false);

  if (status === 'authenticated' && !submitting) return <Navigate to="/analyze" replace />;

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (submitting) return;

    const next: FieldErrors = {
      email: validateEmail(email),
      password: password === '' ? 'Enter your password.' : null,
    };
    setErrors(next);
    setFormError(null);
    if (next.email || next.password) return;

    setSubmitting(true);
    clearSessionExpired();
    try {
      await login({ email, password, rememberMe: env.enableRememberMe ? rememberMe : undefined });
      navigate('/analyze', { replace: true });
    } catch (error) {
      if (isApiError(error) && (error.kind === 'unauthorized' || error.status === 400 || error.status === 422)) {
        setFormError('The email or password is incorrect. Check your details and try again.');
      } else if (isApiError(error) && error.kind === 'rate_limited') {
        setFormError('Too many sign-in attempts. Please wait a moment and try again.');
      } else {
        setFormError(describeError(error));
      }
      setSubmitting(false);
    }
  };

  return (
    <AuthLayout
      title="Log in"
      subtitle="Sign in to analyze X-rays and review your previous analyses."
      footer={
        <>
          New to MedTrust?{' '}
          <Link to="/signup" className="font-medium text-ink underline underline-offset-4">
            Create an account
          </Link>
        </>
      }
    >
      {env.demoMode && (
        <Alert tone="info" title="Demo sign-in" className="mb-5">
          Any well-formed email and a non-empty password will open a local demo session. This is not real authentication.
        </Alert>
      )}
      {sessionExpired && (
        <Alert tone="warning" title="Your session has expired" className="mb-5">
          Please sign in again to continue.
        </Alert>
      )}
      {formError && (
        <Alert tone="danger" title="Sign-in failed" className="mb-5">
          {formError}
        </Alert>
      )}

      <div className="mb-4 rounded-md border border-line bg-subtle/60 p-3 text-xs leading-relaxed text-muted">
        <div className="flex items-center justify-between">
          <span className="font-medium text-ink">Lead Investigator Account:</span>
          <button
            type="button"
            onClick={() => {
              setEmail('doctor@medtrust.ai');
              setPassword('MedTrust2026!');
            }}
            className="rounded bg-ink px-2 py-0.5 text-xs font-medium text-surface transition-opacity hover:opacity-90"
          >
            Auto-fill Credentials
          </button>
        </div>
        <div className="mt-1 font-mono text-[11px]">
          doctor@medtrust.ai &middot; MedTrust2026!
        </div>
      </div>

      <form onSubmit={onSubmit} noValidate className="space-y-4">
        <TextField
          label="Email address"
          type="email"
          name="email"
          autoComplete="email"
          inputMode="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          error={errors.email}
          required
        />
        <PasswordField
          label="Password"
          name="password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          error={errors.password}
          required
        />

        <div className="flex flex-wrap items-center justify-between gap-3 text-sm">
          {env.enableRememberMe ? (
            <label className="inline-flex items-center gap-2">
              <input
                type="checkbox"
                checked={rememberMe}
                onChange={(e) => setRememberMe(e.target.checked)}
                className="h-4 w-4 rounded-sm border-line accent-ink"
              />
              Remember me
            </label>
          ) : (
            <span />
          )}
          <button
            type="button"
            onClick={() => setShowRecovery((v) => !v)}
            aria-expanded={showRecovery}
            aria-controls="recovery-note"
            className="rounded text-muted underline underline-offset-4 hover:text-ink"
          >
            Forgot password?
          </button>
        </div>

        {showRecovery && (
          <div id="recovery-note">
            <Alert tone="info" title="Password recovery is not available">
              This version of MedTrust does not include self-service password recovery. Contact the project
              administrator{env.contactEmail ? ` at ${env.contactEmail}` : ''} to reset your password.
            </Alert>
          </div>
        )}

        <Button type="submit" size="lg" className="w-full" loading={submitting}>
          {submitting ? 'Signing in…' : 'Log in'}
        </Button>
      </form>
    </AuthLayout>
  );
}
