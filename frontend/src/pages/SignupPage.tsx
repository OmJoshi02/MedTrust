import { useState, type FormEvent } from 'react';
import { Link, Navigate } from 'react-router-dom';
import { CircleCheck, Circle } from 'lucide-react';
import { Alert } from '@/components/ui/Alert';
import { Button, ButtonLink } from '@/components/ui/Button';
import { PasswordField } from '@/components/ui/PasswordField';
import { TextField } from '@/components/ui/TextField';
import { AuthLayout } from '@/components/layout/AuthLayout';
import { env } from '@/config/env';
import { useAuth } from '@/hooks/useAuth';
import { useDocumentTitle } from '@/hooks/useDocumentTitle';
import { cn } from '@/lib/cn';
import {
  passwordRules,
  validateConfirmPassword,
  validateEmail,
  validateFullName,
  validateNewPassword,
} from '@/lib/validation';
import { describeError, isApiError } from '@/services';

interface FieldErrors {
  fullName?: string | null;
  email?: string | null;
  password?: string | null;
  confirm?: string | null;
  terms?: string | null;
}

export function SignupPage() {
  useDocumentTitle('Sign Up');
  const { status, register } = useAuth();

  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [accepted, setAccepted] = useState(false);
  const [errors, setErrors] = useState<FieldErrors>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [created, setCreated] = useState<{ message?: string } | null>(null);

  if (status === 'authenticated') return <Navigate to="/analyze" replace />;

  const rules = passwordRules(password);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (submitting) return;

    const next: FieldErrors = {
      fullName: validateFullName(fullName),
      email: validateEmail(email),
      password: validateNewPassword(password),
      confirm: validateConfirmPassword(password, confirm),
      terms: accepted ? null : 'You must acknowledge the terms and privacy notice to continue.',
    };
    setErrors(next);
    setFormError(null);
    if (Object.values(next).some(Boolean)) return;

    setSubmitting(true);
    try {
      const result = await register({ fullName, email, password });
      setCreated({ message: result.message });
      setPassword('');
      setConfirm('');
    } catch (error) {
      if (isApiError(error) && error.fieldErrors) {
        setErrors((current) => ({
          ...current,
          email: error.fieldErrors?.email ?? current.email,
          password: error.fieldErrors?.password ?? current.password,
          fullName: error.fieldErrors?.full_name ?? error.fieldErrors?.fullName ?? current.fullName,
        }));
      }
      setFormError(describeError(error));
    } finally {
      setSubmitting(false);
    }
  };

  if (created) {
    return (
      <AuthLayout
        title="Account created"
        subtitle="Your account is ready. Log in to start analyzing X-rays."
        footer={<>You can close this page after signing in.</>}
      >
        <Alert tone="success" title="Registration succeeded">
          {created.message ?? 'The server confirmed that your account was created.'}
        </Alert>
        <ButtonLink to="/login" size="lg" className="mt-6 w-full">
          Continue to log in
        </ButtonLink>
      </AuthLayout>
    );
  }

  return (
    <AuthLayout
      title="Create an account"
      subtitle="Register to keep a record of your analyses."
      footer={
        <>
          Already have an account?{' '}
          <Link to="/login" className="font-medium text-ink underline underline-offset-4">
            Log in
          </Link>
        </>
      }
    >
      {env.demoMode && (
        <Alert tone="info" title="Demo sign-up" className="mb-5">
          No account is created on a server. A temporary demo profile is kept for this browser session only, and your
          password is never stored.
        </Alert>
      )}
      {formError && (
        <Alert tone="danger" title="Account could not be created" className="mb-5">
          {formError}
        </Alert>
      )}

      <form onSubmit={onSubmit} noValidate className="space-y-4">
        <TextField
          label="Full name"
          name="name"
          autoComplete="name"
          value={fullName}
          onChange={(e) => setFullName(e.target.value)}
          error={errors.fullName}
          required
        />
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
        <div>
          <PasswordField
            label="Password"
            name="new-password"
            autoComplete="new-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            error={errors.password}
            aria-describedby="password-rules"
            required
          />
          <ul id="password-rules" aria-label="Password requirements" className="mt-3 grid gap-1.5 sm:grid-cols-2">
            {rules.map((rule) => (
              <li
                key={rule.id}
                className={cn('flex items-center gap-2 text-xs', rule.met ? 'text-ink' : 'text-muted')}
              >
                {rule.met ? (
                  <CircleCheck aria-hidden="true" className="h-3.5 w-3.5 text-success" />
                ) : (
                  <Circle aria-hidden="true" className="h-3.5 w-3.5" />
                )}
                {rule.label}
                <span className="sr-only">{rule.met ? ' (met)' : ' (not met)'}</span>
              </li>
            ))}
          </ul>
        </div>
        <PasswordField
          label="Confirm password"
          name="confirm-password"
          autoComplete="new-password"
          value={confirm}
          onChange={(e) => setConfirm(e.target.value)}
          error={errors.confirm}
          required
        />

        <div>
          <label className="flex items-start gap-2.5 text-sm leading-snug">
            <input
              type="checkbox"
              checked={accepted}
              onChange={(e) => setAccepted(e.target.checked)}
              aria-invalid={errors.terms ? true : undefined}
              className="mt-0.5 h-4 w-4 shrink-0 rounded-sm border-line accent-ink"
            />
            <span>
              I understand MedTrust is a research tool that is not intended for diagnosis, and I agree to the terms of use
              and privacy notice that apply to this project.
            </span>
          </label>
          {errors.terms && (
            <p role="alert" className="mt-1.5 text-[0.8125rem] text-danger">
              {errors.terms}
            </p>
          )}
        </div>

        <Button type="submit" size="lg" className="w-full" loading={submitting}>
          {submitting ? 'Creating account…' : 'Create account'}
        </Button>
      </form>
    </AuthLayout>
  );
}
