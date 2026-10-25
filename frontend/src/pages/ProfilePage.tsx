import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogOut } from 'lucide-react';
import { PageContainer } from '@/components/layout/PageContainer';
import { Alert } from '@/components/ui/Alert';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { useAuth } from '@/hooks/useAuth';
import { useDocumentTitle } from '@/hooks/useDocumentTitle';
import { formatDate, initials } from '@/lib/format';

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-1 py-4 sm:grid-cols-[11rem_1fr] sm:gap-6">
      <dt className="text-sm text-muted">{label}</dt>
      <dd className="min-w-0 break-words text-sm font-medium">{children}</dd>
    </div>
  );
}

const NOT_PROVIDED = <span className="font-normal text-muted">Not provided by the server</span>;

export function ProfilePage() {
  useDocumentTitle('Profile');
  const { user, logout, isDemoAuth } = useAuth();
  const navigate = useNavigate();
  const [signingOut, setSigningOut] = useState(false);

  if (!user) return null;

  const handleLogout = async () => {
    setSigningOut(true);
    await logout();
    navigate('/', { replace: true });
  };

  return (
    <PageContainer className="py-12 sm:py-16">
      <header className="mb-10 max-w-2xl">
        <p className="eyebrow mb-3">Account</p>
        <h1 className="h-page">Profile</h1>
      </header>

      {isDemoAuth && (
        <Alert tone="warning" title="Demo session" className="mb-8 max-w-3xl">
          You are signed in to a local, non-secure demo session. It is not real authentication and is cleared when this
          browser tab is closed.
        </Alert>
      )}

      <div className="grid max-w-4xl gap-10 md:grid-cols-[auto_1fr] md:gap-14">
        <div
          aria-hidden="true"
          className="flex h-24 w-24 items-center justify-center rounded-full bg-ink text-2xl font-semibold tracking-tight text-surface"
        >
          {initials(user.fullName)}
        </div>

        <div className="min-w-0">
          <dl className="divide-y divide-line border-y border-line">
            <Row label="Full name">{user.fullName}</Row>
            <Row label="Email address">{user.email}</Row>
            <Row label="Account status">
              {user.status ? <Badge tone={user.isDemo ? 'outline' : 'success'}>{user.status}</Badge> : NOT_PROVIDED}
            </Row>
            <Row label="Role">{user.role ?? NOT_PROVIDED}</Row>
            <Row label="Member since">{user.createdAt ? formatDate(user.createdAt) : NOT_PROVIDED}</Row>
            <Row label="User ID">
              <span className="font-mono text-xs">{user.id}</span>
            </Row>
          </dl>

          <p className="mt-6 max-w-prose text-sm leading-relaxed text-muted">
            Editing your profile, changing your password and managing preferences are not available in this version of
            MedTrust.
          </p>

          <div className="mt-8">
            <Button variant="secondary" loading={signingOut} onClick={handleLogout} leadingIcon={<LogOut aria-hidden="true" className="h-4 w-4" />}>
              Sign out
            </Button>
          </div>
        </div>
      </div>
    </PageContainer>
  );
}
