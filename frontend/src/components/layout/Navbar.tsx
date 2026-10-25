import { useEffect, useState } from 'react';
import { Link, NavLink, useLocation, useNavigate } from 'react-router-dom';
import { LogOut, Menu, X } from 'lucide-react';
import { Logo } from '@/components/ui/Logo';
import { buttonStyles } from '@/components/ui/buttonStyles';
import { useAuth } from '@/hooks/useAuth';
import { cn } from '@/lib/cn';
import { MEMBER_LINKS, PUBLIC_LINKS } from './navigation';

const linkClass = ({ isActive }: { isActive: boolean }) =>
  cn(
    'relative rounded px-2.5 py-1.5 text-sm transition-colors',
    isActive ? 'text-ink' : 'text-muted hover:text-ink',
  );

export function Navbar() {
  const { status, user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  const signedIn = status === 'authenticated' && user !== null;
  const links = signedIn ? MEMBER_LINKS : PUBLIC_LINKS;

  useEffect(() => {
    setOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open]);

  const handleLogout = async () => {
    setOpen(false);
    await logout();
    navigate('/', { replace: true });
  };

  return (
    <header
      className={cn(
        'no-print sticky top-0 z-40 border-b transition-[background-color,border-color] duration-200',
        scrolled || open
          ? 'border-line bg-canvas/80 backdrop-blur-xl'
          : 'border-transparent bg-canvas/0',
      )}
    >
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-3 focus:z-50 focus:rounded-md focus:bg-ink focus:px-3 focus:py-2 focus:text-sm focus:text-surface"
      >
        Skip to content
      </a>
      <div className="page-container flex h-16 items-center justify-between gap-6">
        <Link to="/" aria-label="MedTrust home" className="rounded">
          <Logo />
        </Link>

        <nav aria-label="Primary" className="hidden items-center gap-1 md:flex">
          {links.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.to === '/'} className={linkClass}>
              {({ isActive }) => (
                <>
                  {item.label}
                  {isActive && (
                    <span aria-hidden="true" className="absolute inset-x-2.5 -bottom-[13px] h-px bg-ink" />
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="hidden items-center gap-2 md:flex">
          {status === 'loading' ? (
            <span className="h-8 w-24" aria-hidden="true" />
          ) : signedIn ? (
            <button type="button" onClick={handleLogout} className={buttonStyles({ variant: 'secondary', size: 'sm' })}>
              <LogOut aria-hidden="true" className="h-3.5 w-3.5" />
              Logout
            </button>
          ) : (
            <>
              <Link to="/login" className={buttonStyles({ variant: 'ghost', size: 'sm' })}>
                Log In
              </Link>
              <Link to="/signup" className={buttonStyles({ variant: 'primary', size: 'sm' })}>
                Sign Up
              </Link>
            </>
          )}
        </div>

        <button
          type="button"
          className="-mr-2 inline-flex h-10 w-10 items-center justify-center rounded-md text-ink hover:bg-subtle md:hidden"
          aria-label={open ? 'Close menu' : 'Open menu'}
          aria-expanded={open}
          aria-controls="mobile-menu"
          onClick={() => setOpen((value) => !value)}
        >
          {open ? <X aria-hidden="true" className="h-5 w-5" /> : <Menu aria-hidden="true" className="h-5 w-5" />}
        </button>
      </div>

      {open && (
        <div id="mobile-menu" className="animate-fade-in border-t border-line bg-canvas md:hidden">
          <nav aria-label="Mobile" className="page-container flex flex-col py-3">
            {links.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  cn(
                    'border-b border-line/70 py-3 text-base',
                    isActive ? 'font-medium text-ink' : 'text-muted',
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
            <div className="flex gap-2 pt-4">
              {status === 'loading' ? null : signedIn ? (
                <button
                  type="button"
                  onClick={handleLogout}
                  className={buttonStyles({ variant: 'secondary', size: 'md', className: 'flex-1' })}
                >
                  <LogOut aria-hidden="true" className="h-4 w-4" />
                  Logout
                </button>
              ) : (
                <>
                  <Link to="/login" className={buttonStyles({ variant: 'secondary', size: 'md', className: 'flex-1' })}>
                    Log In
                  </Link>
                  <Link to="/signup" className={buttonStyles({ variant: 'primary', size: 'md', className: 'flex-1' })}>
                    Sign Up
                  </Link>
                </>
              )}
            </div>
          </nav>
        </div>
      )}
    </header>
  );
}
