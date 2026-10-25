import { useEffect } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { DemoBanner } from '@/components/feedback/DemoBanner';
import { Footer } from './Footer';
import { Navbar } from './Navbar';

/** Application shell: demo banner, navigation, routed content, footer. */
export function Layout() {
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0 });
    // Move focus to the main region on route change for keyboard and screen-reader users.
    document.getElementById('main')?.focus({ preventScroll: true });
  }, [pathname]);

  return (
    <div className="flex min-h-screen flex-col">
      <DemoBanner />
      <Navbar />
      <main id="main" tabIndex={-1} className="flex-1 outline-none">
        <div key={pathname} className="animate-page-in">
          <Outlet />
        </div>
      </main>
      <Footer />
    </div>
  );
}
