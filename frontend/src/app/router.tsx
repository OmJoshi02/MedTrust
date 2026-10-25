import { createBrowserRouter, createMemoryRouter, type RouteObject } from 'react-router-dom';
import { Layout } from '@/components/layout/Layout';
import { ProtectedRoute } from '@/components/layout/ProtectedRoute';
import { AboutPage } from '@/pages/AboutPage';
import { AnalyzePage } from '@/pages/AnalyzePage';
import {
  ApiUnavailablePage,
  AppErrorPage,
  NotFoundPage,
  UnauthorizedPage,
} from '@/pages/ErrorPages';
import { HistoryPage } from '@/pages/HistoryPage';
import { HomePage } from '@/pages/HomePage';
import { LoginPage } from '@/pages/LoginPage';
import { ProfilePage } from '@/pages/ProfilePage';
import { ReportPage } from '@/pages/ReportPage';
import { SignupPage } from '@/pages/SignupPage';

export const routes: RouteObject[] = [
  {
    element: <Layout />,
    errorElement: <AppErrorPage />,
    children: [
      { index: true, element: <HomePage /> },
      { path: 'analyze', element: <AnalyzePage /> },
      { path: 'about', element: <AboutPage /> },
      { path: 'login', element: <LoginPage /> },
      { path: 'signup', element: <SignupPage /> },
      { path: 'report/:analysisId', element: <ReportPage /> },
      { path: 'unauthorized', element: <UnauthorizedPage /> },
      { path: 'service-unavailable', element: <ApiUnavailablePage /> },
      {
        element: <ProtectedRoute />,
        children: [
          { path: 'history', element: <HistoryPage /> },
          { path: 'profile', element: <ProfilePage /> },
        ],
      },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
];

export const createAppRouter = () => createBrowserRouter(routes);

/** Used by tests, so routes render without a real browser history. */
export const createTestRouter = (initialEntries: string[]) => createMemoryRouter(routes, { initialEntries });
