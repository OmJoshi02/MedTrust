import { render, screen } from '@testing-library/react';
import { RouterProvider } from 'react-router-dom';
import { createTestRouter } from '@/app/router';
import { Providers } from '@/app/providers';
import { demoSession } from '@/demo/demoStore';
import type { User } from '@/types/auth';

export const TEST_USER: User = {
  id: 'demo-tester@example.com',
  email: 'tester@example.com',
  fullName: 'Test User',
  status: 'demo session',
  isDemo: true,
};

/** Seeds a demo session, as if the user had already signed in. */
export function signIn(user: User = TEST_USER): void {
  demoSession.set(user);
}

/** Renders the whole application at a route, with real providers and routing. */
export function renderApp(path = '/') {
  const router = createTestRouter([path]);
  const utils = render(
    <Providers>
      <RouterProvider router={router} />
    </Providers>,
  );
  return { ...utils, router };
}

/** Builds a File whose leading bytes match a real image signature. */
export function makeImageFile(
  name = 'chest.png',
  type = 'image/png',
  kind: 'png' | 'jpeg' | 'webp' | 'text' = 'png',
  size = 64,
): File {
  const bytes = new Uint8Array(size);
  if (kind === 'png') bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  if (kind === 'jpeg') bytes.set([0xff, 0xd8, 0xff, 0xe0]);
  if (kind === 'webp') bytes.set([0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x45, 0x42, 0x50]);
  if (kind === 'text') bytes.set(new TextEncoder().encode('not an image'));
  return new File([bytes], name, { type });
}

/** The site navigation bar. Page-level <header> elements also match role=banner, so take the first. */
export const siteHeader = (): HTMLElement => screen.getAllByRole('banner')[0] as HTMLElement;
