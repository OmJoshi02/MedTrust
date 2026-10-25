import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { renderApp, signIn, siteHeader } from './utils';

describe('protected routes', () => {
  it('redirects logged-out users from /history to the login page', async () => {
    const { router } = renderApp('/history');
    expect(await screen.findByRole('heading', { level: 1, name: /log in/i })).toBeInTheDocument();
    expect(router.state.location.pathname).toBe('/login');
    expect(screen.queryByRole('heading', { name: /recent analyses/i })).not.toBeInTheDocument();
  });

  it('redirects logged-out users from /profile to the login page', async () => {
    const { router } = renderApp('/profile');
    await screen.findByRole('heading', { level: 1, name: /log in/i });
    expect(router.state.location.pathname).toBe('/login');
  });

  it('shows history to signed-in users, labelled as sample data in demo mode', async () => {
    signIn();
    renderApp('/history');
    expect(await screen.findByRole('heading', { level: 1, name: /recent analyses/i })).toBeInTheDocument();
    expect(await screen.findByText('sample_pa_view_01.png')).toBeInTheDocument();
    expect(screen.getByText(/these records are sample data/i)).toBeInTheDocument();
    expect(screen.getAllByText('Sample').length).toBeGreaterThan(0);
  });

  it('filters history by search text and shows an empty state with no matches', async () => {
    const user = userEvent.setup();
    signIn();
    renderApp('/history');
    await screen.findByText('sample_pa_view_01.png');

    await user.type(screen.getByLabelText('Search'), 'zzz-no-match');
    expect(await screen.findByText(/no analyses match your search/i)).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: /clear search/i }));
    expect(await screen.findByText('sample_pa_view_01.png')).toBeInTheDocument();
  });

  it('links history rows to their reports', async () => {
    signIn();
    renderApp('/history');
    const link = (await screen.findByText('sample_pa_view_01.png')).closest('a');
    expect(link).toHaveAttribute('href', '/report/demo-sample-001');
  });

  it('shows profile details to signed-in users and signs out to the home page', async () => {
    const user = userEvent.setup();
    signIn();
    const { router } = renderApp('/profile');

    expect(await screen.findByRole('heading', { level: 1, name: 'Profile' })).toBeInTheDocument();
    expect(screen.getByText('tester@example.com')).toBeInTheDocument();
    expect(screen.getByText(/local, non-secure demo session/i)).toBeInTheDocument();

    await user.click(within(screen.getByRole('main')).getByRole('button', { name: /sign out/i }));
    await waitFor(() => expect(router.state.location.pathname).toBe('/'));

    // The session is gone: protected pages redirect again.
    await router.navigate('/history');
    await screen.findByRole('heading', { level: 1, name: /log in/i });
  });

  it('logs out from the navbar and returns to the home page', async () => {
    const user = userEvent.setup();
    signIn();
    const { router } = renderApp('/analyze');

    await user.click(await within(siteHeader()).findByRole('button', { name: /logout/i }));
    await waitFor(() => expect(router.state.location.pathname).toBe('/'));
    expect(await within(siteHeader()).findByRole('link', { name: 'Log In' })).toBeInTheDocument();
  });
});
