import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { renderApp, signIn, siteHeader } from './utils';

describe('navigation and route rendering', () => {
  it('renders the home page with the required heading and calls to action', async () => {
    renderApp('/');
    expect(
      await screen.findByRole('heading', { level: 1, name: /understand the prediction\. evaluate its reliability\./i }),
    ).toBeInTheDocument();
    expect(screen.getAllByRole('link', { name: 'Analyze an X-Ray' })[0]).toHaveAttribute('href', '/analyze');
    expect(screen.getByRole('link', { name: 'Explore MedTrust' })).toHaveAttribute('href', '#preview');
  });

  it('shows the logged-out navigation exactly as specified', async () => {
    renderApp('/');
    const nav = await screen.findByRole('navigation', { name: 'Primary' });
    const labels = within(nav)
      .getAllByRole('link')
      .map((link) => link.textContent);
    expect(labels).toEqual(['Home', 'Analyze X-Ray', 'About']);

    const header = siteHeader();
    expect(within(header).getAllByRole('link', { name: 'Log In' }).length).toBeGreaterThan(0);
    expect(within(header).getAllByRole('link', { name: 'Sign Up' }).length).toBeGreaterThan(0);
    expect(within(header).queryByRole('link', { name: /documentation/i })).not.toBeInTheDocument();
    expect(within(header).queryByRole('link', { name: /recent analyses/i })).not.toBeInTheDocument();
  });

  it('shows the logged-in navigation with Recent Analyses, Profile and Logout', async () => {
    signIn();
    renderApp('/');
    const nav = await screen.findByRole('navigation', { name: 'Primary' });
    await within(nav).findByRole('link', { name: 'Recent Analyses' });
    const labels = within(nav)
      .getAllByRole('link')
      .map((link) => link.textContent);
    expect(labels).toEqual(['Home', 'Analyze X-Ray', 'Recent Analyses', 'About', 'Profile']);
    expect(within(siteHeader()).getByRole('button', { name: /logout/i })).toBeInTheDocument();
  });

  it('keeps the public home page free of history content', async () => {
    renderApp('/');
    await screen.findByRole('heading', { level: 1 });
    expect(screen.queryByText(/view previous reports/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: /recent analyses/i })).not.toBeInTheDocument();
  });

  it('labels the home-page product preview as sample data', async () => {
    renderApp('/');
    expect(await screen.findByText(/illustration using sample data/i)).toBeInTheDocument();
  });

  it('navigates between pages from the navbar', async () => {
    const user = userEvent.setup();
    renderApp('/');
    const nav = await screen.findByRole('navigation', { name: 'Primary' });

    await user.click(within(nav).getByRole('link', { name: 'About' }));
    expect(await screen.findByRole('heading', { level: 1, name: /trust-aware explainable ai/i })).toBeInTheDocument();

    await user.click(within(nav).getByRole('link', { name: 'Analyze X-Ray' }));
    expect(await screen.findByRole('heading', { level: 1, name: /analyze a chest x-ray/i })).toBeInTheDocument();
  });

  it('lists the 14 diseases in canonical order on the about page', async () => {
    renderApp('/about');
    const section = (await screen.findByRole('heading', { name: /the 14 disease categories/i })).closest('section');
    expect(section).not.toBeNull();
    const items = within(section as HTMLElement).getAllByRole('listitem');
    expect(items).toHaveLength(14);
    expect(items[0]).toHaveTextContent('Atelectasis');
    expect(items[8]).toHaveTextContent('Consolidation');
    expect(items[12]).toHaveTextContent('Pleural Thickening');
    expect(items[13]).toHaveTextContent('Hernia');
  });

  it('renders a 404 page for unknown routes with navigation back', async () => {
    renderApp('/definitely-not-a-page');
    expect(await screen.findByRole('heading', { name: /this page could not be found/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /go to the home page/i })).toHaveAttribute('href', '/');
  });

  it('renders the unauthorized and service-unavailable states', async () => {
    const first = renderApp('/unauthorized');
    expect(await screen.findByRole('heading', { name: /do not have access/i })).toBeInTheDocument();
    first.unmount();

    renderApp('/service-unavailable');
    expect(await screen.findByRole('heading', { name: /service is unavailable/i })).toBeInTheDocument();
  });
});
