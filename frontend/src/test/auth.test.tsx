import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { demoAuthApi } from '@/demo';
import { ApiError } from '@/services/errors';
import { renderApp } from './utils';

describe('login', () => {
  it('validates the email and password before submitting', async () => {
    const user = userEvent.setup();
    const login = vi.spyOn(demoAuthApi, 'login');
    renderApp('/login');

    await user.click(await screen.findByRole('button', { name: 'Log in' }));
    expect(await screen.findByText('Enter your email address.')).toBeInTheDocument();
    expect(screen.getByText('Enter your password.')).toBeInTheDocument();

    await user.type(screen.getByLabelText(/email address/i), 'not-an-email');
    await user.click(screen.getByRole('button', { name: 'Log in' }));
    expect(await screen.findByText('Enter a valid email address.')).toBeInTheDocument();
    expect(login).not.toHaveBeenCalled();
  });

  it('toggles password visibility', async () => {
    const user = userEvent.setup();
    renderApp('/login');
    const input = await screen.findByLabelText(/^password/i);
    expect(input).toHaveAttribute('type', 'password');
    await user.click(screen.getByRole('button', { name: 'Show password' }));
    expect(input).toHaveAttribute('type', 'text');
    expect(screen.getByRole('button', { name: 'Hide password' })).toBeInTheDocument();
  });

  it('identifies demo authentication on the form', async () => {
    renderApp('/login');
    expect(await screen.findByText(/not real authentication/i)).toBeInTheDocument();
  });

  it('explains that password recovery is unavailable', async () => {
    const user = userEvent.setup();
    renderApp('/login');
    await user.click(await screen.findByRole('button', { name: /forgot password/i }));
    expect(screen.getByText(/password recovery is not available/i)).toBeInTheDocument();
  });

  it('shows a clear message for invalid credentials', async () => {
    const user = userEvent.setup();
    vi.spyOn(demoAuthApi, 'login').mockRejectedValue(
      new ApiError({ kind: 'unauthorized', status: 401, message: 'Invalid credentials' }),
    );
    renderApp('/login');

    await user.type(await screen.findByLabelText(/email address/i), 'a@b.co');
    await user.type(screen.getByLabelText(/^password/i), 'wrong-password');
    await user.click(screen.getByRole('button', { name: 'Log in' }));

    expect(await screen.findByText(/email or password is incorrect/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Log in' })).toBeEnabled();
  });

  it('shows a distinct message when the API cannot be reached', async () => {
    const user = userEvent.setup();
    vi.spyOn(demoAuthApi, 'login').mockRejectedValue(
      new ApiError({ kind: 'network', message: 'The MedTrust service could not be reached.' }),
    );
    renderApp('/login');

    await user.type(await screen.findByLabelText(/email address/i), 'a@b.co');
    await user.type(screen.getByLabelText(/^password/i), 'whatever');
    await user.click(screen.getByRole('button', { name: 'Log in' }));

    expect(await screen.findByText(/could not be reached/i)).toBeInTheDocument();
    expect(screen.queryByText(/email or password is incorrect/i)).not.toBeInTheDocument();
  });

  it('disables the submit button and redirects to /analyze after success', async () => {
    const user = userEvent.setup();
    const { router } = renderApp('/login');

    await user.type(await screen.findByLabelText(/email address/i), 'tester@example.com');
    await user.type(screen.getByLabelText(/^password/i), 'a-password');
    await user.click(screen.getByRole('button', { name: 'Log in' }));

    expect(await screen.findByRole('button', { name: /signing in/i })).toBeDisabled();
    await waitFor(() => expect(router.state.location.pathname).toBe('/analyze'));
    expect(await screen.findByRole('heading', { level: 1, name: /analyze a chest x-ray/i })).toBeInTheDocument();
  });
});

describe('signup', () => {
  it('validates every field and the terms acknowledgement', async () => {
    const user = userEvent.setup();
    const register = vi.spyOn(demoAuthApi, 'register');
    renderApp('/signup');

    await user.click(await screen.findByRole('button', { name: 'Create account' }));
    expect(await screen.findByText('Enter your full name.')).toBeInTheDocument();
    expect(screen.getByText('Enter your email address.')).toBeInTheDocument();
    expect(screen.getByText('Create a password.')).toBeInTheDocument();
    expect(screen.getByText('Confirm your password.')).toBeInTheDocument();
    expect(screen.getByText(/acknowledge the terms/i)).toBeInTheDocument();
    expect(register).not.toHaveBeenCalled();
  });

  it('reflects password requirements as they are met', async () => {
    const user = userEvent.setup();
    renderApp('/signup');
    const rules = await screen.findByRole('list', { name: 'Password requirements' });
    expect(rules).toHaveTextContent('At least 10 characters (not met)');

    await user.type(screen.getAllByLabelText(/^password/i)[0] as HTMLElement, 'Abcdefghij1!');
    expect(rules).toHaveTextContent('At least 10 characters (met)');
    expect(rules).toHaveTextContent('At least one symbol (met)');
  });

  it('rejects mismatched passwords', async () => {
    const user = userEvent.setup();
    renderApp('/signup');

    await user.type(await screen.findByLabelText(/full name/i), 'Ada Lovelace');
    await user.type(screen.getByLabelText(/email address/i), 'ada@example.com');
    await user.type(screen.getAllByLabelText(/^password/i)[0] as HTMLElement, 'Abcdefghij1!');
    await user.type(screen.getByLabelText(/confirm password/i), 'Different1!pass');
    await user.click(screen.getByRole('checkbox'));
    await user.click(screen.getByRole('button', { name: 'Create account' }));

    expect(await screen.findByText('The passwords do not match.')).toBeInTheDocument();
  });

  it('does not claim success when the API call fails', async () => {
    const user = userEvent.setup();
    vi.spyOn(demoAuthApi, 'register').mockRejectedValue(
      new ApiError({ kind: 'network', message: 'The MedTrust service could not be reached.' }),
    );
    renderApp('/signup');

    await user.type(await screen.findByLabelText(/full name/i), 'Ada Lovelace');
    await user.type(screen.getByLabelText(/email address/i), 'ada@example.com');
    await user.type(screen.getAllByLabelText(/^password/i)[0] as HTMLElement, 'Abcdefghij1!');
    await user.type(screen.getByLabelText(/confirm password/i), 'Abcdefghij1!');
    await user.click(screen.getByRole('checkbox'));
    await user.click(screen.getByRole('button', { name: 'Create account' }));

    expect(await screen.findByText(/account could not be created/i)).toBeInTheDocument();
    expect(screen.queryByText(/registration succeeded/i)).not.toBeInTheDocument();
  });

  it('shows the success state only after the API confirms', async () => {
    const user = userEvent.setup();
    renderApp('/signup');

    await user.type(await screen.findByLabelText(/full name/i), 'Ada Lovelace');
    await user.type(screen.getByLabelText(/email address/i), 'ada@example.com');
    await user.type(screen.getAllByLabelText(/^password/i)[0] as HTMLElement, 'Abcdefghij1!');
    await user.type(screen.getByLabelText(/confirm password/i), 'Abcdefghij1!');
    await user.click(screen.getByRole('checkbox'));
    await user.click(screen.getByRole('button', { name: 'Create account' }));

    expect(await screen.findByText(/registration succeeded/i)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /continue to log in/i })).toHaveAttribute('href', '/login');
  });
});
