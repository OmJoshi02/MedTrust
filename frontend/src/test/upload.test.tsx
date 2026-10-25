import { fireEvent, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { demoAnalysisApi } from '@/demo';
import { validateImage, validateImageFile } from '@/lib/validation';
import { makeImageFile, renderApp } from './utils';

const OPTIONS = { maxBytes: 1024 * 1024 };

describe('file validation rules', () => {
  it('accepts PNG, JPG, JPEG and WebP by name and type', () => {
    expect(validateImageFile(makeImageFile('a.png', 'image/png'), OPTIONS)).toBeNull();
    expect(validateImageFile(makeImageFile('a.jpg', 'image/jpeg', 'jpeg'), OPTIONS)).toBeNull();
    expect(validateImageFile(makeImageFile('a.jpeg', 'image/jpeg', 'jpeg'), OPTIONS)).toBeNull();
    expect(validateImageFile(makeImageFile('a.webp', 'image/webp', 'webp'), OPTIONS)).toBeNull();
  });

  it('rejects unsupported formats and mismatched extensions', () => {
    expect(validateImageFile(makeImageFile('a.gif', 'image/gif'), OPTIONS)).toMatch(/unsupported file format/i);
    expect(validateImageFile(makeImageFile('a.pdf', 'application/pdf'), OPTIONS)).toMatch(/unsupported/i);
    expect(validateImageFile(makeImageFile('a.png', 'application/pdf'), OPTIONS)).toMatch(/unsupported/i);
  });

  it('rejects empty and oversized files', () => {
    expect(validateImageFile(new File([], 'a.png', { type: 'image/png' }), OPTIONS)).toMatch(/empty/i);
    const big = makeImageFile('big.png', 'image/png', 'png', 2 * 1024 * 1024);
    expect(validateImageFile(big, OPTIONS)).toMatch(/maximum size/i);
  });

  it('rejects a file whose contents are not an image even if the name says so', async () => {
    const fake = makeImageFile('fake.png', 'image/png', 'text');
    expect(await validateImage(fake, OPTIONS)).toMatch(/does not look like a valid/i);
    expect(await validateImage(makeImageFile(), OPTIONS)).toBeNull();
  });
});

describe('analysis upload interface', () => {
  const input = async () => (await screen.findByTestId('file-input')) as HTMLInputElement;

  it('keeps Analyze Image disabled until a valid image is selected', async () => {
    renderApp('/analyze');
    const button = await screen.findByRole('button', { name: 'Analyze Image' });
    expect(button).toBeDisabled();
    expect(screen.getByText(/nothing is analysed until you do/i)).toBeInTheDocument();
  });

  it('shows a clear error for an unsupported format and does not enable analysis', async () => {
    renderApp('/analyze');
    fireEvent.change(await input(), { target: { files: [makeImageFile('scan.gif', 'image/gif')] } });

    expect(await screen.findByText(/unsupported file format/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Analyze Image' })).toBeDisabled();
  });

  it('shows a clear error for an oversized file', async () => {
    renderApp('/analyze');
    fireEvent.change(await input(), {
      target: { files: [makeImageFile('big.png', 'image/png', 'png', 2 * 1024 * 1024)] },
    });
    expect(await screen.findByText(/maximum size is/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Analyze Image' })).toBeDisabled();
  });

  it('previews a valid image and lets the user remove it', async () => {
    renderApp('/analyze');
    fireEvent.change(await input(), { target: { files: [makeImageFile('chest.png')] } });

    expect(await screen.findByRole('img', { name: /preview of the selected image: chest\.png/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Analyze Image' })).toBeEnabled();

    fireEvent.click(screen.getByRole('button', { name: /remove/i }));
    expect(await screen.findByTestId('file-input')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Analyze Image' })).toBeDisabled();
  });

  it('accepts a dropped image', async () => {
    renderApp('/analyze');
    const zone = await screen.findByRole('button', { name: /upload a chest x-ray image/i });
    fireEvent.drop(zone, { dataTransfer: { files: [makeImageFile('dropped.png')] } });
    expect(await screen.findByText('dropped.png')).toBeInTheDocument();
  });

  it('never calls the API when no file has been selected', async () => {
    const create = vi.spyOn(demoAnalysisApi, 'create');
    renderApp('/analyze');
    const button = await screen.findByRole('button', { name: 'Analyze Image' });
    fireEvent.click(button);
    expect(create).not.toHaveBeenCalled();
  });
});
