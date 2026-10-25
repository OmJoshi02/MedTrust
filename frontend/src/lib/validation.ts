import { formatBytes } from './format';

/* ------------------------------------------------------------------ */
/* Account fields                                                      */
/* ------------------------------------------------------------------ */

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

export function validateEmail(value: string): string | null {
  const email = value.trim();
  if (email === '') return 'Enter your email address.';
  if (email.length > 254 || !EMAIL_PATTERN.test(email)) return 'Enter a valid email address.';
  return null;
}

export function validateFullName(value: string): string | null {
  const name = value.trim();
  if (name === '') return 'Enter your full name.';
  if (name.length < 2) return 'Your name must be at least 2 characters.';
  if (name.length > 100) return 'Your name must be 100 characters or fewer.';
  return null;
}

export interface PasswordRule {
  id: string;
  label: string;
  met: boolean;
}

export const PASSWORD_MIN_LENGTH = 10;

export function passwordRules(password: string): PasswordRule[] {
  return [
    {
      id: 'length',
      label: `At least ${PASSWORD_MIN_LENGTH} characters`,
      met: password.length >= PASSWORD_MIN_LENGTH,
    },
    { id: 'case', label: 'Upper and lower case letters', met: /[a-z]/.test(password) && /[A-Z]/.test(password) },
    { id: 'number', label: 'At least one number', met: /\d/.test(password) },
    { id: 'symbol', label: 'At least one symbol', met: /[^A-Za-z0-9\s]/.test(password) },
  ];
}

export function validateNewPassword(password: string): string | null {
  if (password === '') return 'Create a password.';
  const unmet = passwordRules(password).filter((rule) => !rule.met);
  if (unmet.length > 0) return 'Your password does not meet all of the requirements.';
  return null;
}

export function validateConfirmPassword(password: string, confirmation: string): string | null {
  if (confirmation === '') return 'Confirm your password.';
  if (password !== confirmation) return 'The passwords do not match.';
  return null;
}

/* ------------------------------------------------------------------ */
/* Image uploads                                                       */
/* ------------------------------------------------------------------ */

export const ACCEPTED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/webp'] as const;
export const ACCEPTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp'] as const;
export const ACCEPT_ATTRIBUTE = [...ACCEPTED_IMAGE_TYPES, ...ACCEPTED_IMAGE_EXTENSIONS].join(',');

export const UNSUPPORTED_FORMAT_MESSAGE = 'Unsupported file format. Use a PNG, JPG, JPEG or WebP image.';

export interface ImageValidationOptions {
  maxBytes: number;
}

function extensionOf(filename: string): string {
  const dot = filename.lastIndexOf('.');
  return dot === -1 ? '' : filename.slice(dot).toLowerCase();
}

/** Synchronous checks: emptiness, declared type/extension and size. */
export function validateImageFile(file: File, options: ImageValidationOptions): string | null {
  if (file.size === 0) return 'The selected file is empty.';

  const extension = extensionOf(file.name);
  const typeOk = (ACCEPTED_IMAGE_TYPES as readonly string[]).includes(file.type);
  const extensionOk = (ACCEPTED_IMAGE_EXTENSIONS as readonly string[]).includes(extension);
  // A known type with a wrong extension, or an unknown type with a wrong extension, is rejected.
  if (!extensionOk || (file.type !== '' && !typeOk)) return UNSUPPORTED_FORMAT_MESSAGE;

  if (file.size > options.maxBytes) {
    return `This file is ${formatBytes(file.size)}. The maximum size is ${formatBytes(options.maxBytes)}.`;
  }
  return null;
}

export type ImageSignature = 'png' | 'jpeg' | 'webp';

function readHeader(file: File, length = 12): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(new Uint8Array(reader.result as ArrayBuffer));
    reader.onerror = () => reject(reader.error ?? new Error('Unable to read the file.'));
    reader.readAsArrayBuffer(file.slice(0, length));
  });
}

/** Identifies PNG, JPEG and WebP by their file signatures rather than by name. */
export async function sniffImageSignature(file: File): Promise<ImageSignature | null> {
  const bytes = await readHeader(file);
  if (
    bytes.length >= 8 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47
  ) {
    return 'png';
  }
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return 'jpeg';
  }
  if (
    bytes.length >= 12 &&
    bytes[0] === 0x52 && // R
    bytes[1] === 0x49 && // I
    bytes[2] === 0x46 && // F
    bytes[3] === 0x46 && // F
    bytes[8] === 0x57 && // W
    bytes[9] === 0x45 && // E
    bytes[10] === 0x42 && // B
    bytes[11] === 0x50 // P
  ) {
    return 'webp';
  }
  return null;
}

/** Full validation: synchronous checks plus a content-signature check. */
export async function validateImage(
  file: File,
  options: ImageValidationOptions,
): Promise<string | null> {
  const basic = validateImageFile(file, options);
  if (basic) return basic;
  try {
    const signature = await sniffImageSignature(file);
    if (!signature) {
      return 'This file does not look like a valid PNG, JPEG or WebP image.';
    }
  } catch {
    return 'The selected file could not be read.';
  }
  return null;
}
