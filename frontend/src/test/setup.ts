import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, beforeEach, vi } from 'vitest';

// Tests must not depend on the developer's local .env file.
vi.stubEnv('VITE_DEMO_MODE', 'true');
vi.stubEnv('VITE_ENABLE_REMEMBER_ME', 'false');
vi.stubEnv('VITE_MAX_UPLOAD_MB', '1');

// jsdom does not implement these browser APIs.
window.scrollTo = vi.fn();

const NodeRequest = globalThis.Request;
globalThis.Request = class Request extends NodeRequest {
  constructor(input: RequestInfo | URL, init?: RequestInit) {
    if (init && init.signal) {
      const { signal: _signal, ...rest } = init;
      super(input, rest);
      return;
    }
    super(input, init);
  }
};

class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;

let counter = 0;
URL.createObjectURL = vi.fn(() => `blob:test/${(counter += 1)}`);
URL.revokeObjectURL = vi.fn();

beforeEach(() => {
  window.sessionStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
