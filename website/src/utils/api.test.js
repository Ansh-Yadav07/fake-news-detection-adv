import { describe, it, expect, afterEach, vi } from 'vitest';
import { getApiBase, apiUrl } from './api.js';

afterEach(() => {
  vi.unstubAllEnvs();
});

const setEnv = (base, legacy) => {
  vi.stubEnv('VITE_API_BASE', base ?? '');
  vi.stubEnv('VITE_API_URL', legacy ?? '');
};

describe('getApiBase', () => {
  it('falls back to the default when nothing is set', () => {
    setEnv();
    expect(getApiBase()).toBe('http://127.0.0.1:5001');
  });

  it('uses VITE_API_BASE (trimmed)', () => {
    setEnv('  https://api.example.com  ');
    expect(getApiBase()).toBe('https://api.example.com');
  });

  it('VITE_API_BASE takes precedence over VITE_API_URL', () => {
    setEnv('https://base.example.com', 'https://legacy.example.com/predict');
    expect(getApiBase()).toBe('https://base.example.com');
  });

  it('strips /predict from VITE_API_URL', () => {
    setEnv('', 'https://legacy.example.com/predict');
    expect(getApiBase()).toBe('https://legacy.example.com/');
    setEnv('', 'https://legacy.example.com/predict/');
    expect(getApiBase()).toBe('https://legacy.example.com/');
    setEnv('', 'https://host.example.com/api/predict');
    expect(getApiBase()).toBe('https://host.example.com/api');
  });

  it('falls back to default when VITE_API_URL is not a valid URL', () => {
    setEnv('', 'not a url');
    expect(getApiBase()).toBe('http://127.0.0.1:5001');
  });
});

describe('apiUrl', () => {
  it('joins endpoints regardless of slashes', () => {
    expect(apiUrl('analyze', 'http://h:1')).toBe('http://h:1/analyze');
    expect(apiUrl('/analyze', 'http://h:1/')).toBe('http://h:1/analyze');
    expect(apiUrl('///analyze', 'http://h:1//'.replace(/\/+$/, '/'))).toBe('http://h:1/analyze');
  });

  it('preserves a base path', () => {
    expect(apiUrl('analyze', 'https://host/api')).toBe('https://host/api/analyze');
    expect(apiUrl('/analyze', 'https://host/api/')).toBe('https://host/api/analyze');
  });

  it('uses getApiBase() by default', () => {
    setEnv('https://env.example.com');
    expect(apiUrl('predict')).toBe('https://env.example.com/predict');
  });
});
