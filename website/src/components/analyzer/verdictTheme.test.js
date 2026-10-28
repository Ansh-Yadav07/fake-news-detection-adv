import { describe, it, expect } from 'vitest';
import { resolveVerdict, VERDICT_TONES, clampPct, isUnknownModel } from './verdictTheme.js';

describe('resolveVerdict', () => {
  it.each(Object.keys(VERDICT_TONES))('resolves %s', (label) => {
    const v = resolveVerdict(label);
    expect(v.key).toBe(label);
    expect(v.display).toBe(VERDICT_TONES[label].display);
    expect(v.Icon).toBeTruthy();
  });

  it('is case/whitespace insensitive', () => {
    expect(resolveVerdict('  likely fake ').key).toBe('LIKELY FAKE');
  });

  it('renders unknown labels neutrally', () => {
    expect(resolveVerdict('SOMETHING_NEW').display).toBe('Something new');
    expect(resolveVerdict(undefined).display).toBe('Unknown');
    expect(resolveVerdict(42).display).toBe('Unknown');
  });
});

describe('helpers', () => {
  it('clampPct', () => {
    expect(clampPct(150)).toBe(100);
    expect(clampPct(-5)).toBe(0);
    expect(clampPct('x')).toBe(0);
  });
  it('isUnknownModel', () => {
    expect(isUnknownModel('unknown')).toBe(true);
    expect(isUnknownModel(null)).toBe(true);
    expect(isUnknownModel('REAL')).toBe(false);
  });
});
