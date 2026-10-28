import { describe, it, expect } from 'vitest';
import {
  isWikiAvailable,
  isNewsAvailable,
  hasVerificationEvidence,
  getEnhancedDecisionWithVerification,
  calculateRobustness,
} from './decisionLogic.js';

const M = (tl, tc, hl, hc) => ({ t: [tl, tc], h: [hl, hc] });
const clean = { clickbait: 0.1, uppercase: 0.04, punctuation: 0.02 };
const NF = { status: 'NOT FOUND', verification_score: 0 };
const noNews = { verification_score: 0, supporting_articles: 0, trusted_source_count: 0, articles: [] };
const lowNews = {
  verification_score: 2.1,
  supporting_articles: 0,
  trusted_source_count: 0,
  articles: [{ similarity: 4 }, { similarity: 3 }],
};
const article =
  'The government announced a new policy today for farmers in the northern region regarding subsidies and the minister said that the scheme will be rolled out over the next two years across all districts';

// Mirrors how DemoSection invokes the decision function (always the verification path).
function decide(models, features, inputType, text, wikipedia, verification) {
  const wc = text.trim().split(/\s+/).length;
  const cb = Math.round((features.clickbait || 0) * 100);
  const up = Math.round((features.uppercase || 0) * 100);
  const pc = Math.round((features.punctuation || 0) * 100);
  return getEnhancedDecisionWithVerification(
    models.t[0], models.t[1], models.h[0], models.h[1],
    cb, pc, up, wc, inputType, wikipedia, verification
  );
}

const NOT_REAL = ['REAL', 'VERIFIED', 'VERIFIED FACT', 'LIKELY REAL'];
const NOT_FAKE = ['FAKE', 'LIKELY FAKE'];

describe('evidence helpers', () => {
  it('isWikiAvailable truth table', () => {
    expect(isWikiAvailable(null)).toBe(false);
    expect(isWikiAvailable(undefined)).toBe(false);
    expect(isWikiAvailable({ status: 'VERIFIED FACT' })).toBe(true);
    expect(isWikiAvailable({ status: 'PARTIALLY VERIFIED' })).toBe(true);
    for (const status of ['WEAK MATCH', 'NOT VERIFIED', 'NOT FOUND', 'CONTRADICTED']) {
      expect(isWikiAvailable({ status })).toBe(false);
    }
  });

  it('isNewsAvailable truth table', () => {
    expect(isNewsAvailable(null)).toBe(false);
    expect(isNewsAvailable({ error: 'x', supporting_articles: 3 })).toBe(false);
    expect(isNewsAvailable({ supporting_articles: 0 })).toBe(false);
    expect(isNewsAvailable({ supporting_articles: NaN })).toBe(false);
    expect(isNewsAvailable({ supporting_articles: 1 })).toBe(true);
    expect(isNewsAvailable({ supporting_articles: '2' })).toBe(true);
  });

  it('hasVerificationEvidence truth table', () => {
    expect(hasVerificationEvidence(null, null)).toBe(false);
    expect(hasVerificationEvidence(NF, noNews)).toBe(false);
    expect(hasVerificationEvidence(NF, lowNews)).toBe(false);
    expect(hasVerificationEvidence({ status: 'VERIFIED FACT' }, null)).toBe(true);
    expect(hasVerificationEvidence(NF, { supporting_articles: 2 })).toBe(true);
    expect(hasVerificationEvidence({ status: 'CONTRADICTED', is_contradicted: true }, null)).toBe(true);
  });
});

describe('scenarios (verification path, as shown in the UI)', () => {
  const cases = [
    ['1 fact: ML REAL .99, wiki NF, irrelevant news', M('REAL', .99, 'REAL', .99), clean, 'fact_claim', 'Water boils at 100 degrees', NF, lowNews, 'noLikelyFake'],
    ['2 fact: ML REAL .99, wiki NF, no news', M('REAL', .99, 'REAL', .99), clean, 'fact_claim', 'Water boils at 100 degrees', NF, noNews, 'noLikelyFake'],
    ['3 news: ML REAL .95, wiki NF, irrelevant news', M('REAL', .95, 'REAL', .9), clean, 'news_article', article, NF, lowNews, 'noLikelyFake'],
    ['10 fact: wiki NOT VERIFIED 10, ML REAL .99', M('REAL', .99, 'REAL', .99), clean, 'fact_claim', 'Mount Everest is the tallest mountain', { status: 'NOT VERIFIED', verification_score: 10 }, noNews, 'noLikelyFake'],
    ['7 UNKNOWN models, no verification', M('UNKNOWN', .5, 'UNKNOWN', .5), clean, 'mixed', 'Some text here', null, null, 'noReal'],
    ['8 UNKNOWN models, wiki NOT VERIFIED', M('UNKNOWN', .5, 'UNKNOWN', .5), clean, 'fact_claim', 'The moon is made of cheese', { status: 'NOT VERIFIED', verification_score: 12 }, noNews, 'noReal'],
    ['9 null evidence, ML FAKE .9, 1 word', M('FAKE', .9, 'FAKE', .85), {}, 'mixed', 'x', null, null, 'noReal'],
    ['11 both FAKE .62, 22 words', M('FAKE', .62, 'FAKE', .62), clean, 'mixed', 'Local council approves new park budget after long debate among residents and officials in the town hall meeting held last week here', null, null, 'noReal'],
  ];

  it.each(cases)('%s', (_n, models, feats, type, text, wiki, ver, rule) => {
    const d = decide(models, feats, type, text, wiki, ver);
    if (rule === 'noLikelyFake') expect(NOT_FAKE).not.toContain(d.final_label);
    if (rule === 'noReal') expect(NOT_REAL).not.toContain(d.final_label);
    expect(Number.isFinite(d.confidence)).toBe(true);
    expect(d.confidence).toBeGreaterThanOrEqual(0);
    expect(d.confidence).toBeLessThanOrEqual(0.99);
  });

  it('4 contradicted wiki vs 4 trusted sources @90 -> SUSPICIOUS', () => {
    const d = decide(M('REAL', .97, 'REAL', .95), clean, 'mixed', 'Kyiv is the capital of Ukraine says minister',
      { status: 'CONTRADICTED', verification_score: 5, is_contradicted: true, message: 'Wikipedia CONTRADICTS this claim.' },
      { verification_score: 90, supporting_articles: 8, trusted_source_count: 4, articles: [{}] });
    expect(d.final_label).toBe('SUSPICIOUS');
  });

  it('5 ML FAKE + clickbait with a single trusted source is never VERIFIED', () => {
    const d = decide(M('FAKE', .99, 'FAKE', .98), { clickbait: .9, uppercase: .45, punctuation: .12 }, 'news_article',
      'SHOCKING!!! Obama ARRESTED for treason, Reuters reports', NF,
      { verification_score: 36, supporting_articles: 1, trusted_source_count: 1, articles: [{}] });
    expect(d.final_label).not.toBe('VERIFIED');
  });

  it('6 wiki 72 on a news_article does not override strong ML FAKE', () => {
    const d = decide(M('FAKE', .99, 'FAKE', .97), { clickbait: .6, uppercase: .2, punctuation: .06 }, 'news_article',
      'Eiffel Tower in Paris France demolished yesterday',
      { status: 'VERIFIED FACT', verification_score: 72, message: 'Wikipedia confirms' }, noNews);
    expect(['VERIFIED FACT', 'VERIFIED']).not.toContain(d.final_label);
  });

  it('12 verification.error with score > 0 does not throw and is finite', () => {
    const d = decide(M('REAL', .9, 'REAL', .9), clean, 'mixed', 'abc def', NF,
      { error: 'x', verification_score: 40, supporting_articles: 3, trusted_source_count: 2 });
    expect(Number.isFinite(d.confidence)).toBe(true);
  });

  it('13 NaN / undefined confidences give finite confidence', () => {
    const d = decide(M('REAL', NaN, 'FAKE', undefined), { clickbait: NaN }, 'fact_claim', 'Paris is in France', NF, null);
    expect(Number.isFinite(d.confidence)).toBe(true);
    expect(d.confidence).toBeGreaterThanOrEqual(0);
    expect(d.confidence).toBeLessThanOrEqual(0.99);
  });

  it('14 NaN confidences with news evidence give finite confidence', () => {
    const d = decide(M('REAL', NaN, 'REAL', NaN), clean, 'news_article', article, null,
      { verification_score: 40, supporting_articles: 2, trusted_source_count: 0 });
    expect(Number.isFinite(d.confidence)).toBe(true);
  });

  it('15 wiki VERIFIED FACT 85 on a fact claim -> VERIFIED FACT', () => {
    const d = decide(M('REAL', .9, 'REAL', .85), clean, 'fact_claim', 'Paris is the capital of France',
      { status: 'VERIFIED FACT', verification_score: 85, message: 'Wikipedia confirms this claim.' }, noNews);
    expect(d.final_label).toBe('VERIFIED FACT');
  });

  it('16 three trusted sources @75 on news -> VERIFIED', () => {
    const d = decide(M('REAL', .9, 'REAL', .88), clean, 'news_article', article, NF,
      { verification_score: 75, supporting_articles: 6, trusted_source_count: 3 });
    expect(d.final_label).toBe('VERIFIED');
  });

  it('17 strong ML FAKE + clickbait, no evidence -> LIKELY FAKE', () => {
    const d = decide(M('FAKE', .95, 'FAKE', .95), { clickbait: .8, uppercase: .3, punctuation: .1 }, 'news_article', article, NF, noNews);
    expect(d.final_label).toBe('LIKELY FAKE');
  });

  it('18 two trusted sources @80 but ML FAKE .9 -> LIKELY REAL', () => {
    const d = decide(M('FAKE', .9, 'FAKE', .9), clean, 'news_article', article, NF,
      { verification_score: 80, supporting_articles: 4, trusted_source_count: 2 });
    expect(d.final_label).toBe('LIKELY REAL');
  });

  it('19 Wikipedia contradiction with weak news -> LIKELY FAKE', () => {
    const d = decide(M('REAL', .9, 'REAL', .9), clean, 'fact_claim', 'Sydney is the capital of Australia',
      { status: 'CONTRADICTED', is_contradicted: true, verification_score: 0, message: 'Wikipedia: capital of X is Y.' }, noNews);
    expect(d.final_label).toBe('LIKELY FAKE');
  });
});

describe('invariants', () => {
  it('weight profiles (fact / news / mixed) keep confidence in [0, 0.99]', () => {
    const wikis = [null, NF, { status: 'VERIFIED FACT', verification_score: 100 }, { status: 'CONTRADICTED', is_contradicted: true }];
    const news = [null, noNews, { verification_score: 100, supporting_articles: 9, trusted_source_count: 5 }];
    const labels = ['REAL', 'FAKE', 'UNKNOWN'];
    for (const type of ['fact_claim', 'news_article', 'mixed', 'bogus']) {
      for (const w of wikis) for (const n of news) for (const l of labels) {
        for (const c of [0, 0.5, 1, NaN]) {
          const d = getEnhancedDecisionWithVerification(l, c, l, c, 100, 100, 100, 40, type, w, n);
          expect(Number.isFinite(d.confidence)).toBe(true);
          expect(d.confidence).toBeGreaterThanOrEqual(0);
          expect(d.confidence).toBeLessThanOrEqual(0.99);
        }
      }
    }
  });

  it('robustness normalises agreement 1.0 and 100 identically', () => {
    expect(calculateRobustness(0.9, 0.9, 1.0, 60)).toBe(calculateRobustness(0.9, 0.9, 100, 60));
    expect(calculateRobustness(0.5, 0.5, 0.3, 60)).toBe(calculateRobustness(0.5, 0.5, 30, 60));
  });

  it('no evidence is UNVERIFIED-or-weaker, never LIKELY FAKE, unless ML is strongly fake', () => {
    for (const type of ['fact_claim', 'news_article', 'mixed']) {
      const d = getEnhancedDecisionWithVerification('REAL', 1, 'REAL', 1, 0, 0, 0, 30, type, NF, lowNews);
      expect(NOT_FAKE).not.toContain(d.final_label);
      expect(['VERIFIED', 'VERIFIED FACT']).not.toContain(d.final_label);
    }
    const strong = getEnhancedDecisionWithVerification('FAKE', 0.95, 'FAKE', 0.95, 80, 10, 30, 40, 'news_article', NF, noNews);
    expect(strong.final_label).toBe('LIKELY FAKE');
  });
});
