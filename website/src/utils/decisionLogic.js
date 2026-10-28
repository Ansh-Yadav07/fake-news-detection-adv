/**
 * Decision Logic for Fake News Detection — v3 (Parallel Verification, evidence-aware)
 *
 * All score inputs use a 0-100 scale for clickbait / punctuation / uppercase
 * (DemoSection multiplies the backend's 0-1 raw_features by 100), and a 0-1
 * scale for model confidences.
 *
 * === EVIDENCE AVAILABILITY (shared with the UI) ===
 *   isWikiAvailable(wikipedia)      → status is "VERIFIED FACT" or "PARTIALLY VERIFIED"
 *                                     ("WEAK MATCH" / "NOT VERIFIED" / "NOT FOUND" are NOT evidence)
 *   isWikiContradicted(wikipedia)   → wikipedia.is_contradicted === true
 *   isNewsAvailable(verification)   → non-null, no `error`, supporting_articles > 0
 *   hasVerificationEvidence(w, v)   → any of the above
 *
 * === INPUT-AWARE BASE WEIGHTS (verification path) ===
 *
 * Fact Claims:            Wikipedia 60% · ML 20% · Linguistic 10% · Clickbait 10%
 * News Articles:          GNews 50% · ML 20% · Source Credibility 15% · Linguistic 10% · Clickbait 5%
 * Mixed:                  Wikipedia 30% · GNews 25% · ML 20% · Linguistic 15% · Clickbait 10%
 *
 * If the profile's primary source is missing but the other source is present
 * (e.g. a fact claim with news evidence but no Wikipedia match), the MIXED
 * profile is used instead. Signals that are unavailable (a missing source, or
 * ML when both models returned UNKNOWN / non-finite confidence) are dropped and
 * the remaining weights are RENORMALISED to sum to 1, so absence of evidence
 * never drags the score toward FAKE.
 *
 * "Linguistic" in the verification path is style-only (uppercase + punctuation);
 * clickbait has its own weight there, so it is not double-counted.
 *
 * === OVERRIDE RULES (verification path, in order) ===
 *   0. Wikipedia contradicts → LIKELY FAKE, unless news strongly supports
 *      (score > 50% and ≥ 2 trusted sources) → SUSPICIOUS (conflicting evidence).
 *   1. Wikipedia score > 70% → VERIFIED FACT, only when input is not a
 *      news_article and ML fake score < 0.8; otherwise falls through.
 *   2. News score > 50% and ≥ 2 trusted sources → VERIFIED.
 *   2b. News score ≥ 50% and ≥ 1 trusted source → VERIFIED.
 *       (2/2b downgrade to LIKELY REAL when ML fake score > 0.8 or clickbait > 60.)
 *   2c. News score > 25% and ≥ 3 supporting articles → LIKELY REAL
 *       (skipped when ML fake score > 0.8 or clickbait > 60).
 *   3. No evidence → UNVERIFIED (reason notes the ML lean). LIKELY FAKE only
 *      when ML is strongly fake (mlRealScore < 0.2) and the combined score < 0.5.
 *   4. Weighted decision on the renormalised score. Positive verdicts are
 *      downgraded one step when ML fake score > 0.8 (conflicting signals).
 *
 * Verdict categories: VERIFIED, VERIFIED FACT, LIKELY REAL, UNVERIFIED, SUSPICIOUS, LIKELY FAKE
 *
 * === WITHOUT VERIFICATION (legacy fallback: getEnhancedDecision) ===
 * Text-length-based ML/linguistic blend; ML weight is ≥ 0.5 at all lengths.
 * Clean writing style alone can never produce REAL: REAL requires the ML blend
 * itself to lean real. Labels: REAL, FAKE, UNCERTAIN, SUSPICIOUS.
 */

const TRANSFORMER_WEIGHT = 0.60;
const LR_WEIGHT = 0.40;

// Base weight profiles by input type (renormalised over available signals at runtime)
const FACT_WEIGHTS = {
  WIKIPEDIA: 0.60,
  ML_MODELS: 0.20,
  LINGUISTIC: 0.10,
  CLICKBAIT: 0.10
};

const NEWS_WEIGHTS = {
  GNEWS: 0.50,
  ML_MODELS: 0.20,
  SOURCE_CREDIBILITY: 0.15,
  LINGUISTIC: 0.10,
  CLICKBAIT: 0.05
};

const MIXED_WEIGHTS = {
  WIKIPEDIA: 0.30,
  GNEWS: 0.25,
  ML_MODELS: 0.20,
  LINGUISTIC: 0.15,
  CLICKBAIT: 0.10
};

// Thresholds for "strong fake signals" that block evidence-based overrides
const STRONG_ML_FAKE = 0.80;
const STRONG_CLICKBAIT = 60;

// ---------------------------------------------------------------------------
// Input sanitising
// ---------------------------------------------------------------------------

/** Coerce to a finite number, or return the fallback. */
function num(x, fallback = 0) {
  const n = typeof x === 'number' ? x : Number(x);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

/** A model output is usable only with a REAL/FAKE label and a finite confidence. */
function isModelValid(label, conf) {
  return (label === "REAL" || label === "FAKE") && Number.isFinite(conf);
}

/**
 * Blend the two models into a REAL score (0-1), dropping any unusable model.
 * Returns { available, realScore, fakeScore, tValid, hValid }.
 */
function blendModels(t_label, t_conf, h_label, h_conf) {
  const tValid = isModelValid(t_label, t_conf);
  const hValid = isModelValid(h_label, h_conf);
  let wSum = 0;
  let acc = 0;
  if (tValid) {
    const c = clamp(t_conf, 0, 1);
    acc += TRANSFORMER_WEIGHT * (t_label === "REAL" ? c : 1 - c);
    wSum += TRANSFORMER_WEIGHT;
  }
  if (hValid) {
    const c = clamp(h_conf, 0, 1);
    acc += LR_WEIGHT * (h_label === "REAL" ? c : 1 - c);
    wSum += LR_WEIGHT;
  }
  if (wSum === 0) {
    return { available: false, realScore: null, fakeScore: null, tValid, hValid };
  }
  const realScore = acc / wSum;
  return { available: true, realScore, fakeScore: 1 - realScore, tValid, hValid };
}

/** Clamp a final confidence into [0.1, 0.99]; non-finite → 0.5. */
function finalConfidence(c) {
  const v = Number.isFinite(c) ? c : 0.5;
  return parseFloat(clamp(v, 0.1, 0.99).toFixed(3));
}

// ---------------------------------------------------------------------------
// Evidence availability helpers (exported — the UI routes with these)
// ---------------------------------------------------------------------------

/**
 * True only when Wikipedia returned a genuinely relevant match
 * ("VERIFIED FACT" or "PARTIALLY VERIFIED"). WEAK MATCH / NOT VERIFIED /
 * NOT FOUND / CONTRADICTED are not "available" (use isWikiContradicted for that).
 */
export function isWikiAvailable(wikipedia) {
  if (!wikipedia) return false;
  return wikipedia.status === "VERIFIED FACT" || wikipedia.status === "PARTIALLY VERIFIED";
}

/** True when Wikipedia explicitly contradicts the claim. */
export function isWikiContradicted(wikipedia) {
  return !!wikipedia && wikipedia.is_contradicted === true;
}

/** True only when news verification succeeded and found ≥ 1 supporting article. */
export function isNewsAvailable(verification) {
  if (!verification || verification.error) return false;
  return num(verification.supporting_articles) > 0;
}

/** True when there is any real verification evidence (support or contradiction). */
export function hasVerificationEvidence(wikipedia, verification) {
  return isWikiAvailable(wikipedia) || isNewsAvailable(verification) || isWikiContradicted(wikipedia);
}

// ---------------------------------------------------------------------------
// Scores
// ---------------------------------------------------------------------------

/**
 * Calculate weighted agreement between the two models.
 * Returns 0-100. Returns 0 when either model is unusable (UNKNOWN / non-finite).
 */
export function calculateAgreement(t_conf, h_conf, t_label, h_label) {
  if (!isModelValid(t_label, t_conf) || !isModelValid(h_label, h_conf)) return 0;
  const t = clamp(t_conf, 0, 1);
  const h = clamp(h_conf, 0, 1);
  if (t_label === h_label) {
    return Math.round(((t + h) / 2) * 100);
  }
  const disagreement = (t + h) / 2;
  return Math.round(Math.max(0, (1 - disagreement) * 100));
}

/**
 * Compute a "linguistic fake score" from raw feature signals (inputs 0-100).
 * Returns 0-1 where higher = more likely fake based on writing style alone.
 *
 * includeClickbait = true  (legacy path): clickbait 50%, uppercase 30%, punctuation 20%.
 * includeClickbait = false (verification path, where clickbait has its own weight):
 *                    uppercase 60%, punctuation 40%.
 */
function computeLinguisticFakeScore(clickbait_score, punctuation_score, uppercase_ratio, includeClickbait = true) {
  const cb = clamp(num(clickbait_score), 0, 100);
  const pc = Math.max(0, num(punctuation_score));
  const up = Math.max(0, num(uppercase_ratio));

  const wCb = includeClickbait ? 0.50 : 0;
  const wUp = includeClickbait ? 0.30 : 0.60;
  const wPc = includeClickbait ? 0.20 : 0.40;

  let score = 0;
  score += (cb / 100) * wCb;
  // Excessive uppercase (> 20%) is very suspicious
  if (up > 20) score += Math.min((up - 20) / 80, 1.0) * wUp;
  // Heavy punctuation
  if (pc > 5) score += Math.min((pc - 5) / 30, 1.0) * wPc;

  return Math.min(score, 1.0);
}

// ---------------------------------------------------------------------------
// Insights
// ---------------------------------------------------------------------------

/**
 * Generate rich insights based on the analysis results (legacy — without verification).
 */
function generateInsights(t_label, t_conf, h_label, h_conf, clickbait_score, punctuation_score, uppercase_ratio, wordCount, final_label) {
  const insights = [];
  const tValid = isModelValid(t_label, t_conf);
  const hValid = isModelValid(h_label, h_conf);

  // Model agreement insight
  if (tValid && hValid) {
    if (t_label === h_label) {
      const avgConf = ((t_conf + h_conf) / 2 * 100).toFixed(0);
      insights.push(`Both AI models agree: this content is likely ${t_label} with ${avgConf}% average confidence.`);
    } else {
      insights.push(`Models disagree — Transformer predicts ${t_label} (${(t_conf * 100).toFixed(0)}%) while LR model predicts ${h_label} (${(h_conf * 100).toFixed(0)}%).`);
    }
  } else if (tValid || hValid) {
    const [name, lbl, c] = tValid ? ["Transformer", t_label, t_conf] : ["LR model", h_label, h_conf];
    insights.push(`Only the ${name} produced a usable prediction: ${lbl} (${(c * 100).toFixed(0)}%). The other model was unavailable and excluded.`);
  } else {
    insights.push(`Neither AI model produced a usable prediction — the verdict relies on other signals only.`);
  }

  // Clickbait analysis
  if (clickbait_score > 60) {
    insights.push(`High clickbait indicators detected (${clickbait_score}%) — sensationalist language, excessive punctuation, or ALL CAPS patterns found.`);
  } else if (clickbait_score > 30) {
    insights.push(`Moderate clickbait signals (${clickbait_score}%) — some sensational language patterns detected.`);
  } else {
    insights.push(`Low clickbait score (${clickbait_score}%) — writing style appears measured, though clean style alone does not prove accuracy.`);
  }

  // Text length analysis
  if (wordCount < 10) {
    insights.push(`Very short text (${wordCount} words) — ML models are less reliable on text this short.`);
  } else if (wordCount < 20) {
    insights.push(`Short text (${wordCount} words) — limited content for ML models. Linguistic features have increased influence.`);
  } else if (wordCount < 50) {
    insights.push(`Moderate text length (${wordCount} words) — adequate for analysis but longer content yields higher accuracy.`);
  } else {
    insights.push(`Sufficient text length (${wordCount} words) — provides strong signal for model analysis.`);
  }

  // Uppercase analysis
  if (uppercase_ratio > 30) {
    insights.push(`Unusually high uppercase ratio (${uppercase_ratio}%) — excessive capitalization is common in misleading content and clickbait.`);
  } else if (uppercase_ratio > 15) {
    insights.push(`Elevated uppercase usage (${uppercase_ratio}%) — more capitalization than typical news articles.`);
  }

  // Punctuation analysis
  if (punctuation_score > 10) {
    insights.push(`Above-average punctuation density (${punctuation_score}%) — heavy punctuation can indicate sensationalized writing.`);
  }

  // Confidence insight
  const confs = [tValid ? t_conf : null, hValid ? h_conf : null].filter(c => c !== null);
  if (confs.length > 0) {
    const maxConf = Math.max(...confs);
    if (maxConf < 0.6) {
      insights.push(`Model confidence is low — this content may contain mixed signals or be genuinely ambiguous.`);
    } else if (maxConf > 0.9 && final_label !== "UNCERTAIN" && final_label !== "UNVERIFIED") {
      insights.push(`High model confidence (${(maxConf * 100).toFixed(0)}%) — strong signal detected in the content patterns.`);
    }
  }

  return insights;
}

/**
 * Generate Wikipedia-specific insights.
 */
function generateWikipediaInsights(wikipedia) {
  const insights = [];
  if (!wikipedia) return insights;

  const status = wikipedia.status || "NOT FOUND";
  const score = num(wikipedia.verification_score);
  const title = wikipedia.wiki_title || "";

  if (status === "CONTRADICTED" || wikipedia.is_contradicted === true) {
    insights.push(`⚠️ Wikipedia CONTRADICTS this claim.`);
    insights.push(wikipedia.message || 'The claim does not match Wikipedia evidence.');
  } else if (status === "VERIFIED FACT") {
    insights.push(`Wikipedia confirms the statement.`);
    insights.push(`Fact matches trusted knowledge sources.`);
    insights.push(`Wikipedia verification score: ${score}%.`);
  } else if (status === "PARTIALLY VERIFIED") {
    insights.push(`Wikipedia contains related information in "${title}".`);
    insights.push(`Wikipedia verification score: ${score}%.`);
  } else if (status === "NOT FOUND") {
    insights.push(`No relevant Wikipedia article found for this claim.`);
  } else {
    insights.push(`Wikipedia article "${title}" found but claim could not be verified (${score}%) — not counted as evidence.`);
  }

  if (wikipedia.wiki_extract) {
    const snippet = wikipedia.wiki_extract.length > 120
      ? wikipedia.wiki_extract.substring(0, 120) + '...'
      : wikipedia.wiki_extract;
    insights.push(`Wikipedia excerpt: "${snippet}"`);
  }

  return insights;
}

/**
 * Generate GNews-specific insights.
 */
function generateGNewsInsights(verification) {
  const insights = [];
  if (!verification) return insights;
  if (Array.isArray(verification.insights)) {
    verification.insights.forEach(insight => insights.push(insight));
  }
  return insights;
}

/**
 * Generate enhanced insights that include ALL verification findings.
 */
function generateInsightsWithVerification(t_label, t_conf, h_label, h_conf, clickbait_score, punctuation_score, uppercase_ratio, wordCount, final_label, wikipedia, verification) {
  const insights = generateInsights(t_label, t_conf, h_label, h_conf, clickbait_score, punctuation_score, uppercase_ratio, wordCount, final_label);
  generateWikipediaInsights(wikipedia).forEach(i => insights.push(i));
  generateGNewsInsights(verification).forEach(i => insights.push(i));
  return insights;
}

// ---------------------------------------------------------------------------
// Legacy decision (no verification evidence)
// ---------------------------------------------------------------------------

/**
 * Core decision function — LEGACY path, used when no verification evidence exists.
 *
 * Blends ML fake score with a linguistic fake score using text-length weights
 * (ML weight ≥ 0.5 at every length). Absence of clickbait is not evidence of
 * truth, so REAL additionally requires the ML blend itself to lean real
 * (mlFakeScore ≤ 0.5). UNKNOWN / non-finite model outputs are dropped.
 *
 * Returns { final_label: "REAL"|"FAKE"|"UNCERTAIN"|"SUSPICIOUS", confidence, agreement, reason, insights }.
 */
export function getEnhancedDecision(t_label, t_conf, h_label, h_conf, clickbait_score, punctuation_score, uppercase_ratio, wordCount = 50) {
  clickbait_score = num(clickbait_score);
  punctuation_score = num(punctuation_score);
  uppercase_ratio = num(uppercase_ratio);
  wordCount = num(wordCount, 50);

  let final_label = "UNCERTAIN";
  let confidence = 0.5;
  let reason = "";

  const agreement = calculateAgreement(t_conf, h_conf, t_label, h_label);
  const ml = blendModels(t_label, t_conf, h_label, h_conf);

  // STEP 1: Linguistic fake score (clickbait included — no separate clickbait weight here)
  const linguisticFakeScore = computeLinguisticFakeScore(clickbait_score, punctuation_score, uppercase_ratio, true);

  // STEP 2: Trust weights by text length (ML never below 0.5)
  let modelTrust;
  if (wordCount < 10) modelTrust = 0.50;
  else if (wordCount < 20) modelTrust = 0.60;
  else if (wordCount < 50) modelTrust = 0.75;
  else modelTrust = 0.85;

  if (!ml.available) {
    // STEP 3a: No usable ML prediction — style alone can flag, never clear.
    if (linguisticFakeScore > 0.55) {
      final_label = "SUSPICIOUS";
      confidence = linguisticFakeScore;
      reason = `ML models unavailable. Strong sensationalist writing patterns detected (${clickbait_score}% clickbait score), but style alone cannot confirm the content is fake.`;
    } else {
      final_label = "UNCERTAIN";
      confidence = 0.5;
      reason = `ML models unavailable and no verification evidence — writing style alone cannot determine authenticity.`;
    }
  } else {
    // STEP 3b: Weighted final fake score
    const mlFakeScore = ml.fakeScore;
    const linguisticTrust = 1 - modelTrust;
    const finalFakeScore = (modelTrust * mlFakeScore) + (linguisticTrust * linguisticFakeScore);
    const bothValid = ml.tValid && ml.hValid;

    if (finalFakeScore > 0.55) {
      final_label = "FAKE";
      confidence = finalFakeScore;
      if (linguisticFakeScore > 0.4 && mlFakeScore < 0.5) {
        reason = `Strong clickbait/sensationalist patterns detected (${clickbait_score}% clickbait score) outweigh uncertain ML models.`;
      } else if (bothValid && t_label === "FAKE" && h_label === "FAKE") {
        reason = `Both models agree on FAKE.`;
      } else {
        reason = `Weighted analysis leans FAKE — combined model and linguistic evidence (${(finalFakeScore * 100).toFixed(0)}%).`;
      }
    } else if (finalFakeScore < 0.40 && mlFakeScore <= 0.5) {
      final_label = "REAL";
      confidence = 1 - finalFakeScore;
      if (bothValid && t_label === "REAL" && h_label === "REAL") {
        reason = `Both models agree on REAL.`;
      } else {
        reason = `Weighted analysis indicates REAL — combined evidence score ${((1 - finalFakeScore) * 100).toFixed(0)}%.`;
      }
      if (clickbait_score > 40) {
        reason += ` Note: some clickbait patterns detected but not enough to override model predictions.`;
      }
    } else if (mlFakeScore >= 0.65) {
      // ML leans clearly fake but clean style pulled the blend down — not enough for FAKE, never REAL.
      final_label = "SUSPICIOUS";
      confidence = mlFakeScore;
      reason = `ML models lean FAKE (${(mlFakeScore * 100).toFixed(0)}% fake score). Clean writing style does not offset this, but there is not enough evidence to call it fake.`;
    } else {
      final_label = "UNCERTAIN";
      confidence = 0.5;
      reason = `Analysis is inconclusive — combined fake evidence is ${(finalFakeScore * 100).toFixed(0)}%, which is in the uncertain zone.`;
      if (mlFakeScore > 0.5) {
        reason += ` ML models lean FAKE (${(mlFakeScore * 100).toFixed(0)}%).`;
      }
      if (bothValid && t_label !== h_label) {
        reason += ` Models also disagree (Transformer: ${t_label}, LR: ${h_label}).`;
      }
    }

    if (!bothValid) {
      reason += ` (Only one model produced a usable prediction.)`;
    }
  }

  const insights = generateInsights(
    t_label, t_conf, h_label, h_conf,
    clickbait_score, punctuation_score, uppercase_ratio,
    wordCount, final_label
  );

  return {
    final_label,
    confidence: finalConfidence(confidence),
    agreement,
    reason,
    insights
  };
}

// ---------------------------------------------------------------------------
// Verification-aware decision
// ---------------------------------------------------------------------------

/**
 * Enhanced decision function WITH verification — v3.
 *
 * See the file header for weight profiles and override rules.
 *
 * Returns {
 *   final_label: "VERIFIED"|"VERIFIED FACT"|"LIKELY REAL"|"UNVERIFIED"|"SUSPICIOUS"|"LIKELY FAKE",
 *   confidence (0.1-0.99), agreement (0-100), reason, insights,
 *   verification_source: "wikipedia"|"gnews"|"both"|"none",
 *   input_type,
 *   weights: per-signal REAL scores 0-100 { wikipedia, gnews, ml, linguistic, clickbait, credibility }
 *            (0 for unavailable signals),
 *   effective_weights: renormalised weights actually used (fractions summing to 1)
 * }
 */
export function getEnhancedDecisionWithVerification(
  t_label, t_conf, h_label, h_conf,
  clickbait_score, punctuation_score, uppercase_ratio,
  wordCount, inputType, wikipedia, verification
) {
  clickbait_score = num(clickbait_score);
  punctuation_score = num(punctuation_score);
  uppercase_ratio = num(uppercase_ratio);
  wordCount = num(wordCount, 50);

  const agreement = calculateAgreement(t_conf, h_conf, t_label, h_label);

  // ---- Component scores (0-1, higher = more evidence for REAL) ----
  const ml = blendModels(t_label, t_conf, h_label, h_conf);
  const mlRealScore = ml.available ? ml.realScore : null;
  const mlFakeScore = ml.available ? ml.fakeScore : null;

  // Style-only linguistic score; clickbait is weighted separately below.
  const linguisticRealScore = 1 - computeLinguisticFakeScore(clickbait_score, punctuation_score, uppercase_ratio, false);
  const clickbaitRealScore = 1 - clamp(clickbait_score, 0, 100) / 100;

  const wikiAvailable = isWikiAvailable(wikipedia);
  const wikiContradicted = isWikiContradicted(wikipedia);
  const wikiScore = wikiAvailable ? clamp(num(wikipedia.verification_score) / 100, 0, 1) : 0;

  const newsAvailable = isNewsAvailable(verification);
  const gnewsScore = newsAvailable ? clamp(num(verification.verification_score) / 100, 0, 1) : 0;
  const trustedCount = newsAvailable ? Math.max(0, num(verification.trusted_source_count)) : 0;
  const supportingCount = newsAvailable ? Math.max(0, num(verification.supporting_articles)) : 0;
  const credibilityScore = Math.min(1.0, trustedCount / 4);

  const strongMlFake = mlFakeScore !== null && mlFakeScore > STRONG_ML_FAKE;
  const strongFakeSignals = strongMlFake || clickbait_score > STRONG_CLICKBAIT;

  // ---- Select weight profile; fall back to MIXED when the primary source is missing ----
  let profileType = inputType === "fact_claim" || inputType === "news_article" ? inputType : "mixed";
  if (profileType === "fact_claim" && !wikiAvailable && newsAvailable) profileType = "mixed";
  if (profileType === "news_article" && !newsAvailable && wikiAvailable) profileType = "mixed";

  let base;
  if (profileType === "fact_claim") {
    base = { wikipedia: FACT_WEIGHTS.WIKIPEDIA, ml: FACT_WEIGHTS.ML_MODELS, linguistic: FACT_WEIGHTS.LINGUISTIC, clickbait: FACT_WEIGHTS.CLICKBAIT };
  } else if (profileType === "news_article") {
    base = { gnews: NEWS_WEIGHTS.GNEWS, ml: NEWS_WEIGHTS.ML_MODELS, credibility: NEWS_WEIGHTS.SOURCE_CREDIBILITY, linguistic: NEWS_WEIGHTS.LINGUISTIC, clickbait: NEWS_WEIGHTS.CLICKBAIT };
  } else {
    base = { wikipedia: MIXED_WEIGHTS.WIKIPEDIA, gnews: MIXED_WEIGHTS.GNEWS, ml: MIXED_WEIGHTS.ML_MODELS, linguistic: MIXED_WEIGHTS.LINGUISTIC, clickbait: MIXED_WEIGHTS.CLICKBAIT };
  }

  const signals = {
    wikipedia: { score: wikiScore, available: wikiAvailable },
    gnews: { score: gnewsScore, available: newsAvailable },
    credibility: { score: credibilityScore, available: newsAvailable },
    ml: { score: mlRealScore, available: ml.available },
    linguistic: { score: linguisticRealScore, available: true },
    clickbait: { score: clickbaitRealScore, available: true }
  };

  // Renormalise over available signals
  let wTotal = 0;
  for (const k of Object.keys(base)) if (signals[k].available) wTotal += base[k];
  const effective_weights = {};
  let finalRealScore = 0;
  for (const k of Object.keys(base)) {
    const w = signals[k].available && wTotal > 0 ? base[k] / wTotal : 0;
    effective_weights[k] = parseFloat(w.toFixed(3));
    if (w > 0) finalRealScore += w * signals[k].score;
  }
  if (!Number.isFinite(finalRealScore)) finalRealScore = 0.5;

  const activeWeights = {
    wikipedia: Math.round(wikiScore * 100),
    gnews: Math.round(gnewsScore * 100),
    ml: ml.available ? Math.round(mlRealScore * 100) : 0,
    linguistic: Math.round(linguisticRealScore * 100),
    clickbait: Math.round(clickbaitRealScore * 100),
    credibility: Math.round(credibilityScore * 100)
  };

  const verificationSource = wikiAvailable && newsAvailable ? "both" :
                             wikiAvailable || wikiContradicted ? "wikipedia" :
                             newsAvailable ? "gnews" : "none";

  const mlLean = !ml.available
    ? "ML models unavailable"
    : mlRealScore >= 0.6 ? `ML models lean REAL (${(mlRealScore * 100).toFixed(0)}%)`
    : mlRealScore <= 0.4 ? `ML models lean FAKE (${(mlFakeScore * 100).toFixed(0)}%)`
    : `ML models are inconclusive (${(mlRealScore * 100).toFixed(0)}% real)`;

  const strongFakeNote = () => {
    const parts = [];
    if (strongMlFake) parts.push(`ML models strongly flag this as fake (${(mlFakeScore * 100).toFixed(0)}%)`);
    if (clickbait_score > STRONG_CLICKBAIT) parts.push(`heavy clickbait patterns (${clickbait_score}%)`);
    return parts.join(' and ');
  };

  // ---- Override rules ----
  let final_label = "UNVERIFIED";
  let confidence = 0.5;
  let reason = "";

  // RULE 0: Wikipedia contradicts the claim
  if (wikiContradicted) {
    if (newsAvailable && gnewsScore > 0.50 && trustedCount >= 2) {
      final_label = "SUSPICIOUS";
      confidence = 0.6;
      reason = `Conflicting evidence: Wikipedia contradicts this claim, but ${trustedCount} trusted news sources report it (verification score ${num(verification.verification_score)}%). Treat with caution.`;
    } else {
      final_label = "LIKELY FAKE";
      confidence = 0.85;
      reason = wikipedia.message || 'Wikipedia contradicts this claim.';
    }
  }
  // RULE 1: Wikipedia confirms a (non-news) claim and ML does not strongly disagree
  else if (wikiAvailable && wikiScore > 0.70 && inputType !== "news_article" && !strongMlFake) {
    final_label = "VERIFIED FACT";
    confidence = Math.max(finalRealScore, 0.85);
    reason = wikipedia.message || `Wikipedia confirms this claim.`;
    if (clickbait_score > 50) {
      reason += ` Note: clickbait patterns detected (${clickbait_score}%) but overridden by Wikipedia evidence.`;
    }
  }
  // RULE 2: Multiple trusted sources confirm
  else if (newsAvailable && gnewsScore > 0.50 && trustedCount >= 2) {
    if (strongFakeSignals) {
      final_label = "LIKELY REAL";
      confidence = Math.max(Math.min(finalRealScore, 0.75), 0.6);
      reason = `${trustedCount} trusted news sources report matching content (verification score ${num(verification.verification_score)}%), but ${strongFakeNote()} — downgraded from VERIFIED. Check the sources directly.`;
    } else {
      final_label = "VERIFIED";
      confidence = Math.max(finalRealScore, 0.85);
      reason = `Multiple trusted news organizations independently confirm this claim. Verification score: ${num(verification.verification_score)}% with ${trustedCount} trusted sources.`;
      if (clickbait_score > 50) {
        reason += ` Note: clickbait patterns detected (${clickbait_score}%) but overridden by strong real-world evidence.`;
      }
    }
  }
  // RULE 2b: Single trusted source with a strong match
  else if (newsAvailable && gnewsScore >= 0.50 && trustedCount >= 1) {
    if (strongFakeSignals) {
      final_label = "LIKELY REAL";
      confidence = Math.max(Math.min(finalRealScore, 0.7), 0.55);
      reason = `A trusted news source reports matching content (verification score ${num(verification.verification_score)}%), but ${strongFakeNote()} — downgraded from VERIFIED. Check the source directly.`;
    } else {
      final_label = "VERIFIED";
      confidence = Math.max(finalRealScore, 0.80);
      reason = `Trusted news source confirms this claim. Verification score: ${num(verification.verification_score)}% with ${trustedCount} trusted source(s).`;
    }
  }
  // RULE 2c: Many supporting articles (even without trusted ones), no strong fake signals
  else if (newsAvailable && gnewsScore > 0.25 && supportingCount >= 3 && !strongFakeSignals) {
    final_label = "LIKELY REAL";
    confidence = Math.max(finalRealScore, 0.75);
    reason = `${supportingCount} online sources report similar content. Verification score: ${num(verification.verification_score)}%.`;
  }
  // RULE 3: No verification evidence — never fake purely from absence of evidence
  else if (!wikiAvailable && !newsAvailable) {
    if (ml.available && mlRealScore < 0.2 && finalRealScore < 0.5) {
      final_label = "LIKELY FAKE";
      confidence = 1 - finalRealScore;
      reason = `No online evidence found, and ${mlLean}.`;
      if (clickbait_score > 40) reason += ` Clickbait patterns detected (${clickbait_score}%).`;
    } else {
      final_label = "UNVERIFIED";
      // Without evidence, confidence in an UNVERIFIED verdict is capped (it reflects only the ML lean).
      confidence = ml.available && finalRealScore > 0.55 ? Math.min(finalRealScore, 0.70) : 0.5;
      reason = `No online evidence found to confirm or refute this claim. ${mlLean}, but this cannot be verified.`;
    }
  }
  // RULE 4: Weighted decision
  else {
    if (finalRealScore > 0.65) {
      final_label = wikiAvailable && wikiScore > 0.50 && inputType !== "news_article" ? "VERIFIED FACT" : "VERIFIED";
      confidence = finalRealScore;
      const scoreSource = wikiAvailable ? `Wikipedia: ${num(wikipedia.verification_score)}%` : `News: ${num(verification?.verification_score)}%`;
      reason = `Strong combined evidence supports this claim. ${scoreSource}.`;
    } else if (finalRealScore > 0.45) {
      final_label = "LIKELY REAL";
      confidence = finalRealScore;
      reason = `Multiple signals suggest this content is likely real.`;
      if (supportingCount > 0) reason += ` ${supportingCount} supporting articles found online.`;
      if (wikiAvailable) reason += ` Wikipedia contains related information.`;
    } else if (finalRealScore > 0.30) {
      final_label = "UNVERIFIED";
      confidence = 0.5;
      reason = `Analysis is inconclusive — mixed signals from ML models and online verification.`;
    } else if (finalRealScore > 0.20) {
      final_label = "SUSPICIOUS";
      confidence = 1 - finalRealScore;
      reason = `Multiple signals raise concerns.`;
      if (newsAvailable) reason += ` Low verification score (${num(verification.verification_score)}%).`;
    } else {
      final_label = "LIKELY FAKE";
      confidence = 1 - finalRealScore;
      reason = `Strong evidence suggests this content is not reliable. Low verification and ML models flag suspicious patterns.`;
    }

    // Conflicting signals: partial evidence vs. strong ML fake → downgrade one step
    if (strongMlFake && (final_label === "VERIFIED" || final_label === "VERIFIED FACT")) {
      final_label = "LIKELY REAL";
      confidence = Math.min(confidence, 0.7);
      reason += ` Downgraded: ${strongFakeNote()}.`;
    } else if (strongMlFake && final_label === "LIKELY REAL") {
      final_label = "UNVERIFIED";
      confidence = 0.5;
      reason = `Conflicting evidence: partial online support was found${wikiAvailable ? ' (Wikipedia contains related information)' : ''}, but ${strongFakeNote()}. The claim could not be verified.`;
    }
  }

  const insights = generateInsightsWithVerification(
    t_label, t_conf, h_label, h_conf,
    clickbait_score, punctuation_score, uppercase_ratio,
    wordCount, final_label, wikipedia, verification
  );

  return {
    final_label,
    confidence: finalConfidence(confidence),
    agreement,
    reason,
    insights,
    verification_source: verificationSource,
    input_type: inputType,
    weights: activeWeights,
    effective_weights
  };
}

// ---------------------------------------------------------------------------
// Robustness
// ---------------------------------------------------------------------------

/** Accept agreement on either a 0-1 or 0-100 scale; return 0-100. */
function normaliseAgreement(agreement) {
  const a = num(agreement, 0);
  return a <= 1 ? a * 100 : a;
}

/**
 * Calculate analysis robustness score (0-10).
 * `agreement` is 0-100 (values ≤ 1 are treated as a 0-1 fraction).
 */
export function calculateRobustness(t_conf, h_conf, agreement, wordCount) {
  const confs = [t_conf, h_conf].filter(Number.isFinite).map(c => clamp(c, 0, 1));
  const avgConf = confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : 0.5;
  let baseScore = avgConf * 10;

  const agr = normaliseAgreement(agreement);
  if (agr < 50) baseScore -= 1.5;
  else if (agr > 80) baseScore += 0.5;

  const wc = num(wordCount, 50);
  if (wc < 10) baseScore -= 2.5;
  else if (wc < 20) baseScore -= 1.5;
  else if (wc < 50) baseScore -= 0.5;
  else if (wc > 100) baseScore += 1.0;

  return Number(clamp(baseScore, 0, 10).toFixed(1));
}

/**
 * Calculate enhanced robustness that includes verification signals.
 * Only genuine evidence (see isWikiAvailable / isNewsAvailable) adds robustness.
 */
export function calculateRobustnessWithVerification(t_conf, h_conf, agreement, wordCount, wikipedia, verification) {
  let baseScore = calculateRobustness(t_conf, h_conf, agreement, wordCount);

  // Wikipedia boost
  if (isWikiAvailable(wikipedia)) {
    const wikiScore = num(wikipedia.verification_score) / 100;
    if (wikiScore > 0.7) baseScore += 2.0;
    else if (wikiScore > 0.4) baseScore += 1.0;
  } else if (isWikiContradicted(wikipedia)) {
    baseScore += 1.0; // a clear contradiction is also a firm signal
  }

  // GNews boost
  if (verification && !verification.error) {
    const verScore = num(verification.verification_score) / 100;
    const trustedCount = num(verification.trusted_source_count);
    const supporting = num(verification.supporting_articles);

    if (supporting > 0 && verScore > 0.7 && trustedCount >= 2) baseScore += 2.0;
    else if (supporting > 0 && verScore > 0.5) baseScore += 1.0;
    else if (verScore < 0.2 && supporting === 0) baseScore -= 1.0;
  }

  return Number(clamp(baseScore, 0, 10).toFixed(1));
}
