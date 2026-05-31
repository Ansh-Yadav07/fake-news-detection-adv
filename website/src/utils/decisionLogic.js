/**
 * Decision Logic for Fake News Detection
 * 
 * === WITH VERIFICATION (new flow) ===
 * Weights:
 *   Online Verification = 50%
 *   ML Models          = 20%
 *   Source Credibility  = 15%
 *   Linguistic Features = 10%
 *   Clickbait Detection = 5%
 * 
 * Verdict Categories: VERIFIED, LIKELY REAL, UNVERIFIED, SUSPICIOUS, LIKELY FAKE
 * 
 * === WITHOUT VERIFICATION (legacy fallback) ===
 * Key principles:
 * - For LONG texts (50+ words): ML models are dominant, linguistics are minor modifiers
 * - For MEDIUM texts (15-50 words): Balanced between models and linguistics
 * - For SHORT texts (<15 words): Linguistic features carry MUCH more weight
 *   because ML models trained on full articles are unreliable on headlines
 * 
 * - Transformer model gets higher weight (60%) as it's the stronger model
 * - Clickbait/linguistic scores can shift or flip verdict for short texts
 */

const TRANSFORMER_WEIGHT = 0.60;
const LR_WEIGHT = 0.40;

// Verification-aware weights
const WEIGHTS = {
  ONLINE_VERIFICATION: 0.50,
  ML_MODELS: 0.20,
  SOURCE_CREDIBILITY: 0.15,
  LINGUISTIC_FEATURES: 0.10,
  CLICKBAIT_DETECTION: 0.05
};

/**
 * Calculate weighted agreement between the two models.
 * Returns 0-100 percentage.
 */
export function calculateAgreement(t_conf, h_conf, t_label, h_label) {
  if (t_label === h_label) {
    const avgConf = (t_conf + h_conf) / 2;
    return Math.round(avgConf * 100);
  }
  // They disagree — agreement is inversely proportional to confidence
  const disagreement = (t_conf + h_conf) / 2;
  return Math.round(Math.max(0, (1 - disagreement) * 100));
}

/**
 * Compute a "linguistic fake score" from raw feature signals.
 * Returns 0-1 where higher = more likely fake based on writing style alone.
 */
function computeLinguisticFakeScore(clickbait_score, punctuation_score, uppercase_ratio) {
  let score = 0;

  // Clickbait is the strongest linguistic signal
  score += (clickbait_score / 100) * 0.50;

  // Excessive uppercase (> 20%) is very suspicious
  if (uppercase_ratio > 20) {
    score += Math.min((uppercase_ratio - 20) / 80, 1.0) * 0.30;
  }

  // Heavy punctuation
  if (punctuation_score > 5) {
    score += Math.min((punctuation_score - 5) / 30, 1.0) * 0.20;
  }

  return Math.min(score, 1.0);
}

/**
 * Generate rich insights based on the analysis results (legacy — without verification).
 */
function generateInsights(t_label, t_conf, h_label, h_conf, clickbait_score, punctuation_score, uppercase_ratio, wordCount, final_label) {
  const insights = [];

  // Model agreement insight
  if (t_label === h_label) {
    const avgConf = ((t_conf + h_conf) / 2 * 100).toFixed(0);
    insights.push(`Both AI models agree: this content is likely ${t_label} with ${avgConf}% average confidence.`);
  } else {
    insights.push(`Models disagree — Transformer predicts ${t_label} (${(t_conf * 100).toFixed(0)}%) while LR model predicts ${h_label} (${(h_conf * 100).toFixed(0)}%).`);
  }

  // Clickbait analysis
  if (clickbait_score > 60) {
    insights.push(`High clickbait indicators detected (${clickbait_score}%) — sensationalist language, excessive punctuation, or ALL CAPS patterns found.`);
  } else if (clickbait_score > 30) {
    insights.push(`Moderate clickbait signals (${clickbait_score}%) — some sensational language patterns detected.`);
  } else {
    insights.push(`Low clickbait score (${clickbait_score}%) — writing style appears measured and professional.`);
  }

  // Text length analysis — critical for understanding model reliability
  if (wordCount < 10) {
    insights.push(`Very short text (${wordCount} words) — ML models are unreliable on headlines this short. Linguistic analysis weighted heavily.`);
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
  const maxConf = Math.max(t_conf, h_conf);
  if (maxConf < 0.6) {
    insights.push(`Both models show low confidence — this content may contain mixed signals or be genuinely ambiguous.`);
  } else if (maxConf > 0.9 && final_label !== "UNCERTAIN") {
    insights.push(`High model confidence (${(maxConf * 100).toFixed(0)}%) — strong signal detected in the content patterns.`);
  }

  return insights;
}

/**
 * Generate enhanced insights that include verification findings.
 */
function generateInsightsWithVerification(t_label, t_conf, h_label, h_conf, clickbait_score, punctuation_score, uppercase_ratio, wordCount, final_label, verification) {
  // Start with base ML/linguistic insights
  const insights = generateInsights(t_label, t_conf, h_label, h_conf, clickbait_score, punctuation_score, uppercase_ratio, wordCount, final_label);

  // Append verification-specific insights from the backend
  if (verification && verification.insights) {
    verification.insights.forEach(insight => {
      insights.push(insight);
    });
  }

  return insights;
}

/**
 * Core decision function — determines the final FAKE/REAL/UNCERTAIN verdict.
 * LEGACY: Used when verification data is NOT available (fallback).
 * 
 * The key insight: ML models trained on full articles are UNRELIABLE on short
 * headlines (< 15 words). For short text, linguistic features (clickbait,
 * uppercase, punctuation) carry much more weight.
 * 
 * Logic flow:
 * 1. Compute linguistic fake score from writing style
 * 2. Determine how much to trust ML models vs linguistics (based on text length)
 * 3. Compute weighted final score
 * 4. Apply final threshold checks
 */
export function getEnhancedDecision(t_label, t_conf, h_label, h_conf, clickbait_score, punctuation_score, uppercase_ratio, wordCount = 50) {
  let final_label = "UNCERTAIN";
  let confidence = 0;
  let reason = "";

  const agreement = calculateAgreement(t_conf, h_conf, t_label, h_label);

  // ---- STEP 1: Compute linguistic fake score ----
  const linguisticFakeScore = computeLinguisticFakeScore(clickbait_score, punctuation_score, uppercase_ratio);

  // ---- STEP 2: Determine trust weights based on text length ----
  // Short text → trust linguistics more, models less
  // Long text → trust models more, linguistics less
  let modelTrust, linguisticTrust;

  if (wordCount < 10) {
    // Ultra-short: headlines, tweets — models trained on articles are unreliable here
    modelTrust = 0.15;
    linguisticTrust = 0.85;
  } else if (wordCount < 20) {
    // Short: brief headlines
    modelTrust = 0.50;
    linguisticTrust = 0.50;
  } else if (wordCount < 50) {
    // Medium: paragraphs
    modelTrust = 0.75;
    linguisticTrust = 0.25;
  } else {
    // Long: full articles — models are most reliable
    modelTrust = 0.85;
    linguisticTrust = 0.15;
  }

  // ---- STEP 3: Compute ML model fake score ----
  // Higher = more evidence for FAKE from ML models
  const t_fake_score = t_label === "FAKE" ? t_conf : (1 - t_conf);
  const h_fake_score = h_label === "FAKE" ? h_conf : (1 - h_conf);
  const mlFakeScore = (TRANSFORMER_WEIGHT * t_fake_score) + (LR_WEIGHT * h_fake_score);

  // ---- STEP 4: Weighted final fake score ----
  const finalFakeScore = (modelTrust * mlFakeScore) + (linguisticTrust * linguisticFakeScore);

  // ---- STEP 5: Determine verdict ----
  if (finalFakeScore > 0.55) {
    final_label = "FAKE";
    confidence = finalFakeScore;

    // Build reason
    if (linguisticFakeScore > 0.4 && mlFakeScore < 0.5) {
      reason = `Linguistic analysis overrides uncertain ML models — strong clickbait/sensationalist patterns detected (${clickbait_score}% clickbait score).`;
    } else if (t_label === "FAKE" && h_label === "FAKE") {
      reason = `Both models agree on FAKE.`;
    } else if (t_label === h_label) {
      reason = `ML models and linguistic analysis combine to indicate FAKE.`;
    } else {
      reason = `Weighted analysis leans FAKE — combined model and linguistic evidence (${(finalFakeScore * 100).toFixed(0)}%).`;
    }

  } else if (finalFakeScore < 0.40) {
    final_label = "REAL";
    confidence = 1 - finalFakeScore;

    if (t_label === "REAL" && h_label === "REAL") {
      reason = `Both models agree on REAL.`;
    } else {
      reason = `Weighted analysis indicates REAL — combined evidence score ${((1 - finalFakeScore) * 100).toFixed(0)}%.`;
    }

    // Additional check: if clickbait is high, add a warning
    if (clickbait_score > 40) {
      reason += ` Note: some clickbait patterns detected but not enough to override model predictions.`;
    }

  } else {
    // Between 0.40 and 0.55 — genuinely uncertain
    final_label = "UNCERTAIN";
    confidence = 0.5;
    reason = `Analysis is inconclusive — combined fake evidence is ${(finalFakeScore * 100).toFixed(0)}%, which is in the uncertain zone.`;

    if (t_label !== h_label) {
      reason += ` Models also disagree (Transformer: ${t_label}, LR: ${h_label}).`;
    }
  }

  // ---- STEP 6: Final confidence bounds ----
  confidence = Math.max(0.1, Math.min(0.99, confidence));

  // Generate detailed insights
  const insights = generateInsights(
    t_label, t_conf, h_label, h_conf,
    clickbait_score, punctuation_score, uppercase_ratio,
    wordCount, final_label
  );

  return {
    final_label,
    confidence: parseFloat(confidence.toFixed(3)),
    agreement,
    reason,
    insights
  };
}

/**
 * Enhanced decision function WITH online verification.
 * 
 * Uses a weighted scoring system where online verification is the strongest signal:
 *   Online Verification = 50%
 *   ML Models          = 20%
 *   Source Credibility  = 15%
 *   Linguistic Features = 10%
 *   Clickbait Detection = 5%
 *
 * Verdict categories: VERIFIED, LIKELY REAL, UNVERIFIED, SUSPICIOUS, LIKELY FAKE
 * 
 * Override rules:
 * - If verification_score > 80 AND trusted_sources >= 3 → verification is primary signal
 * - If supporting_articles == 0 → UNVERIFIED (never auto-classify as fake)
 * - If contradicting evidence → SUSPICIOUS
 */
export function getEnhancedDecisionWithVerification(
  t_label, t_conf, h_label, h_conf,
  clickbait_score, punctuation_score, uppercase_ratio,
  wordCount, verification
) {
  const agreement = calculateAgreement(t_conf, h_conf, t_label, h_label);

  // ---- Compute component scores (0-1 scale, higher = more evidence for REAL) ----

  // 1. Online Verification score (0-1)
  const verificationRealScore = (verification.verification_score || 0) / 100;

  // 2. ML Model score (0-1, higher = more evidence for REAL)
  const t_real_score = t_label === "REAL" ? t_conf : (1 - t_conf);
  const h_real_score = h_label === "REAL" ? h_conf : (1 - h_conf);
  const mlRealScore = (TRANSFORMER_WEIGHT * t_real_score) + (LR_WEIGHT * h_real_score);

  // 3. Source Credibility score (0-1)
  const trustedCount = verification.trusted_source_count || 0;
  const supportingCount = verification.supporting_articles || 0;
  const credibilityScore = Math.min(1.0, trustedCount / 4); // 4+ trusted sources = max

  // 4. Linguistic "real" score (inverse of fake score, 0-1)
  const linguisticFakeScore = computeLinguisticFakeScore(clickbait_score, punctuation_score, uppercase_ratio);
  const linguisticRealScore = 1 - linguisticFakeScore;

  // 5. Clickbait "real" score (inverse, 0-1)
  const clickbaitRealScore = 1 - (clickbait_score / 100);

  // ---- Compute weighted final score ----
  const finalRealScore =
    (WEIGHTS.ONLINE_VERIFICATION * verificationRealScore) +
    (WEIGHTS.ML_MODELS * mlRealScore) +
    (WEIGHTS.SOURCE_CREDIBILITY * credibilityScore) +
    (WEIGHTS.LINGUISTIC_FEATURES * linguisticRealScore) +
    (WEIGHTS.CLICKBAIT_DETECTION * clickbaitRealScore);

  // ---- Apply override rules and determine verdict ----
  let final_label = "UNVERIFIED";
  let confidence = 0;
  let reason = "";

  const verificationStatus = verification.status || "UNVERIFIED";

  // OVERRIDE RULE 1: Strong verification override
  if (verificationRealScore > 0.80 && trustedCount >= 3) {
    final_label = "VERIFIED";
    confidence = Math.max(finalRealScore, 0.85);
    reason = `Multiple trusted news organizations independently confirm this claim. Verification score: ${verification.verification_score}% with ${trustedCount} trusted sources.`;

    // Even if clickbait is high, verified trumps it
    if (clickbait_score > 50) {
      reason += ` Note: clickbait patterns detected (${clickbait_score}%) but overridden by strong real-world evidence.`;
    }
  }
  // OVERRIDE RULE 2: No evidence found — UNVERIFIED, never auto-fake
  else if (supportingCount === 0) {
    // Fall back to ML-only decision but label as UNVERIFIED
    if (finalRealScore > 0.55) {
      final_label = "UNVERIFIED";
      confidence = finalRealScore;
      reason = `No online evidence found. ML models suggest real content but cannot be verified online.`;
    } else if (finalRealScore < 0.40) {
      final_label = "LIKELY FAKE";
      confidence = 1 - finalRealScore;
      reason = `No online evidence found and ML analysis indicates suspicious content.`;
    } else {
      final_label = "UNVERIFIED";
      confidence = 0.5;
      reason = `No online evidence available. ML analysis is inconclusive.`;
    }
  }
  // OVERRIDE RULE 3: Contradicting evidence
  else if (verificationStatus === "SUSPICIOUS") {
    final_label = "SUSPICIOUS";
    confidence = 0.6;
    reason = `Conflicting information detected from trusted sources. ${verification.message || ''}`;
  }
  // Normal weighted decision
  else if (finalRealScore > 0.70) {
    final_label = "VERIFIED";
    confidence = finalRealScore;
    reason = `Strong combined evidence supports this article. Verification score: ${verification.verification_score}%.`;
  } else if (finalRealScore > 0.55) {
    final_label = "LIKELY REAL";
    confidence = finalRealScore;
    reason = `Multiple signals suggest this article is likely real. ${supportingCount} supporting articles found online.`;
    if (trustedCount > 0) {
      reason += ` ${trustedCount} trusted source${trustedCount > 1 ? 's' : ''} found.`;
    }
  } else if (finalRealScore > 0.40) {
    final_label = "UNVERIFIED";
    confidence = 0.5;
    reason = `Analysis is inconclusive — mixed signals from ML models and online verification.`;
  } else if (finalRealScore > 0.25) {
    final_label = "SUSPICIOUS";
    confidence = 1 - finalRealScore;
    reason = `Multiple signals raise concerns. Low verification score (${verification.verification_score}%) combined with ML analysis.`;
  } else {
    final_label = "LIKELY FAKE";
    confidence = 1 - finalRealScore;
    reason = `Strong evidence suggests this content is not reliable. Low online verification and ML models flag suspicious patterns.`;
  }

  // ---- Final confidence bounds ----
  confidence = Math.max(0.1, Math.min(0.99, confidence));

  // Generate insights including verification
  const insights = generateInsightsWithVerification(
    t_label, t_conf, h_label, h_conf,
    clickbait_score, punctuation_score, uppercase_ratio,
    wordCount, final_label, verification
  );

  return {
    final_label,
    confidence: parseFloat(confidence.toFixed(3)),
    agreement,
    reason,
    insights,
    weights: {
      verification: Math.round(verificationRealScore * 100),
      ml: Math.round(mlRealScore * 100),
      credibility: Math.round(credibilityScore * 100),
      linguistic: Math.round(linguisticRealScore * 100),
      clickbait: Math.round(clickbaitRealScore * 100)
    }
  };
}

/**
 * Calculate analysis robustness score (0-10).
 * Considers model confidence, agreement, and text length.
 */
export function calculateRobustness(t_conf, h_conf, agreement, wordCount) {
  let baseScore = ((t_conf + h_conf) / 2) * 10;

  // Agreement bonus/penalty
  if (agreement < 50) baseScore -= 1.5;
  else if (agreement > 80) baseScore += 0.5;

  // Word count factor — short text = less robust
  if (wordCount < 10) baseScore -= 2.5;
  else if (wordCount < 20) baseScore -= 1.5;
  else if (wordCount < 50) baseScore -= 0.5;
  else if (wordCount > 100) baseScore += 1.0;

  const finalScore = Math.max(0, Math.min(10, baseScore));
  return Number(finalScore.toFixed(1));
}

/**
 * Calculate enhanced robustness that includes verification signal.
 */
export function calculateRobustnessWithVerification(t_conf, h_conf, agreement, wordCount, verification) {
  let baseScore = calculateRobustness(t_conf, h_conf, agreement, wordCount);

  if (verification) {
    const verScore = (verification.verification_score || 0) / 100;
    const trustedCount = verification.trusted_source_count || 0;

    // Verification boost
    if (verScore > 0.7 && trustedCount >= 2) baseScore += 2.0;
    else if (verScore > 0.5) baseScore += 1.0;
    else if (verScore < 0.2 && verification.supporting_articles === 0) baseScore -= 1.0;
  }

  return Number(Math.max(0, Math.min(10, baseScore)).toFixed(1));
}
