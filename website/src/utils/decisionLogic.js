/**
 * Decision Logic for Fake News Detection — v2 (Parallel Verification)
 * 
 * === INPUT-AWARE WEIGHTING ===
 * 
 * Fact Claims (< 25 words, factual):
 *   Wikipedia       = 60%
 *   ML Models       = 20%
 *   Linguistic      = 10%
 *   Clickbait       = 10%
 * 
 * News Articles (> 25 words, events):
 *   GNews           = 50%
 *   ML Models       = 20%
 *   Source Credibility = 15%
 *   Linguistic      = 10%
 *   Clickbait       = 5%
 * 
 * Mixed (both sources):
 *   Wikipedia       = 30%
 *   GNews           = 25%
 *   ML Models       = 20%
 *   Linguistic      = 15%
 *   Clickbait       = 10%
 * 
 * === WITHOUT VERIFICATION (legacy fallback) ===
 * Text-length-based ML/linguistic weighting (unchanged)
 * 
 * Verdict Categories: VERIFIED, VERIFIED FACT, LIKELY REAL, UNVERIFIED, SUSPICIOUS, LIKELY FAKE
 */

const TRANSFORMER_WEIGHT = 0.60;
const LR_WEIGHT = 0.40;

// Weight profiles by input type
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

/**
 * Calculate weighted agreement between the two models.
 * Returns 0-100 percentage.
 */
export function calculateAgreement(t_conf, h_conf, t_label, h_label) {
  if (t_label === h_label) {
    const avgConf = (t_conf + h_conf) / 2;
    return Math.round(avgConf * 100);
  }
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

  // Text length analysis
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
 * Generate Wikipedia-specific insights.
 */
function generateWikipediaInsights(wikipedia) {
  const insights = [];
  if (!wikipedia) return insights;

  const status = wikipedia.status || "NOT FOUND";
  const score = wikipedia.verification_score || 0;
  const title = wikipedia.wiki_title || "";

  if (status === "CONTRADICTED") {
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
    insights.push(`Wikipedia article "${title}" found but claim could not be fully verified (${score}%).`);
  }

  // Add description
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

  if (verification.insights) {
    verification.insights.forEach(insight => {
      insights.push(insight);
    });
  }

  return insights;
}

/**
 * Generate enhanced insights that include ALL verification findings.
 */
function generateInsightsWithVerification(t_label, t_conf, h_label, h_conf, clickbait_score, punctuation_score, uppercase_ratio, wordCount, final_label, wikipedia, verification) {
  // Start with base ML/linguistic insights
  const insights = generateInsights(t_label, t_conf, h_label, h_conf, clickbait_score, punctuation_score, uppercase_ratio, wordCount, final_label);

  // Add Wikipedia insights
  const wikiInsights = generateWikipediaInsights(wikipedia);
  wikiInsights.forEach(i => insights.push(i));

  // Add GNews insights
  const gnewsInsights = generateGNewsInsights(verification);
  gnewsInsights.forEach(i => insights.push(i));

  return insights;
}

/**
 * Core decision function — determines the final verdict.
 * LEGACY: Used when verification data is NOT available (fallback).
 */
export function getEnhancedDecision(t_label, t_conf, h_label, h_conf, clickbait_score, punctuation_score, uppercase_ratio, wordCount = 50) {
  let final_label = "UNCERTAIN";
  let confidence = 0;
  let reason = "";

  const agreement = calculateAgreement(t_conf, h_conf, t_label, h_label);

  // STEP 1: Compute linguistic fake score
  const linguisticFakeScore = computeLinguisticFakeScore(clickbait_score, punctuation_score, uppercase_ratio);

  // STEP 2: Determine trust weights based on text length
  let modelTrust, linguisticTrust;

  if (wordCount < 10) {
    modelTrust = 0.15;
    linguisticTrust = 0.85;
  } else if (wordCount < 20) {
    modelTrust = 0.50;
    linguisticTrust = 0.50;
  } else if (wordCount < 50) {
    modelTrust = 0.75;
    linguisticTrust = 0.25;
  } else {
    modelTrust = 0.85;
    linguisticTrust = 0.15;
  }

  // STEP 3: Compute ML model fake score
  const t_fake_score = t_label === "FAKE" ? t_conf : (1 - t_conf);
  const h_fake_score = h_label === "FAKE" ? h_conf : (1 - h_conf);
  const mlFakeScore = (TRANSFORMER_WEIGHT * t_fake_score) + (LR_WEIGHT * h_fake_score);

  // STEP 4: Weighted final fake score
  const finalFakeScore = (modelTrust * mlFakeScore) + (linguisticTrust * linguisticFakeScore);

  // STEP 5: Determine verdict
  if (finalFakeScore > 0.55) {
    final_label = "FAKE";
    confidence = finalFakeScore;

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

    if (clickbait_score > 40) {
      reason += ` Note: some clickbait patterns detected but not enough to override model predictions.`;
    }

  } else {
    final_label = "UNCERTAIN";
    confidence = 0.5;
    reason = `Analysis is inconclusive — combined fake evidence is ${(finalFakeScore * 100).toFixed(0)}%, which is in the uncertain zone.`;

    if (t_label !== h_label) {
      reason += ` Models also disagree (Transformer: ${t_label}, LR: ${h_label}).`;
    }
  }

  confidence = Math.max(0.1, Math.min(0.99, confidence));

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
 * Enhanced decision function WITH verification — v2 (parallel).
 * 
 * Routes to different weight profiles based on input type:
 * - fact_claim: Wikipedia is dominant signal (60%)
 * - news_article: GNews is dominant signal (50%)
 * - mixed: Both sources combined
 * 
 * Verdict categories: VERIFIED, VERIFIED FACT, LIKELY REAL, UNVERIFIED, SUSPICIOUS, LIKELY FAKE
 * 
 * Override rules:
 * - Wikipedia score > 70 for fact claims → VERIFIED FACT
 * - GNews verification_score > 80 AND trusted_sources >= 3 → VERIFIED
 * - No evidence at all → UNVERIFIED (never auto-fake)
 * - Contradicting evidence → SUSPICIOUS
 */
export function getEnhancedDecisionWithVerification(
  t_label, t_conf, h_label, h_conf,
  clickbait_score, punctuation_score, uppercase_ratio,
  wordCount, inputType, wikipedia, verification
) {
  const agreement = calculateAgreement(t_conf, h_conf, t_label, h_label);

  // ---- Compute component scores (0-1 scale, higher = more evidence for REAL) ----

  // ML Model score
  const t_real_score = t_label === "REAL" ? t_conf : (1 - t_conf);
  const h_real_score = h_label === "REAL" ? h_conf : (1 - h_conf);
  const mlRealScore = (TRANSFORMER_WEIGHT * t_real_score) + (LR_WEIGHT * h_real_score);

  // Linguistic "real" score (inverse of fake score)
  const linguisticFakeScore = computeLinguisticFakeScore(clickbait_score, punctuation_score, uppercase_ratio);
  const linguisticRealScore = 1 - linguisticFakeScore;

  // Clickbait "real" score
  const clickbaitRealScore = 1 - (clickbait_score / 100);

  // Wikipedia score
  const wikiScore = wikipedia ? (wikipedia.verification_score || 0) / 100 : 0;
  const wikiAvailable = wikipedia && wikipedia.status && wikipedia.status !== "NOT FOUND";
  const wikiContradicted = wikipedia && wikipedia.is_contradicted === true;

  // GNews score
  const gnewsScore = verification ? (verification.verification_score || 0) / 100 : 0;
  const gnewsAvailable = verification && verification.supporting_articles > 0;
  const trustedCount = verification ? (verification.trusted_source_count || 0) : 0;
  const supportingCount = verification ? (verification.supporting_articles || 0) : 0;
  const credibilityScore = Math.min(1.0, trustedCount / 4);

  // ---- Select weight profile based on input type ----
  let finalRealScore = 0;
  let activeWeights = {};
  let verificationSource = "none";

  if (inputType === "fact_claim") {
    // Fact claim → Wikipedia is dominant, but include GNews if available
    finalRealScore =
      (FACT_WEIGHTS.WIKIPEDIA * wikiScore) +
      (FACT_WEIGHTS.ML_MODELS * mlRealScore) +
      (FACT_WEIGHTS.LINGUISTIC * linguisticRealScore) +
      (FACT_WEIGHTS.CLICKBAIT * clickbaitRealScore);

    activeWeights = {
      wikipedia: Math.round(wikiScore * 100),
      ml: Math.round(mlRealScore * 100),
      linguistic: Math.round(linguisticRealScore * 100),
      clickbait: Math.round(clickbaitRealScore * 100),
      gnews: Math.round(gnewsScore * 100),
      credibility: Math.round(credibilityScore * 100)
    };
    verificationSource = gnewsAvailable ? "both" : "wikipedia";

  } else if (inputType === "news_article") {
    // News article → GNews is dominant
    finalRealScore =
      (NEWS_WEIGHTS.GNEWS * gnewsScore) +
      (NEWS_WEIGHTS.ML_MODELS * mlRealScore) +
      (NEWS_WEIGHTS.SOURCE_CREDIBILITY * credibilityScore) +
      (NEWS_WEIGHTS.LINGUISTIC * linguisticRealScore) +
      (NEWS_WEIGHTS.CLICKBAIT * clickbaitRealScore);

    activeWeights = {
      gnews: Math.round(gnewsScore * 100),
      ml: Math.round(mlRealScore * 100),
      credibility: Math.round(credibilityScore * 100),
      linguistic: Math.round(linguisticRealScore * 100),
      clickbait: Math.round(clickbaitRealScore * 100),
      wikipedia: Math.round(wikiScore * 100)
    };
    verificationSource = wikiAvailable ? "both" : "gnews";

  } else {
    // Mixed → use both sources
    finalRealScore =
      (MIXED_WEIGHTS.WIKIPEDIA * wikiScore) +
      (MIXED_WEIGHTS.GNEWS * gnewsScore) +
      (MIXED_WEIGHTS.ML_MODELS * mlRealScore) +
      (MIXED_WEIGHTS.LINGUISTIC * linguisticRealScore) +
      (MIXED_WEIGHTS.CLICKBAIT * clickbaitRealScore);

    activeWeights = {
      wikipedia: Math.round(wikiScore * 100),
      gnews: Math.round(gnewsScore * 100),
      ml: Math.round(mlRealScore * 100),
      linguistic: Math.round(linguisticRealScore * 100),
      clickbait: Math.round(clickbaitRealScore * 100),
      credibility: Math.round(credibilityScore * 100)
    };
    verificationSource = wikiAvailable && gnewsAvailable ? "both" :
                          wikiAvailable ? "wikipedia" :
                          gnewsAvailable ? "gnews" : "none";
  }

  // ---- Apply override rules and determine verdict ----
  let final_label = "UNVERIFIED";
  let confidence = 0;
  let reason = "";

  // OVERRIDE RULE 0: Wikipedia CONTRADICTS the claim
  if (wikiContradicted) {
    final_label = "LIKELY FAKE";
    confidence = 0.85;
    reason = wikipedia.message || 'Wikipedia contradicts this claim.';
  }
  // OVERRIDE RULE 1: Wikipedia confirms a fact claim
  else if (wikiAvailable && wikiScore > 0.70) {
    final_label = "VERIFIED FACT";
    confidence = Math.max(finalRealScore, 0.85);
    reason = `Wikipedia confirms this claim. ${wikipedia.message || ''}`;

    if (clickbait_score > 50) {
      reason += ` Note: clickbait patterns detected (${clickbait_score}%) but overridden by Wikipedia evidence.`;
    }
  }
  // OVERRIDE RULE 2: Strong news verification (trusted source confirms)
  else if (gnewsScore > 0.50 && trustedCount >= 2) {
    final_label = "VERIFIED";
    confidence = Math.max(finalRealScore, 0.85);
    reason = `Multiple trusted news organizations independently confirm this claim. Verification score: ${verification.verification_score}% with ${trustedCount} trusted sources.`;

    if (clickbait_score > 50) {
      reason += ` Note: clickbait patterns detected (${clickbait_score}%) but overridden by strong real-world evidence.`;
    }
  }
  // OVERRIDE RULE 2b: Single trusted source with strong match
  else if (gnewsScore > 0.35 && trustedCount >= 1) {
    final_label = "VERIFIED";
    confidence = Math.max(finalRealScore, 0.80);
    reason = `Trusted news source confirms this claim. Verification score: ${verification.verification_score}% with ${trustedCount} trusted source(s).`;
  }
  // OVERRIDE RULE 2c: Many supporting articles (even without trusted)
  else if (gnewsScore > 0.25 && supportingCount >= 3) {
    final_label = "LIKELY REAL";
    confidence = Math.max(finalRealScore, 0.75);
    reason = `${supportingCount} online sources report similar content. Verification score: ${verification.verification_score}%.`;
  }
  // OVERRIDE RULE 3: No evidence found at all
  else if (!wikiAvailable && !gnewsAvailable) {
    if (finalRealScore > 0.55) {
      final_label = "UNVERIFIED";
      confidence = finalRealScore;
      reason = `No online evidence found. ML models suggest real content but cannot be verified.`;
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
  // Normal weighted decision
  else if (finalRealScore > 0.65) {
    if (wikiAvailable && wikiScore > 0.50) {
      final_label = "VERIFIED FACT";
    } else {
      final_label = "VERIFIED";
    }
    confidence = finalRealScore;
    const scoreSource = wikiAvailable ? `Wikipedia: ${wikipedia?.verification_score}%` : `News: ${verification?.verification_score}%`;
    reason = `Strong combined evidence supports this claim. ${scoreSource}.`;
  } else if (finalRealScore > 0.45) {
    final_label = "LIKELY REAL";
    confidence = finalRealScore;
    reason = `Multiple signals suggest this content is likely real.`;
    if (supportingCount > 0) {
      reason += ` ${supportingCount} supporting articles found online.`;
    }
    if (wikiAvailable) {
      reason += ` Wikipedia contains related information.`;
    }
  } else if (finalRealScore > 0.30) {
    final_label = "UNVERIFIED";
    confidence = 0.5;
    reason = `Analysis is inconclusive — mixed signals from ML models and online verification.`;
  } else if (finalRealScore > 0.20) {
    final_label = "SUSPICIOUS";
    confidence = 1 - finalRealScore;
    reason = `Multiple signals raise concerns.`;
    if (verification) {
      reason += ` Low verification score (${verification.verification_score}%).`;
    }
  } else {
    final_label = "LIKELY FAKE";
    confidence = 1 - finalRealScore;
    reason = `Strong evidence suggests this content is not reliable. Low verification and ML models flag suspicious patterns.`;
  }

  // Final confidence bounds
  confidence = Math.max(0.1, Math.min(0.99, confidence));

  // Generate insights
  const insights = generateInsightsWithVerification(
    t_label, t_conf, h_label, h_conf,
    clickbait_score, punctuation_score, uppercase_ratio,
    wordCount, final_label, wikipedia, verification
  );

  return {
    final_label,
    confidence: parseFloat(confidence.toFixed(3)),
    agreement,
    reason,
    insights,
    verification_source: verificationSource,
    input_type: inputType,
    weights: activeWeights
  };
}

/**
 * Calculate analysis robustness score (0-10).
 */
export function calculateRobustness(t_conf, h_conf, agreement, wordCount) {
  let baseScore = ((t_conf + h_conf) / 2) * 10;

  if (agreement < 50) baseScore -= 1.5;
  else if (agreement > 80) baseScore += 0.5;

  if (wordCount < 10) baseScore -= 2.5;
  else if (wordCount < 20) baseScore -= 1.5;
  else if (wordCount < 50) baseScore -= 0.5;
  else if (wordCount > 100) baseScore += 1.0;

  const finalScore = Math.max(0, Math.min(10, baseScore));
  return Number(finalScore.toFixed(1));
}

/**
 * Calculate enhanced robustness that includes verification signals.
 */
export function calculateRobustnessWithVerification(t_conf, h_conf, agreement, wordCount, wikipedia, verification) {
  let baseScore = calculateRobustness(t_conf, h_conf, agreement, wordCount);

  // Wikipedia boost
  if (wikipedia) {
    const wikiScore = (wikipedia.verification_score || 0) / 100;
    if (wikiScore > 0.7) baseScore += 2.0;
    else if (wikiScore > 0.4) baseScore += 1.0;
  }

  // GNews boost
  if (verification) {
    const verScore = (verification.verification_score || 0) / 100;
    const trustedCount = verification.trusted_source_count || 0;

    if (verScore > 0.7 && trustedCount >= 2) baseScore += 2.0;
    else if (verScore > 0.5) baseScore += 1.0;
    else if (verScore < 0.2 && verification.supporting_articles === 0) baseScore -= 1.0;
  }

  return Number(Math.max(0, Math.min(10, baseScore)).toFixed(1));
}
