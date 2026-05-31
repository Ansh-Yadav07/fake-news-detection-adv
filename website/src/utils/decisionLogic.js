/**
 * Decision Logic for Fake News Detection
 * 
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
 * Generate rich insights based on the analysis results.
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
 * Core decision function — determines the final FAKE/REAL/UNCERTAIN verdict.
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
