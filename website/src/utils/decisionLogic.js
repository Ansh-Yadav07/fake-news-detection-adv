/**
 * Decision Logic for Fake News Detection
 * 
 * Key principles:
 * - ML model predictions (Transformer + LR) are the PRIMARY signals
 * - Linguistic features (clickbait, uppercase, punctuation) are SECONDARY modifiers
 * - Clickbait score modifies confidence but does NOT override model predictions alone
 * - Transformer model gets higher weight (60%) as it's the stronger model
 */

const TRANSFORMER_WEIGHT = 0.60;
const LR_WEIGHT = 0.40;

/**
 * Calculate weighted agreement between the two models.
 * Returns 0-100 percentage.
 */
export function calculateAgreement(t_conf, h_conf, t_label, h_label) {
  if (t_label === h_label) {
    // Both agree — agreement is the average of their confidences
    const avgConf = (t_conf + h_conf) / 2;
    return Math.round(avgConf * 100);
  }
  // They disagree — agreement is inversely proportional to the confidence gap
  const confDiff = Math.abs(t_conf - h_conf);
  // If one is very confident and the other isn't, agreement is very low
  const disagreement = (t_conf + h_conf) / 2;
  return Math.round(Math.max(0, (1 - disagreement) * 100));
}

/**
 * Generate rich insights based on the analysis results.
 */
function generateInsights(t_label, t_conf, h_label, h_conf, clickbait_score, punctuation_score, uppercase_ratio, wordCount) {
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
  if (wordCount < 15) {
    insights.push(`Very short text (${wordCount} words) — insufficient content for reliable analysis. Longer articles produce more accurate results.`);
  } else if (wordCount < 50) {
    insights.push(`Moderate text length (${wordCount} words) — adequate for analysis but longer content yields higher accuracy.`);
  } else {
    insights.push(`Sufficient text length (${wordCount} words) — provides strong signal for model analysis.`);
  }

  // Uppercase analysis
  if (uppercase_ratio > 30) {
    insights.push(`Unusually high uppercase ratio (${uppercase_ratio}%) — excessive capitalization is common in misleading content.`);
  }

  // Punctuation analysis
  if (punctuation_score > 10) {
    insights.push(`Above-average punctuation density (${punctuation_score}%) — heavy punctuation can indicate sensationalized writing.`);
  }

  // Confidence insight
  const maxConf = Math.max(t_conf, h_conf);
  if (maxConf < 0.6) {
    insights.push(`Both models show low confidence — this content may contain mixed signals or be genuinely ambiguous.`);
  } else if (maxConf > 0.9) {
    insights.push(`High model confidence (${(maxConf * 100).toFixed(0)}%) — strong signal detected in the content patterns.`);
  }

  return insights;
}

/**
 * Core decision function — determines the final FAKE/REAL/UNCERTAIN verdict.
 * 
 * Logic:
 * 1. If both models agree → use that label, confidence = weighted average
 * 2. If they disagree → weighted vote (transformer 60%, LR 40%)
 * 3. Clickbait and linguistic features act as confidence MODIFIERS, not overrides
 * 4. Very short text (< 15 words) or very low confidence → UNCERTAIN
 */
export function getEnhancedDecision(t_label, t_conf, h_label, h_conf, clickbait_score, punctuation_score, uppercase_ratio, wordCount = 50) {
  let final_label = "UNCERTAIN";
  let confidence = 0;
  let reason = "";

  const agreement = calculateAgreement(t_conf, h_conf, t_label, h_label);

  // ---- STEP 1: Primary decision from ML models ----

  if (t_label === h_label) {
    // Both models agree — this is the strongest signal
    final_label = t_label;
    confidence = (TRANSFORMER_WEIGHT * t_conf) + (LR_WEIGHT * h_conf);
    reason = `Both models agree: content is ${t_label}.`;

  } else {
    // Models disagree — use weighted voting
    // Calculate weighted "FAKE score": how much evidence points to FAKE
    const t_fake_score = t_label === "FAKE" ? t_conf : (1 - t_conf);
    const h_fake_score = h_label === "FAKE" ? h_conf : (1 - h_conf);
    const weighted_fake = (TRANSFORMER_WEIGHT * t_fake_score) + (LR_WEIGHT * h_fake_score);

    if (weighted_fake > 0.55) {
      final_label = "FAKE";
      confidence = weighted_fake;
      reason = `Weighted model analysis leans toward FAKE (${(weighted_fake * 100).toFixed(0)}% weighted evidence).`;
    } else if (weighted_fake < 0.45) {
      final_label = "REAL";
      confidence = 1 - weighted_fake;
      reason = `Weighted model analysis leans toward REAL (${((1 - weighted_fake) * 100).toFixed(0)}% weighted evidence).`;
    } else {
      // Very close call — models truly disagree
      final_label = "UNCERTAIN";
      confidence = 0.5;
      reason = `Models disagree and weighted evidence is nearly balanced (${(weighted_fake * 100).toFixed(0)}% fake vs ${((1 - weighted_fake) * 100).toFixed(0)}% real).`;
    }
  }

  // ---- STEP 2: Linguistic feature modifiers ----
  // These adjust confidence but DON'T flip the verdict on their own

  let confidenceModifier = 0;

  if (clickbait_score > 60) {
    if (final_label === "REAL") {
      // High clickbait + REAL verdict → reduce confidence
      confidenceModifier -= 0.08;
      reason += " Note: high clickbait indicators detected, reducing confidence.";
    } else if (final_label === "FAKE") {
      // High clickbait + FAKE verdict → reinforces the FAKE signal
      confidenceModifier += 0.03;
    }
  }

  if (uppercase_ratio > 30) {
    if (final_label === "REAL") {
      confidenceModifier -= 0.05;
    }
  }

  // ---- STEP 3: Text length penalty ----
  if (wordCount < 10) {
    confidenceModifier -= 0.10;
    if (confidence + confidenceModifier < 0.55 && final_label !== "UNCERTAIN") {
      reason += " Very short text reduces analysis reliability.";
    }
  } else if (wordCount < 20) {
    confidenceModifier -= 0.05;
  }

  // Apply modifier
  confidence = Math.max(0.1, Math.min(0.99, confidence + confidenceModifier));

  // ---- STEP 4: Final uncertainty check ----
  if (confidence < 0.52 && final_label !== "UNCERTAIN") {
    final_label = "UNCERTAIN";
    reason = "Confidence too low for a definitive verdict. " + reason;
  }

  // Generate detailed insights
  const insights = generateInsights(
    t_label, t_conf, h_label, h_conf,
    clickbait_score, punctuation_score, uppercase_ratio,
    wordCount
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
  // Base: average model confidence scaled to 10
  let baseScore = ((t_conf + h_conf) / 2) * 10;

  // Agreement bonus/penalty
  if (agreement < 50) baseScore -= 1.5;
  else if (agreement > 80) baseScore += 0.5;

  // Word count factor
  if (wordCount < 10) baseScore -= 2.5;
  else if (wordCount < 20) baseScore -= 1.5;
  else if (wordCount < 50) baseScore -= 0.5;
  else if (wordCount > 100) baseScore += 1.0;

  const finalScore = Math.max(0, Math.min(10, baseScore));
  return Number(finalScore.toFixed(1));
}
