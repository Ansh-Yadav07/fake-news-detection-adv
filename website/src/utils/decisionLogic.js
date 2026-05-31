export function calculateAgreement(t_conf, h_conf, t_label, h_label) {
  if (t_label === h_label) {
    return (t_conf + h_conf) / 2;
  }
  return 1.0 - ((t_conf + h_conf) / 2);
}

export function getEnhancedDecision(t_label, t_conf, h_label, h_conf, clickbait_score, punctuation_score, uppercase_ratio) {
  let final_label = "UNCERTAIN";
  let confidence = 0;
  let reason = "";

  const agreementRaw = calculateAgreement(t_conf, h_conf, t_label, h_label);
  const agreement = parseFloat((Math.max(0, agreementRaw) * 100).toFixed(1));

  // 1. If both models agree: return that label
  if (t_label === h_label) {
    final_label = t_label;
    confidence = (0.7 * t_conf) + (0.3 * h_conf);
    reason = `Both models agree on ${t_label}.`;
  }
  // 2. If strong fake signal (clickbait_score > 70): return "FAKE"
  else if (clickbait_score > 70) {
    final_label = "FAKE";
    confidence = Math.max(t_conf, h_conf);
    reason = "Strong fake signal: high clickbait score detected.";
  }
  // 3. If hybrid strongly predicts fake: return "FAKE"
  else if (h_label === "FAKE" && h_conf > 0.6) {
    final_label = "FAKE";
    confidence = h_conf;
    reason = "Strong fake signal: linguistic features indicate FAKE.";
  }
  // 4. Only return UNCERTAIN if both confidences < 0.6 AND no strong signals
  else if (t_conf < 0.6 && h_conf < 0.6) {
    final_label = "UNCERTAIN";
    confidence = (t_conf + h_conf) / 2;
    reason = "Models disagree and confidence is too low to determine.";
  }
  // 5. Disagreement fallback: return label with higher confidence
  else {
    final_label = t_conf > h_conf ? t_label : h_label;
    confidence = Math.max(t_conf, h_conf);
    reason = "Models disagree: prioritized the higher confidence model.";
  }

  return {
    final_label,
    confidence: parseFloat(confidence.toFixed(3)),
    agreement,
    reason
  };
}

export function calculateRobustness(t_conf, h_conf, agreement, wordCount) {
  let baseScore = ((t_conf + h_conf) / 2) * 10;
  
  if (agreement < 0.5) baseScore -= 1.5;
  if (wordCount < 15) baseScore -= 2.0; 
  else if (wordCount > 50) baseScore += 1.0;

  const finalScore = Math.max(0, Math.min(10, baseScore));
  return Number(finalScore.toFixed(1));
}
