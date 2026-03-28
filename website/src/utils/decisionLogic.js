export function calculateAgreement(t_conf, h_conf, t_label, h_label) {
  if (t_label === h_label) return 1.0;
  return 1.0 - Math.abs(t_conf - h_conf);
}

export function getFinalVerdict(t_label, t_conf, h_label, h_conf) {
  // Console logging inside decision function for debugging as requested
  console.log({
    transformer_label: t_label,
    transformer_confidence: t_conf,
    hybrid_label: h_label,
    hybrid_confidence: h_conf
  });

  // 1. If both models agree -> return that label
  if (t_label === h_label) {
    return t_label;
  }

  // 2. If disagreement:
  // If transformer_confidence > 0.85 AND greater than hybrid by 0.1 -> use transformer
  if (t_conf > 0.85 && (t_conf - h_conf) > 0.1) return t_label;
  // If hybrid_confidence > 0.85 AND greater than transformer by 0.1 -> use hybrid
  if (h_conf > 0.85 && (h_conf - t_conf) > 0.1) return h_label;

  // 3. If both are weak (confidence < 0.6) -> return "UNCERTAIN"
  if (t_conf < 0.6 && h_conf < 0.6) {
    return "UNCERTAIN";
  }

  // 4. Otherwise -> choose model with higher confidence
  return t_conf > h_conf ? t_label : h_label;
}

export function getWeightedFusion(t_conf, h_conf) {
  return (0.7 * t_conf) + (0.3 * h_conf);
}

export function generateExplanations(features, t_label, h_label, finalVerdict) {
  const { clickbait, uppercase, punctuation } = features;
  const exps = [];

  if (clickbait > 0.4) exps.push("High clickbait patterns detected.");
  if (uppercase > 0.15) exps.push("Excessive capitalization suggests sensational tone.");
  if (punctuation > 0.15) exps.push("Frequent punctuation indicates emotional emphasis.");
  
  if (exps.length === 0 && finalVerdict !== 'UNCERTAIN') {
    exps.push("Text appears neutral and informational.");
  }

  if (t_label !== h_label) {
    exps.push("Models show conflicting signals.");
  }

  if (finalVerdict === "UNCERTAIN") {
    exps.push("Confidence levels are insufficient for a definitive classification.");
  }

  return exps;
}
