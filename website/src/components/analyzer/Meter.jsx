import React from 'react';
import { motion } from 'framer-motion';
import { EASE_OUT } from './motion';

/**
 * Thin horizontal meter. The fill animates with scaleX (a transform), so
 * MotionConfig reducedMotion="user" renders it at its final width instantly.
 *
 * `percent` is 0-100. When `unavailable` is true a hatched empty track is
 * rendered instead of a misleading 0% bar.
 */
const Meter = ({
  percent,
  color = '#18181b',
  label,
  valueText,
  unavailable = false,
  size = 'md',
  className = '',
}) => {
  const height = size === 'sm' ? 'h-1' : 'h-1.5';

  if (unavailable) {
    return (
      <div
        role="img"
        aria-label={`${label}: unavailable`}
        className={`${height} w-full rounded-full meter-unavailable ${className}`}
      />
    );
  }

  const pct = Math.min(100, Math.max(0, Number(percent) || 0));

  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pct)}
      aria-valuetext={valueText}
      className={`${height} w-full overflow-hidden rounded-full bg-zinc-100 ${className}`}
    >
      <motion.div
        className="h-full w-full origin-left rounded-full"
        style={{ backgroundColor: color }}
        initial={{ scaleX: 0 }}
        animate={{ scaleX: pct / 100 }}
        transition={{ duration: 0.9, ease: EASE_OUT, delay: 0.15 }}
      />
    </div>
  );
};

export default Meter;
