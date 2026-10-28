import React from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { EASE_OUT } from './motion';

const VIEWBOX = 100;

/**
 * Single responsive progress ring. Geometry lives in a fixed 100x100 viewBox,
 * so on-screen size is controlled purely by Tailwind classes (`sizeClass`).
 *
 * The arc draws in once on mount and eases between values on update; it is
 * static for users who prefer reduced motion. With `unavailable` only the
 * track is drawn (never a misleading 0% arc).
 */
const Ring = ({
  value,
  label,
  color = '#18181b',
  trackColor = '#f4f4f5',
  strokeWidth = 8,
  sizeClass = 'w-16 h-16',
  unavailable = false,
  children,
}) => {
  const reduceMotion = useReducedMotion();
  const radius = (VIEWBOX - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const percentage = Math.min(100, Math.max(0, Number(value) || 0));
  const offset = circumference - (percentage / 100) * circumference;
  const center = VIEWBOX / 2;

  return (
    <div className={`relative flex-shrink-0 ${sizeClass}`}>
      <svg
        viewBox={`0 0 ${VIEWBOX} ${VIEWBOX}`}
        className="h-full w-full -rotate-90"
        role="img"
        aria-label={label}
      >
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={trackColor}
          strokeWidth={strokeWidth}
          strokeDasharray={unavailable ? '3 5' : undefined}
        />
        {!unavailable && (
          <motion.circle
            cx={center}
            cy={center}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={reduceMotion ? false : { strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={reduceMotion ? { duration: 0 } : { duration: 1.1, ease: EASE_OUT, delay: 0.2 }}
          />
        )}
      </svg>
      {children && (
        <div className="absolute inset-0 flex flex-col items-center justify-center" aria-hidden="true">
          {children}
        </div>
      )}
    </div>
  );
};

export default Ring;
