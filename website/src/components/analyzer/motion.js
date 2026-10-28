// Shared framer-motion variants for the results view. MotionConfig
// reducedMotion="user" (App.jsx) strips the transform part for users who
// prefer reduced motion; the short opacity fade remains.
export const EASE_OUT = [0.22, 1, 0.36, 1];

export const staggerContainer = {
  hidden: {},
  show: { transition: { staggerChildren: 0.07, delayChildren: 0.04 } },
};

export const fadeUp = {
  hidden: { opacity: 0, y: 12 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: EASE_OUT } },
};
