import React from 'react';
import { motion } from 'framer-motion';
import { fadeUp, staggerContainer } from '../analyzer/motion';

const VIEWPORT = { once: true, margin: '0px 0px -10% 0px' };

/**
 * One-shot fade-up when scrolled into view. With `stagger`, direct children
 * that are `RevealItem`s animate in sequence. MotionConfig reducedMotion="user"
 * (App.jsx) drops the transform for users who prefer reduced motion.
 */
export const Reveal = ({ as = 'div', stagger = false, className = '', children, ...rest }) => {
  const Component = motion[as] ?? motion.div;
  return (
    <Component
      className={className}
      variants={stagger ? staggerContainer : fadeUp}
      initial="hidden"
      whileInView="show"
      viewport={VIEWPORT}
      {...rest}
    >
      {children}
    </Component>
  );
};

export const RevealItem = ({ as = 'div', className = '', children, ...rest }) => {
  const Component = motion[as] ?? motion.div;
  return (
    <Component className={className} variants={fadeUp} {...rest}>
      {children}
    </Component>
  );
};

export const SectionHeader = ({ id, eyebrow, title, children, align = 'center' }) => (
  <Reveal className={align === 'center' ? 'mx-auto max-w-2xl text-center' : 'max-w-2xl'}>
    <p className="micro-label">{eyebrow}</p>
    <h2
      id={id}
      className="mt-3 text-[32px] font-semibold leading-[1.1] tracking-[-0.03em] text-zinc-950 sm:text-[44px]"
    >
      {title}
    </h2>
    {children && <p className="mt-4 text-base leading-7 text-zinc-600 sm:text-[17px]">{children}</p>}
  </Reveal>
);
