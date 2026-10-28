import React, { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import Hero from '../components/Hero';
import About from '../components/About';
import Features from '../components/Features';
import VerdictScale from '../components/landing/VerdictScale';
import CtaBand from '../components/landing/CtaBand';

const LandingPage = () => {
  const { hash } = useLocation();

  // Arriving from another route via /#section: scroll to it once mounted
  // (App's ScrollToTop resets to the top on pathname change).
  useEffect(() => {
    if (!hash) return;
    const el = document.getElementById(decodeURIComponent(hash.slice(1)));
    el?.scrollIntoView({ block: 'start' });
  }, [hash]);

  return (
    <div>
      <Hero />
      <About />
      <Features />
      <VerdictScale />
      <CtaBand />
    </div>
  );
};

export default LandingPage;
