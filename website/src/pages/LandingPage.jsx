import React from 'react';
import Hero from '../components/Hero';
import About from '../components/About';
import Features from '../components/Features';

const LandingPage = () => {
  return (
    <div className="animate-fade-in" style={{ animationTimingFunction: 'cubic-bezier(0.4, 0, 0.2, 1)' }}>
      <Hero />
      <About />
      <Features />
    </div>
  );
};

export default LandingPage;
