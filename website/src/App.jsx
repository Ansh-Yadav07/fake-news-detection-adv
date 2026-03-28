import React, { useEffect } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import LandingPage from './pages/LandingPage';
import AnalysisPage from './pages/AnalysisPage';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

function AnimatedRoutes() {
  const location = useLocation();
  
  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route 
          path="/" 
          element={
            <motion.div
              initial={{ x: '-10%', opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: '-10%', opacity: 0 }}
              transition={{ duration: 0.4, ease: "easeInOut" }}
            >
              <LandingPage />
            </motion.div>
          } 
        />
        <Route 
          path="/analyze" 
          element={
            <motion.div
              initial={{ x: '10%', opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: '10%', opacity: 0 }}
              transition={{ duration: 0.4, ease: "easeInOut" }}
            >
              <AnalysisPage />
            </motion.div>
          } 
        />
      </Routes>
    </AnimatePresence>
  );
}

function App() {
  return (
    <BrowserRouter>
      <ScrollToTop />
      <div className="min-h-screen flex flex-col font-sans text-zinc-900 bg-[#f5f5f6]">
        <Navbar />
        <main className="flex-grow pt-[5rem] relative overflow-hidden">
          <AnimatedRoutes />
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  );
}

export default App;