import React from 'react';
import { ShieldCheck } from 'lucide-react';

const Navbar = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass-card mx-4 mt-4 px-6 py-4 flex justify-between items-center bg-white/70 backdrop-blur-md rounded-2xl border border-black/5">
      <div className="flex items-center gap-2">
        <ShieldCheck className="w-6 h-6 text-zinc-900" />
        <span className="font-semibold tracking-tight text-lg">Veritas</span>
      </div>
      <div className="hidden md:flex items-center gap-8 text-sm font-medium text-zinc-500">
        <a href="#about" className="hover:text-zinc-900 transition-colors">About</a>
        <a href="#features" className="hover:text-zinc-900 transition-colors">Features</a>
        <a href="#demo" className="hover:text-zinc-900 transition-colors">Try Demo</a>
      </div>
      <a href="#demo" className="px-4 py-2 bg-zinc-900 text-white rounded-xl text-sm font-medium hover:bg-zinc-800 transition-colors">
        Analyze
      </a>
    </nav>
  );
};

export default Navbar;
