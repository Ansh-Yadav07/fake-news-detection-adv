import React from 'react';
import { ShieldCheck } from 'lucide-react';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass-card mx-4 mt-4 px-6 md:px-8 py-4 flex items-center justify-between bg-white/70 backdrop-blur-md rounded-2xl border border-black/5 shadow-sm">
      <Link to="/" className="flex items-center gap-2 group">
        <ShieldCheck className="w-6 h-6 text-zinc-900 transition-transform group-hover:scale-110 duration-300" />
        <span className="font-bold tracking-tight text-lg text-zinc-900">News-Check</span>
      </Link>

      <div className="flex items-center gap-6 md:gap-8 text-sm font-semibold text-zinc-500">
        <Link to="/" className="relative text-zinc-500 hover:text-zinc-900 transition-colors duration-300 after:content-[''] after:absolute after:-bottom-1 after:left-0 after:w-0 after:h-0.5 after:bg-zinc-900 after:transition-all after:duration-300 hover:after:w-full">
          Home
        </Link>
        <Link to="/analyze" className="relative text-zinc-500 hover:text-zinc-900 transition-colors duration-300 after:content-[''] after:absolute after:-bottom-1 after:left-0 after:w-0 after:h-0.5 after:bg-zinc-900 after:transition-all after:duration-300 hover:after:w-full">
          Analyzer
        </Link>
      </div>
    </nav>
  );
};

export default Navbar;
