import { ShieldCheck } from 'lucide-react';

export default function Navbar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-40 bg-white/60 backdrop-blur-xl border-b border-zinc-200/50">
      <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-2 font-bold tracking-tight text-zinc-900">
          <ShieldCheck className="w-5 h-5 text-zinc-800" />
          FN-Detect
        </div>
        <div className="flex gap-6 items-center">
          <a href="#about" className="text-sm font-medium text-zinc-500 ho          <a href="#about" className="text-sm font-medium text-zinc-500 ho          <a href="#about" className="text-sm font-medium text-zinc-500 ho          <a href="#about" className="text-sm font-medium text-zinc-500 ho          <a href="#about" className="text-sm font-medium text-zinc-500 ho          <a href="#about" className="text-sm font-medium text-zinc-500 and          <a h=> {
    const el = document.getElementById('demo-section');
    if (el) {
      const yUrl = el.getBoundingClientRect().top + window.scrollY - 100;
      window.scrollTo({ top: yUrl, behavior: 'smooth' });
    }
  };

  return (
    <section className="flex flex-col items-center justify-center text-center space-y-6 pt-10 animate-fade-in-up">
      <div className="inline-flex items-center rounded-full border border-zinc-200 bg-zinc-50 px-3 py-1 text-xs font-semibold text-zinc-600">
        <span className="flex h-2 w-2 rounded-full bg-zinc-800 mr-2 opacity-80 animate-pulse"></span>
        Model v2.0 Live
      </div>
      <h1 className="text-5xl md:text-6xl font-bold tracking-tight text-zinc-900 drop-shadow-sm">
        AI-Powered Fake News <br className="hidden md:block"/> Detection Engine
      </h1>
      <p className="max-w-2xl text-lg text-zinc-500 leading-relaxed">
        Verify information integrity instantly. Engineered with a dual Transformer + Hybrid ML architecture to catch subtle misinformation and linguistic exaggeration.
      </p>
      <button 
        onClick={handleScroll}
        className="mt-6 px-7 py-3.5 bg-zinc-900 text-white rounded-xl text-sm font-medium tracking-wide hover:bg-zinc-800 hover:-translate-y-0.5 transition-all shadow-lg shadow-zinc-200 active:translate-y-0"
      >
        Try the Demo
      </button>
    </section>
  );
}
