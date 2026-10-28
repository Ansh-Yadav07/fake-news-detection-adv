import React from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import { ArrowRight, ShieldCheck } from 'lucide-react';

const FOCUS = 'focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900/40 focus-visible:ring-offset-2';
const LINK = `inline-flex min-h-10 items-center rounded-lg px-3 text-sm font-medium text-zinc-600 transition-colors hover:text-zinc-950 ${FOCUS}`;

const SECTIONS = [
  { hash: 'how-it-works', label: 'How it works' },
  { hash: 'features', label: 'Features' },
];

/** In-page anchor on the landing page; router link back to it elsewhere. */
const SectionLink = ({ onHome, hash, children }) =>
  onHome ? (
    <a href={`#${hash}`} className={LINK}>
      {children}
    </a>
  ) : (
    <Link to={`/#${hash}`} className={LINK}>
      {children}
    </Link>
  );

const Navbar = () => {
  const { pathname } = useLocation();
  const onHome = pathname === '/';

  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-zinc-200/70 bg-white/80 backdrop-blur-md">
      <nav aria-label="Primary" className="mx-auto flex h-16 max-w-6xl items-center justify-between gap-3 px-4 sm:px-6">
        <Link to="/" className={`-ml-1 flex min-h-10 items-center gap-2.5 rounded-lg px-1 ${FOCUS}`}>
          <span className="flex h-8 w-8 items-center justify-center rounded-[10px] bg-zinc-900 text-white shadow-card" aria-hidden="true">
            <ShieldCheck className="h-[18px] w-[18px]" />
          </span>
          <span className="text-[17px] font-semibold tracking-[-0.02em] text-zinc-950">News-Check</span>
        </Link>

        <div className="flex items-center gap-1">
          <ul className="hidden items-center gap-1 md:flex">
            {SECTIONS.map((s) => (
              <li key={s.hash}>
                <SectionLink onHome={onHome} hash={s.hash}>
                  {s.label}
                </SectionLink>
              </li>
            ))}
            <li>
              <NavLink
                to="/analyze"
                className={({ isActive }) => (isActive ? LINK.replace('text-zinc-600', 'text-zinc-950') : LINK)}
              >
                Analyzer
              </NavLink>
            </li>
          </ul>

          {onHome ? (
            <Link
              to="/analyze"
              className={`ml-2 inline-flex min-h-10 items-center gap-1.5 rounded-xl bg-zinc-900 px-3.5 text-sm font-medium text-white shadow-card transition-colors hover:bg-zinc-800 ${FOCUS}`}
            >
              <span className="hidden sm:inline">Try the analyzer</span>
              <span className="sm:hidden">Analyzer</span>
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </Link>
          ) : (
            <Link to="/" className={`${LINK} md:hidden`}>
              Home
            </Link>
          )}
        </div>
      </nav>
    </header>
  );
};

export default Navbar;
