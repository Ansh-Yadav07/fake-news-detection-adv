import React from 'react';
import { Link } from 'react-router-dom';
import { ShieldCheck } from 'lucide-react';

const Footer = () => (
  <footer className="border-t border-zinc-200/70 bg-white">
    <div className="mx-auto flex max-w-6xl flex-col gap-4 px-4 py-8 sm:flex-row sm:items-center sm:justify-between sm:px-6">
      <Link
        to="/"
        className="flex min-h-10 items-center gap-2 self-start rounded-lg text-sm font-semibold text-zinc-900 focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900/40 focus-visible:ring-offset-2 sm:self-auto"
      >
        <ShieldCheck className="h-4 w-4 text-zinc-500" aria-hidden="true" />
        News-Check
      </Link>
      <p className="text-[13px] text-zinc-500">
        © {new Date().getFullYear()} News-Check · Automated analysis; verify important claims with primary sources.
      </p>
    </div>
  </footer>
);

export default Footer;
