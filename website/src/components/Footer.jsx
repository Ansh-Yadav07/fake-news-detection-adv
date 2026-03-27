import React from 'react';

const Footer = () => {
  return (
    <footer className="py-8 text-center border-t border-black/5 mt-10">
      <p className="text-sm text-zinc-400 font-medium">
        © {new Date().getFullYear()} Fake News Detection Engine
      </p>
    </footer>
  );
};

export default Footer;