export default function Footer() {
  return (
    <footer className="mt-auto border-t border-zinc-200/60 bg-[#f8f9fa] pt-10 pb-10 text-center text-zinc-500 text-sm flex flex-col items-center">
      <div className="flex items-center gap-1 font-semibold text-zinc-800 mb-2 tracking-tight">
        FN-Detect Engine
      </div>
      <p>© {new Date().getFullYear()} Advanced Fake News Detection. Conceptualized for high accuracy pipelines.</p>
    </footer>
  );
}
