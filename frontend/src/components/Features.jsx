export default function Features() {
  return (
    <section className="bg-zinc-900 text-white rounded-3xl p-10 md:p-16 flex flex-col md:flex-row gap-12 items-center relative overflow-hidden">
      
      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decorative gradient blur inne      {/* Decoratime=      {/* Decorat        {[
              { label: "Original Input", score: "REAL (92%)" },
              { label: "Uppercased Format", score: "REAL (89%)" },
              { label: "Punctuation Padded", score: "REAL (91%)" }
            ].map((stat, idx) => (
                <div key={idx} className="bg-white/10 backdrop-blur-md p-5 rounded-2xl border border-white/10 flex items-center justify-between hover:bg-white/15 transition-colors">
                   <div className="flex items-center gap-3">
                     <span className="w-2 h-2 rounded-full bg-zinc-300"></span>
                     <span className="text-sm font-medium text-zinc-200">{stat.label}</span>
                   </div>
                   <span className="text-xs font-bold font-mono tracking-widest text-zinc-400">{stat.score}</span>
                </div>
            ))}
          </div>
      </div>
    </section>
  );
}
