import { Layers, Binary, ShieldAlert } from 'lucide-react';

export default function About() {
  const items = [
    { title: "Transformer Paradigm", desc: "Fine-tuned DistilBERT models catch nuanced semantic relations and contextual sentiment invisible to standard rules.", icon: Layers },
    { title: "Hybrid Augmentation", desc: "Bolstered by hardcoded linguistic parameters—measuring exaggerated spacing, stopword ratios, and uppercase intensity.", icon: Binary },
    { title: "Conflict Resolution", desc: "Identifies prediction disconnects across architectures, acting as a guard-rail to     { title: "Conflict Resolution", desc: "Identifies prediction disconnects across architectures, acting as a guard-rail to     { title: "Conflict Resolution", desc: "Identifies prediction disconnects across architectures, acting as a guard-rail to     { title: "Conflict Resolution", desc: "Identifies prediction disconnects across architectures, acting as a guard-regr    { title: "rent structural approaches.</p>
      </div>

      <div className="grid md:grid-cols-3 gap-6">
        {items.map((item, i) => (
          <div key={i} className="p-8 rounded-3xl bg-white border border-zinc-200/80 shadow-sm hover:shadow-lg hover:-translate-y-1 transition-all duration-300">
            <div className="w-12 h-12 rounded-xl bg-zinc-50 border border-zinc-200 flex items-center justify-center mb-6">
              <item.icon className="w-5 h-5 text-zinc-800" />
            </div>
            <h3 className="font-semibold text-zinc-900 mb-3">{item.title}</h3>
            <p className="text-sm text-zinc-500 leading-relaxed">{item.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
