import { ChevronDown, ChevronUp, Code } from "lucide-react";
import { useState } from "react";

export default function RawResponse({ rawResponse }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="overflow-hidden rounded-2xl border border-white/20 bg-slate-900 shadow-lg dark:border-white/10">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between p-4 text-left transition hover:bg-white/5"
      >
        <div className="flex items-center gap-2 text-sm font-medium text-slate-200">
          <Code size={16} />
          <span>Backend Response (Raw JSON)</span>
        </div>
        {isOpen ? <ChevronUp size={16} className="text-slate-400" /> : <ChevronDown size={16} className="text-slate-400" />}
      </button>

      {isOpen && (
        <div className="border-t border-white/10 bg-black/20 p-4">
          <pre className="overflow-x-auto text-xs text-slate-300 md:text-sm">
            {JSON.stringify(rawResponse || {}, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
