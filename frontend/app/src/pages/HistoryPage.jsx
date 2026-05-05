import { Search, Trash2, Calendar, MoreVertical } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useAppContext } from "../context/AppContext";
import { fetchHistory } from "../services/api";
import { formatDate, normalizeResult } from "../utils/formatters";

export default function HistoryPage() {
  const { history, setHistory } = useAppContext();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("All");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (history.length > 0) return;
    const load = async () => {
      setIsLoading(true);
      setError("");
      try {
        const records = await fetchHistory();
        setHistory(records.map((record) => normalizeResult(record)));
      } catch {
        setError("Failed to load analysis history.");
      } finally {
        setIsLoading(false);
      }
    };
    load();
  }, [history.length, setHistory]);

  const filteredHistory = useMemo(() => {
    const lower = query.trim().toLowerCase();
    return history.filter((record) => {
      const matchesSearch =
        !lower ||
        record.fruitType.toLowerCase().includes(lower) ||
        (record.stage || record.freshness || "").toLowerCase().includes(lower);
      const matchesFilter = filter === "All" || (record.stage || record.freshness) === filter;
      return matchesSearch && matchesFilter;
    });
  }, [filter, history, query]);

  const stageOptions = useMemo(() => {
    const values = Array.from(new Set(history.map((item) => item.stage || item.freshness).filter(Boolean)));
    return ["All", ...values];
  }, [history]);

  const handleClearHistory = () => {
    if (window.confirm("Are you sure you want to clear all history? This cannot be undone.")) {
      setHistory([]);
      localStorage.removeItem("dragon-fruit-ai.history");
    }
  };

  return (
    <section className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold md:text-3xl">Scan History</h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">Review and manage your previous fruit analyses.</p>
        </div>
        {history.length > 0 && (
          <button
            onClick={handleClearHistory}
            className="flex items-center gap-2 rounded-xl border border-rose-500/20 bg-rose-500/10 px-4 py-2 text-sm font-medium text-rose-600 transition hover:bg-rose-500/20 dark:text-rose-400"
          >
            <Trash2 size={16} />
            <span>Clear History</span>
          </button>
        )}
      </div>

      <div className="flex flex-col gap-3 rounded-2xl border border-white/20 bg-white/30 p-4 backdrop-blur-xl dark:border-white/10 dark:bg-white/5 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search scans..."
            className="w-full rounded-xl border border-white/10 bg-white/40 py-2.5 pl-10 pr-4 text-sm outline-none transition focus:border-violet-500/50 dark:bg-white/5"
          />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-slate-500">Filter:</span>
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="rounded-xl border border-white/10 bg-white/40 px-3 py-2.5 text-sm outline-none dark:bg-white/5"
          >
            {stageOptions.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {isLoading ? (
          <div className="col-span-full py-12 text-center text-sm text-slate-500">Loading history...</div>
        ) : error ? (
          <div className="col-span-full py-12 text-center text-sm text-rose-500">{error}</div>
        ) : filteredHistory.length === 0 ? (
          <div className="col-span-full rounded-3xl border border-dashed border-slate-300 py-16 text-center dark:border-slate-800">
            <p className="text-sm text-slate-500">No records found matching your criteria.</p>
          </div>
        ) : (
          filteredHistory.map((row) => (
            <div 
              key={row.id} 
              className="group relative overflow-hidden rounded-2xl border border-white/20 bg-white/30 p-4 transition-all hover:border-white/40 hover:shadow-lg dark:border-white/10 dark:bg-white/5"
            >
              <div className="flex gap-4">
                <div className="h-16 w-16 shrink-0 overflow-hidden rounded-xl bg-slate-200 dark:bg-slate-800">
                  {row.imageUrl ? (
                    <img src={row.imageUrl} alt={row.fruitType} className="h-full w-full object-cover transition duration-300 group-hover:scale-110" />
                  ) : (
                    <div className="flex h-full w-full items-center justify-center">
                      <Calendar size={20} className="text-slate-400" />
                    </div>
                  )}
                </div>
                <div className="flex flex-col justify-between overflow-hidden">
                  <div>
                    <h3 className="truncate font-semibold text-slate-900 dark:text-white">{row.fruitType}</h3>
                    <p className="text-[10px] text-slate-500">{formatDate(row.date)}</p>
                  </div>
                  <div className="mt-2 flex items-center gap-2">
                    <span
                      className={`rounded-lg px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${
                        (row.stage || row.freshness) === "Mature"
                          ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                          : (row.stage || row.freshness) === "Not Detected"
                            ? "bg-amber-500/10 text-amber-600 dark:text-amber-400"
                            : "bg-violet-500/10 text-violet-600 dark:text-violet-400"
                      }`}
                    >
                      {row.stage || row.freshness}
                    </span>
                  </div>
                </div>
              </div>
              <div className="mt-4 grid grid-cols-2 gap-2 border-t border-white/10 pt-4">
                <div>
                  <p className="text-[10px] text-slate-500">Confidence</p>
                  <p className="text-xs font-semibold">{row.confidence == null ? "N/A" : `${row.confidence}%`}</p>
                </div>
                <div>
                  <p className="text-[10px] text-slate-500">Quality Score</p>
                  <p className="text-xs font-semibold">{row.qualityScore == null ? "N/A" : row.qualityScore}</p>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
