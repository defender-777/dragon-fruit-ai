const statusColors = {
  success: "#10b981",
  invalid: "#f59e0b",
  error: "#ef4444",
  unknown: "#64748b",
};

export default function StatusHistory({ status, message, stageItems }) {
  const statusKey = String(status || "unknown").toLowerCase();

  return (
    <div className="rounded-2xl border border-white/20 bg-white/30 p-5 backdrop-blur-xl dark:border-white/10 dark:bg-white/5">
      <p className="mb-4 text-sm font-medium">Status & Trends</p>
      <div className="inline-flex items-center gap-2 rounded-xl border border-white/20 bg-white/20 px-3 py-1.5 text-xs font-medium dark:border-white/10">
        <span
          className="h-2 w-2 animate-pulse rounded-full"
          style={{ backgroundColor: statusColors[statusKey] || statusColors.unknown }}
        />
        {status || "Unknown"}
      </div>

      {message && (
        <div className="mt-4 rounded-xl border border-amber-500/20 bg-amber-500/5 p-3 text-xs text-amber-700 dark:text-amber-300">
          {message}
        </div>
      )}

      <div className="mt-6 space-y-3">
        <p className="text-[10px] font-bold uppercase tracking-wider text-slate-500">Recent Distribution</p>
        <div className="grid grid-cols-2 gap-2">
          {stageItems.length === 0 ? (
            <p className="col-span-2 text-xs text-slate-500">No history data available.</p>
          ) : (
            stageItems.map(([stage, count]) => (
              <div key={stage} className="rounded-lg bg-white/10 p-2 text-center">
                <p className="text-xs font-semibold">{count}</p>
                <p className="text-[10px] text-slate-500 truncate">{stage}</p>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
