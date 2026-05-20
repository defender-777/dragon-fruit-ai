export default function PredictionProgress({ hasValidPrediction, ripeness, stage, message }) {
  return (
    <div className="rounded-2xl border border-white/20 bg-white/30 p-5 backdrop-blur-xl dark:border-white/10 dark:bg-white/5">
      <p className="text-sm font-medium">{hasValidPrediction ? "Ripeness Progress" : "Prediction Status"}</p>
      {hasValidPrediction ? (
        <div
          className="mx-auto mt-4 flex h-40 w-40 items-center justify-center rounded-full transition-all duration-700"
          style={{
            background: `conic-gradient(#8b5cf6 ${ripeness * 3.6}deg, rgba(148,163,184,0.15) 0deg)`,
          }}
        >
          <div className="flex h-32 w-32 items-center justify-center rounded-full bg-white/90 shadow-inner dark:bg-slate-900/90">
            <div className="text-center">
              <p className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
                {ripeness.toFixed(0)}%
              </p>
              <p className="text-[10px] uppercase tracking-wider text-slate-500">Ripeness</p>
            </div>
          </div>
        </div>
      ) : (
        <div className="mt-6 flex min-h-40 items-center justify-center rounded-2xl border border-amber-500/20 bg-amber-500/5 p-4 text-center text-sm text-amber-700 dark:text-amber-300">
          {message || "Input not recognized as dragon fruit."}
        </div>
      )}
      <p className="mt-4 text-center text-sm font-medium text-slate-600 dark:text-slate-300">
        Stage: <span className="text-slate-900 dark:text-white">{hasValidPrediction ? stage || "Unknown" : "N/A"}</span>
      </p>
    </div>
  );
}
