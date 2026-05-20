export default function ModelOutputs({ latestResult, confidence, ripeness, qualityScore, hasValidPrediction }) {
  const items = [
    { label: "Confidence", value: latestResult.confidence == null ? "N/A" : `${confidence.toFixed(1)}%` },
    { label: "Ripeness", value: latestResult.ripenessPercent == null ? "N/A" : `${ripeness.toFixed(1)}%` },
    { label: "Quality Score", value: latestResult.qualityScore == null ? "N/A" : qualityScore.toFixed(1) },
    { label: "Shelf Life", value: hasValidPrediction ? latestResult.shelfLife : "N/A" },
    { label: "Market", value: hasValidPrediction ? latestResult.market : "N/A" },
    { label: "Price Category", value: hasValidPrediction ? latestResult.priceCategory : "N/A" },
    { label: "Recommendation", value: hasValidPrediction ? latestResult.recommendation : "N/A" },
  ];

  return (
    <div className="rounded-2xl border border-white/20 bg-white/30 p-5 backdrop-blur-xl dark:border-white/10 dark:bg-white/5">
      <p className="mb-4 text-sm font-medium">Detailed Analysis</p>
      <div className="space-y-3">
        {items.map((item) => (
          <div key={item.label} className="flex items-center justify-between border-b border-white/10 pb-2 last:border-0">
            <span className="text-xs text-slate-500 dark:text-slate-400">{item.label}</span>
            <span className="text-sm font-medium text-slate-900 dark:text-white">{item.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
