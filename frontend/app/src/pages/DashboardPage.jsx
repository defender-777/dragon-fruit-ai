import MetricCard from "../components/ui/MetricCard";
import { useAppContext } from "../context/AppContext";
import DashboardEmptyState from "../components/dashboard/DashboardEmptyState";
import MetricsChart from "../components/dashboard/MetricsChart";
import PredictionProgress from "../components/dashboard/PredictionProgress";
import ModelOutputs from "../components/dashboard/ModelOutputs";
import StatusHistory from "../components/dashboard/StatusHistory";
import RawResponse from "../components/dashboard/RawResponse";

export default function DashboardPage() {
  const { latestResult, history } = useAppContext();

  if (!latestResult) {
    return <DashboardEmptyState />;
  }

  const hasValidPrediction = latestResult.isValidPrediction;
  const confidence = Math.max(0, Math.min(100, latestResult.confidence ?? 0));
  const ripeness = Math.max(0, Math.min(100, latestResult.ripenessPercent ?? 0));
  const qualityScore = Math.max(0, Math.min(100, latestResult.qualityScore ?? 0));

  const bars = hasValidPrediction
    ? [
        { metric: "Ripeness", value: ripeness },
        { metric: "Confidence", value: confidence },
        { metric: "Quality", value: qualityScore },
      ]
    : [{ metric: "Confidence", value: confidence }];

  const stageBreakdown = history.reduce((acc, item) => {
    const key = String(item.stage || item.freshness || "Unknown");
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});

  const stageItems = Object.entries(stageBreakdown).slice(0, 4);

  return (
    <section className="space-y-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold tracking-tight md:text-3xl">Results Dashboard</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Live insights from your latest fruit scan.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <MetricCard label="Fruit Type" value={latestResult.fruitType || "Unknown"} />
        <MetricCard 
          label="Stage" 
          value={hasValidPrediction ? latestResult.stage || "Unknown" : "Not Available"} 
          accent="from-violet-500 to-fuchsia-500" 
        />
        <MetricCard 
          label="Grade" 
          value={hasValidPrediction ? latestResult.grade || "N/A" : "Not Available"} 
          accent="from-cyan-500 to-blue-600" 
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <PredictionProgress 
          hasValidPrediction={hasValidPrediction}
          ripeness={ripeness}
          stage={latestResult.stage}
          message={latestResult.message}
        />
        <MetricsChart data={bars} />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <ModelOutputs 
          latestResult={latestResult}
          confidence={confidence}
          ripeness={ripeness}
          qualityScore={qualityScore}
          hasValidPrediction={hasValidPrediction}
        />
        <StatusHistory 
          status={latestResult.status}
          message={latestResult.message}
          stageItems={stageItems}
        />
      </div>

      <RawResponse rawResponse={latestResult.rawResponse} />
    </section>
  );
}
