import { ImagePlus, UploadCloud, X, CheckCircle2, AlertCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import MetricCard from "../components/ui/MetricCard";
import { useAppContext } from "../context/AppContext";
import { analyzeFruit, backendBaseUrl, checkBackendHealth } from "../services/api";
import { normalizeResult } from "../utils/formatters";

export default function AnalyzePage() {
  const { setLatestResult, setHistory } = useAppContext();
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [isBackendConnected, setIsBackendConnected] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const navigate = useNavigate();

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  useEffect(() => {
    let active = true;
    const checkConnection = async () => {
      try {
        await checkBackendHealth();
        if (active) setIsBackendConnected(true);
      } catch {
        if (active) setIsBackendConnected(false);
      }
    };
    checkConnection();
    return () => {
      active = false;
    };
  }, []);

  const handleFileSelect = (selected) => {
    if (!selected) return;
    if (!selected.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }
    setError("");
    setFile(selected);
  };

  const handleRemoveFile = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setFile(null);
    setError("");
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError("Please upload a fruit image before running analysis.");
      return;
    }
    setIsLoading(true);
    setError("");
    try {
      const response = await analyzeFruit(file);
      const normalized = normalizeResult({
        ...response,
        imageUrl: previewUrl,
        date: new Date().toISOString(),
      });
      setLatestResult(normalized);
      setHistory((previousHistory) => [normalized, ...previousHistory]);
      navigate("/dashboard");
    } catch (requestError) {
      setError(requestError?.response?.data?.message || "Unable to analyze image right now. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <section className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold md:text-3xl">Fruit Analysis</h1>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">Upload an image to identify ripeness and quality.</p>
        </div>
        <div className={`inline-flex items-center gap-2 rounded-xl border px-3 py-1.5 text-xs font-medium transition-colors ${
          isBackendConnected === true 
            ? "border-emerald-500/20 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" 
            : isBackendConnected === false 
              ? "border-rose-500/20 bg-rose-500/10 text-rose-600 dark:text-rose-400"
              : "border-slate-500/20 bg-slate-500/10 text-slate-500"
        }`}>
          {isBackendConnected === true ? <CheckCircle2 size={14} /> : isBackendConnected === false ? <AlertCircle size={14} /> : <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-slate-400 border-t-transparent" />}
          <span>Backend: {isBackendConnected === true ? "Online" : isBackendConnected === false ? "Offline" : "Checking..."}</span>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragging(false);
              handleFileSelect(e.dataTransfer.files?.[0]);
            }}
            className={`relative flex min-h-[320px] cursor-pointer flex-col items-center justify-center rounded-3xl border-2 border-dashed transition-all duration-200 ${
              isDragging 
                ? "border-violet-500 bg-violet-500/5 shadow-inner" 
                : "border-slate-300 bg-white/30 hover:border-violet-400 hover:bg-white/40 dark:border-slate-800 dark:bg-white/5 dark:hover:bg-white/10"
            }`}
          >
            <input
              id="fruit-upload"
              type="file"
              className="absolute inset-0 cursor-pointer opacity-0"
              accept="image/*"
              onChange={(e) => handleFileSelect(e.target.files?.[0])}
            />
            
            {previewUrl ? (
              <div className="relative group p-4">
                <img src={previewUrl} alt="Fruit preview" className="max-h-64 rounded-2xl object-contain shadow-2xl transition duration-300 group-hover:scale-[1.02]" />
                <button
                  onClick={handleRemoveFile}
                  className="absolute -right-2 -top-2 rounded-full bg-rose-500 p-1.5 text-white shadow-lg transition hover:scale-110 hover:bg-rose-600"
                  title="Remove image"
                >
                  <X size={16} />
                </button>
              </div>
            ) : (
              <div className="flex flex-col items-center p-8 text-center">
                <div className="mb-4 rounded-2xl bg-violet-500/10 p-4 text-violet-600 dark:bg-violet-500/20 dark:text-violet-400">
                  <UploadCloud size={40} />
                </div>
                <p className="text-lg font-semibold">Drop your fruit image here</p>
                <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">Supports JPG, PNG and WEBP formats</p>
                <button className="mt-6 rounded-xl bg-slate-900 px-6 py-2 text-sm font-medium text-white transition hover:bg-slate-800 dark:bg-white dark:text-slate-900 dark:hover:bg-slate-200">
                  Browse Files
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <MetricCard label="Expected Response" value="~1.2s" accent="from-emerald-500 to-teal-500" />
          <MetricCard label="Model Version" value="v2.4-stable" accent="from-blue-500 to-indigo-500" />
          
          <div className="mt-auto space-y-4 pt-4">
            {error && (
              <div className="flex items-center gap-2 rounded-xl bg-rose-500/10 p-3 text-sm text-rose-600 dark:text-rose-400">
                <AlertCircle size={16} />
                <span>{error}</span>
              </div>
            )}
            
            <button
              type="button"
              onClick={handleAnalyze}
              disabled={isLoading || !file}
              className="relative flex w-full items-center justify-center gap-3 overflow-hidden rounded-2xl bg-gradient-to-r from-violet-600 to-blue-600 px-6 py-4 text-sm font-bold text-white shadow-xl transition-all hover:scale-[1.02] active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isLoading ? (
                <>
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <ImagePlus size={20} />
                  <span>Run Analysis</span>
                </>
              )}
            </button>
            
            {isLoading && (
              <div className="overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                <div className="h-1.5 w-full animate-[progress_2s_ease-in-out_infinite] bg-gradient-to-r from-violet-500 to-blue-500" />
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
