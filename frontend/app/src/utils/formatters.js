function normalizePercent(value) {
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return 0;
  if (numeric > 100) return 100;
  if (numeric < 0) return 0;
  return Math.round(numeric * 100) / 100;
}

export function normalizeResult(result = {}) {
  const fruitType =
    result.fruitType ||
    result.fruit_type ||
    result.type ||
    result.prediction ||
    result.class_name ||
    "Dragon Fruit";
  const stageValue = result.stage || result.freshness || result.result || result.label || "Unknown";
  const stage = String(stageValue);
  const status = String(result.status || "unknown").toLowerCase();
  const isValidPrediction = status === "success";
  const confidenceRaw = Number(result.confidence ?? result.score ?? 0);
  const confidence = confidenceRaw > 1 ? confidenceRaw : confidenceRaw * 100;
  const ripenessFromApi = result.ripeness_percent ?? result.ripenessPercent;
  const qualityFromApi = result.quality_score ?? result.qualityScore;
  const ripenessPercent = ripenessFromApi == null ? null : normalizePercent(ripenessFromApi);
  const qualityScore = qualityFromApi == null ? null : normalizePercent(qualityFromApi);

  return {
    id: result.id || crypto.randomUUID(),
    fruitType,
    freshness: stage,
    stage,
    status,
    isValidPrediction,
    confidence: Number.isNaN(confidence) ? null : normalizePercent(confidence),
    ripenessPercent,
    shelfLife: result.shelf_life || result.shelfLife || "N/A",
    qualityScore,
    grade: result.grade || "N/A",
    market: result.market || "N/A",
    priceCategory: result.price_category || result.priceCategory || "N/A",
    recommendation: result.recommendation || "N/A",
    imageUrl: result.imageUrl || result.image_url || result.image || "",
    message: result.message || (isValidPrediction ? "" : "Input not recognized as dragon fruit."),
    rawResponse: result,
    date: result.date || result.createdAt || new Date().toISOString(),
  };
}

export function formatDate(dateValue) {
  const date = new Date(dateValue);
  if (Number.isNaN(date.getTime())) return "Unknown";
  return date.toLocaleString();
}
