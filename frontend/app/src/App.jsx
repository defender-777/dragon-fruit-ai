import { useEffect, useMemo, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import AppLayout from "./components/layout/AppLayout";
import { AppContext } from "./context/AppContext";
import AnalyzePage from "./pages/AnalyzePage";
import DashboardPage from "./pages/DashboardPage";
import HistoryPage from "./pages/HistoryPage";
import LandingPage from "./pages/LandingPage";

const STORAGE_KEYS = {
  latestResult: "dragon-fruit-ai.latest-result",
  history: "dragon-fruit-ai.history",
};

function readStoredJson(key, fallbackValue) {
  try {
    const rawValue = localStorage.getItem(key);
    return rawValue ? JSON.parse(rawValue) : fallbackValue;
  } catch {
    return fallbackValue;
  }
}

function App() {
  const [latestResult, setLatestResult] = useState(() => readStoredJson(STORAGE_KEYS.latestResult, null));
  const [history, setHistory] = useState(() => readStoredJson(STORAGE_KEYS.history, []));

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.latestResult, JSON.stringify(latestResult));
  }, [latestResult]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.history, JSON.stringify(history));
  }, [history]);

  const contextValue = useMemo(
    () => ({ latestResult, setLatestResult, history, setHistory }),
    [latestResult, history]
  );

  return (
    <AppContext.Provider value={contextValue}>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route element={<AppLayout />}>
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/analyze" element={<AnalyzePage />} />
          <Route path="/history" element={<HistoryPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppContext.Provider>
  );
}

export default App;