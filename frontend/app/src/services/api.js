import axios from "axios";

const fallbackBaseUrl = "https://dragon-fruit-ai.onrender.com";
const devProxyBaseUrl = "/api";
export const backendBaseUrl =
  import.meta.env.VITE_API_BASE_URL || (import.meta.env.DEV ? devProxyBaseUrl : fallbackBaseUrl);

const api = axios.create({
  baseURL: backendBaseUrl,
});

export async function analyzeFruit(file) {
  const formData = new FormData();
  formData.append("file", file);

  const { data } = await api.post("/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return data;
}

export async function fetchHistory() {
  try {
    const { data } = await api.get("/history");
    return Array.isArray(data) ? data : data?.records || [];
  } catch (error) {
    if (error?.response?.status && error.response.status !== 404) {
      throw error;
    }
    return [];
  }
}

export async function checkBackendHealth() {
  const { data } = await api.get("/");
  return data;
}
