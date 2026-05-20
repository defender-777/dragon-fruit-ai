import { createContext, useContext } from "react";

export const AppContext = createContext(null);

export function useAppContext() {
  const ctx = useContext(AppContext);
  if (!ctx) {
    throw new Error("useAppContext must be used within AppContext.Provider");
  }
  return ctx;
}
