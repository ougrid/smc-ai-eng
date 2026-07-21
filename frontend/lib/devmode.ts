const DEVMODE_KEY = "smc_devmode";
const DEVMODE_EVENT = "smc-devmode-change";

// Dev/debug mode reveals the agent's internal pipeline data (routing, coverage
// gate, executed SQL, Python-computed figures, verification, retrieval scores)
// under each answer -- a demo affordance for showing the anti-hallucination
// machinery. Persisted in localStorage and broadcast via a custom event so the
// header toggle and every message panel (siblings in the tree) stay in sync
// without prop-drilling.

export function getDevMode(): boolean {
  if (typeof window === "undefined") return false;
  return window.localStorage.getItem(DEVMODE_KEY) === "on";
}

export function setDevMode(on: boolean): void {
  window.localStorage.setItem(DEVMODE_KEY, on ? "on" : "off");
  window.dispatchEvent(new Event(DEVMODE_EVENT));
}

export function subscribeDevMode(callback: () => void): () => void {
  window.addEventListener(DEVMODE_EVENT, callback);
  // cross-tab: another tab writing localStorage fires a native storage event
  window.addEventListener("storage", callback);
  return () => {
    window.removeEventListener(DEVMODE_EVENT, callback);
    window.removeEventListener("storage", callback);
  };
}
