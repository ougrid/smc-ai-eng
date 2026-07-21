const THEME_KEY = "smc_theme";

export type Theme = "light" | "dark";

export function getStoredTheme(): Theme | null {
  if (typeof window === "undefined") return null;
  const stored = window.localStorage.getItem(THEME_KEY);
  return stored === "light" || stored === "dark" ? stored : null;
}

export function resolveTheme(): Theme {
  return (
    getStoredTheme() ??
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
  );
}

export function applyTheme(theme: Theme): void {
  document.documentElement.classList.toggle("dark", theme === "dark");
  window.localStorage.setItem(THEME_KEY, theme);
}

// Inlined into a blocking <script> in the root layout so the theme class is
// set before first paint -- otherwise the page flashes light before React
// hydrates and useEffect runs.
export const THEME_BOOTSTRAP_SCRIPT = `
(function () {
  try {
    var stored = window.localStorage.getItem("${THEME_KEY}");
    var dark = stored ? stored === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.classList.toggle("dark", dark);
  } catch (e) {}
})();
`;
