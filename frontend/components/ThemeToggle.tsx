"use client";

import { Moon, Sun } from "lucide-react";

import { Button } from "@/components/ui/button";
import { applyTheme } from "@/lib/theme";

export function ThemeToggle() {
  // Icon swap is pure CSS (dark: variants) driven by the .dark class already
  // set by the bootstrap script -- no React state needed, so the next theme
  // is read straight from the DOM instead of mirrored into state.
  function toggle() {
    const isDark = document.documentElement.classList.contains("dark");
    applyTheme(isDark ? "light" : "dark");
  }

  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      onClick={toggle}
      aria-label="Toggle theme"
      className="relative"
    >
      <Sun className="size-4 scale-100 rotate-0 transition-all dark:scale-0 dark:-rotate-90" />
      <Moon className="absolute size-4 scale-0 rotate-90 transition-all dark:scale-100 dark:rotate-0" />
    </Button>
  );
}
