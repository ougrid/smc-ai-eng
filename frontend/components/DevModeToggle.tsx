"use client";

import { Bug } from "lucide-react";

import { Button } from "@/components/ui/button";
import { useDevMode } from "@/hooks/use-dev-mode";
import { setDevMode } from "@/lib/devmode";

export function DevModeToggle() {
  const enabled = useDevMode();

  return (
    <Button
      type="button"
      variant={enabled ? "secondary" : "ghost"}
      size="icon"
      onClick={() => setDevMode(!enabled)}
      aria-label="Toggle debug mode"
      aria-pressed={enabled}
      title={enabled ? "Debug mode on — showing pipeline internals" : "Debug mode off"}
      className="relative"
    >
      <Bug className={`size-4 ${enabled ? "text-foreground" : "text-muted-foreground"}`} />
    </Button>
  );
}
