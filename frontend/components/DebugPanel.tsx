"use client";

import { useState } from "react";

import { buttonVariants } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";

export type DebugData = {
  route_decision?: Record<string, unknown>;
  gate_result?: Record<string, unknown>;
  sql?: string | null;
  computed?: Record<string, unknown>;
  verify?: { ok?: boolean; ungrounded?: string[]; dangling_citations?: string[]; attempt?: number };
  rejected_chunks?: { id?: string; company?: string; score?: number; reason?: string }[];
};

function isEmpty(obj: unknown): boolean {
  return !obj || (typeof obj === "object" && Object.keys(obj as object).length === 0);
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <div className="font-semibold text-foreground">{title}</div>
      {children}
    </div>
  );
}

function Json({ value }: { value: unknown }) {
  return (
    <pre className="max-h-56 overflow-auto rounded border bg-background p-2 font-mono text-[0.7rem] leading-relaxed">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

export function DebugPanel({ debug }: { debug?: DebugData }) {
  const [open, setOpen] = useState(false);
  if (!debug || Object.keys(debug).length === 0) return null;

  const verify = debug.verify;
  const rejected = debug.rejected_chunks ?? [];

  // Group dropped chunks by why they were dropped (below score floor,
  // boilerplate, duplicate, reranked out) for an at-a-glance retrieval story.
  const rejectedByReason = rejected.reduce<Record<string, number>>((acc, c) => {
    const reason = c.reason ?? "unknown";
    acc[reason] = (acc[reason] ?? 0) + 1;
    return acc;
  }, {});

  const summary = [
    verify ? `verify ${verify.ok ? "✓" : "✗"}` : null,
    debug.sql ? "SQL" : null,
    rejected.length ? `${rejected.length} dropped` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="mt-2 w-full">
      <CollapsibleTrigger
        className={cn(buttonVariants({ variant: "ghost", size: "xs" }), "text-xs font-mono")}
      >
        {open ? "▾" : "▸"} debug{summary ? ` — ${summary}` : ""}
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-2 space-y-3 rounded-lg border bg-muted/40 p-3 text-xs">
        {verify && (
          <Section title="Verification (deterministic, no LLM)">
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
              <span>
                grounded:{" "}
                <span className={verify.ok ? "text-emerald-600 dark:text-emerald-400" : "text-destructive"}>
                  {verify.ok ? "pass" : "fail"}
                </span>
              </span>
              {typeof verify.attempt === "number" && <span>attempt: {verify.attempt}</span>}
            </div>
            {verify.ungrounded && verify.ungrounded.length > 0 && (
              <div className="text-destructive">
                ungrounded numbers: {verify.ungrounded.join(", ")}
              </div>
            )}
            {verify.dangling_citations && verify.dangling_citations.length > 0 && (
              <div className="text-destructive">
                dangling citations: {verify.dangling_citations.join(", ")}
              </div>
            )}
          </Section>
        )}

        {debug.sql && (
          <Section title="Executed SQL (allowlisted, read-only role)">
            <pre className="overflow-auto rounded border bg-background p-2 font-mono text-[0.7rem]">
              {debug.sql}
            </pre>
          </Section>
        )}

        {!isEmpty(debug.computed) && (
          <Section title="Python-computed figures (not LLM-generated)">
            <Json value={debug.computed} />
          </Section>
        )}

        {rejected.length > 0 && (
          <Section title="Retrieval — dropped chunks">
            <div className="flex flex-wrap gap-x-4 gap-y-1">
              {Object.entries(rejectedByReason).map(([reason, n]) => (
                <span key={reason}>
                  {reason}: <span className="text-muted-foreground">{n}</span>
                </span>
              ))}
            </div>
          </Section>
        )}

        {!isEmpty(debug.route_decision) && (
          <Section title="Router decision (intent · entity resolution · route)">
            <Json value={debug.route_decision} />
          </Section>
        )}

        {!isEmpty(debug.gate_result) && (
          <Section title="Coverage gate (deterministic)">
            <Json value={debug.gate_result} />
          </Section>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
