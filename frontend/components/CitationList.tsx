"use client";

import { useState } from "react";

import { buttonVariants } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";

type SqlRow = Record<string, unknown>;

type Chunk = {
  id: string;
  company: string;
  source: string | null;
  page: number | null;
  text: string;
  score: number;
};

// Scores aren't a bounded 0-1 confidence in every mode (a hybrid/RRF-fused
// score can be as small as ~0.03), so relevance bars are normalized relative
// to the other chunks in this same citation list rather than read as an
// absolute percentage.
function relevancePercents(chunks: Chunk[]): number[] {
  const scores = chunks.map((c) => c.score);
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  if (max === min) return scores.map(() => 100);
  return scores.map((s) => Math.round(((s - min) / (max - min)) * 100));
}

export function CitationList({ sqlRows, chunks }: { sqlRows?: SqlRow[]; chunks?: Chunk[] }) {
  const [open, setOpen] = useState(false);
  const hasSql = Boolean(sqlRows && sqlRows.length > 0);
  const hasChunks = Boolean(chunks && chunks.length > 0);
  if (!hasSql && !hasChunks) return null;

  const columns = hasSql ? Object.keys(sqlRows![0]) : [];
  const count = (sqlRows?.length ?? 0) + (chunks?.length ?? 0);
  const relevance = hasChunks ? relevancePercents(chunks!) : [];

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="mt-2">
      <CollapsibleTrigger className={cn(buttonVariants({ variant: "ghost", size: "xs" }), "text-xs")}>
        {open ? "Hide sources" : "Show sources"} ({count})
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-2 space-y-2 text-xs">
        {hasSql && (
          <Table>
            <TableHeader>
              <TableRow>
                {columns.map((column) => (
                  <TableHead key={column}>{column}</TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {sqlRows!.map((row, i) => (
                <TableRow key={i}>
                  {columns.map((column) => (
                    <TableCell key={column}>{String(row[column])}</TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
        {hasChunks && (
          <ul className="space-y-1.5">
            {chunks!.map((chunk, i) => (
              <li key={chunk.id} className="rounded border p-2">
                <div className="flex items-center justify-between gap-2">
                  <div className="font-medium">
                    {chunk.company} — {chunk.source ?? "unknown source"}
                    {chunk.page != null ? `, p.${chunk.page}` : ""}
                  </div>
                  <div
                    className="flex shrink-0 items-center gap-1"
                    title={`Relevance score: ${chunk.score.toFixed(4)}`}
                  >
                    <div className="h-1 w-10 overflow-hidden rounded-full bg-border">
                      <div
                        className="h-full rounded-full bg-primary"
                        style={{ width: `${relevance[i]}%` }}
                      />
                    </div>
                    <span className="text-[0.7rem] text-muted-foreground">{relevance[i]}%</span>
                  </div>
                </div>
                <p className="line-clamp-2 text-muted-foreground">{chunk.text}</p>
              </li>
            ))}
          </ul>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
