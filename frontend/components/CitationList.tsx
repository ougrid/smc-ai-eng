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

export function CitationList({ sqlRows, chunks }: { sqlRows?: SqlRow[]; chunks?: Chunk[] }) {
  const [open, setOpen] = useState(false);
  const hasSql = Boolean(sqlRows && sqlRows.length > 0);
  const hasChunks = Boolean(chunks && chunks.length > 0);
  if (!hasSql && !hasChunks) return null;

  const columns = hasSql ? Object.keys(sqlRows![0]) : [];
  const count = (sqlRows?.length ?? 0) + (chunks?.length ?? 0);

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
          <ul className="space-y-1">
            {chunks!.map((chunk) => (
              <li key={chunk.id} className="rounded border p-2">
                <div className="font-medium">
                  {chunk.company} — {chunk.source ?? "unknown source"}
                  {chunk.page != null ? `, p.${chunk.page}` : ""}
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
