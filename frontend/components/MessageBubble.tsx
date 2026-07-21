"use client";

import type { UIMessage } from "ai";
import { Check, CircleQuestionMark, Copy, TriangleAlert } from "lucide-react";
import { useState } from "react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { CitationList } from "@/components/CitationList";
import { Markdown } from "@/components/Markdown";
import { RouteBadge } from "@/components/RouteBadge";
import { TypingDots, TypingIndicator } from "@/components/TypingIndicator";

type RouteData = { route?: string };
type CoverageData = { notes?: string[] };
type CitationsData = { sql_rows?: Record<string, unknown>[]; chunks?: Chunk[] };
type StatusData = { stage?: string; label?: string };
type VerifyData = { ok?: boolean; ungrounded?: string[]; attempt?: number };
type Chunk = {
  id: string;
  company: string;
  source: string | null;
  page: number | null;
  text: string;
  score: number;
};

function partData<T>(message: UIMessage, type: string): T | undefined {
  const part = message.parts.find((p) => p.type === type) as { data?: T } | undefined;
  return part?.data;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  return (
    <Button
      type="button"
      variant="ghost"
      size="icon-xs"
      className="text-muted-foreground hover:text-foreground"
      aria-label="Copy answer"
      onClick={async () => {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
    >
      {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
    </Button>
  );
}

export function MessageBubble({
  message,
  isStreaming = false,
}: {
  message: UIMessage;
  isStreaming?: boolean;
}) {
  const isUser = message.role === "user";

  // Stream-veto rule: a failed verify redrafts the answer as a brand-new
  // text part rather than editing the old one, so multiple "text" parts can
  // exist on one message -- only the LAST one is ever the real answer.
  const textParts = message.parts.filter((p) => p.type === "text");
  const text = textParts.length ? textParts[textParts.length - 1].text : "";

  const route = partData<RouteData>(message, "data-route")?.route;
  const coverageNotes = partData<CoverageData>(message, "data-coverage")?.notes;
  const citations = partData<CitationsData>(message, "data-citations");
  const verify = partData<VerifyData>(message, "data-verify");
  // Live progress label (fixed id "status-1" server-side, so this reconciles
  // to the latest stage rather than accumulating). Only meaningful before the
  // answer text starts streaming; hidden once it does.
  const statusLabel = partData<StatusData>(message, "data-status")?.label;

  const provisional = isStreaming && !isUser && !verify;
  const hasContent = Boolean(route || text || citations || verify || statusLabel);

  // Nothing has arrived from the graph yet -- show a typing cue instead of
  // an empty bubble.
  if (!isUser && isStreaming && !hasContent) {
    return <TypingIndicator />;
  }

  const isRefusal = !isUser && route === "refuse";
  const isClarify = !isUser && route === "clarify";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`flex max-w-[85%] flex-col gap-1.5 sm:max-w-[80%] ${isUser ? "items-end" : "items-start"}`}>
        {!isUser && (route || verify) && (
          <div className="flex items-center gap-1.5">
            <RouteBadge route={route} />
            {verify && verify.ok === false && (
              <Badge variant="outline">Not fully verified</Badge>
            )}
          </div>
        )}
        {isRefusal && text ? (
          <Alert variant="destructive" className={`w-full ${provisional ? "opacity-70" : ""}`}>
            <TriangleAlert />
            <AlertDescription>
              <Markdown text={text} />
            </AlertDescription>
          </Alert>
        ) : isClarify && text ? (
          <Alert className={`w-full ${provisional ? "opacity-70" : ""}`}>
            <CircleQuestionMark />
            <AlertDescription>
              <Markdown text={text} />
            </AlertDescription>
          </Alert>
        ) : (
          <div
            className={`group relative rounded-2xl px-4 py-2.5 ${
              isUser
                ? "bg-primary text-primary-foreground"
                : "bg-muted"
            } ${provisional ? "opacity-70" : ""}`}
          >
            {isUser ? (
              <p className="text-sm">{text}</p>
            ) : text ? (
              <Markdown text={text} />
            ) : isStreaming ? (
              // Retrieval/synthesis is running but no answer text yet -- keep
              // the typing cue alive and label the current stage so the
              // multi-second wait doesn't read as dead air.
              <div className="flex flex-col gap-2">
                <TypingDots />
                {statusLabel && (
                  <span className="text-xs text-muted-foreground">{statusLabel}</span>
                )}
              </div>
            ) : (
              <span className="text-sm italic text-muted-foreground">...</span>
            )}
          </div>
        )}
        {!isUser && !provisional && text && (
          <CopyButton text={text} />
        )}
        {!isUser && coverageNotes && coverageNotes.length > 0 && (
          <Alert className="w-full">
            <AlertDescription>{coverageNotes.join(" ")}</AlertDescription>
          </Alert>
        )}
        {!isUser && citations && (
          <CitationList sqlRows={citations.sql_rows} chunks={citations.chunks} />
        )}
      </div>
    </div>
  );
}
