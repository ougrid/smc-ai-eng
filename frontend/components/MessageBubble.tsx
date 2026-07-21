import type { UIMessage } from "ai";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { CitationList } from "@/components/CitationList";
import { RouteBadge } from "@/components/RouteBadge";

type RouteData = { route?: string };
type CoverageData = { notes?: string[] };
type CitationsData = { sql_rows?: Record<string, unknown>[]; chunks?: Chunk[] };
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

  const provisional = isStreaming && !isUser && !verify;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`flex max-w-[80%] flex-col gap-1.5 ${isUser ? "items-end" : "items-start"}`}>
        {!isUser && (route || verify) && (
          <div className="flex items-center gap-1.5">
            <RouteBadge route={route} />
            {verify && verify.ok === false && (
              <Badge variant="outline">Not fully verified</Badge>
            )}
          </div>
        )}
        <div
          className={`rounded-lg px-3 py-2 text-sm ${
            isUser ? "bg-primary text-primary-foreground" : "bg-muted"
          } ${provisional ? "opacity-70" : ""}`}
        >
          {text || <span className="italic text-muted-foreground">...</span>}
        </div>
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
