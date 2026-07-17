import { Badge } from "@/components/ui/badge";

const LABELS: Record<string, string> = {
  sql: "SQL",
  vector: "10-K",
  both: "Hybrid",
  refuse: "Refused",
  clarify: "Needs info",
};

const VARIANTS: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
  sql: "default",
  vector: "default",
  both: "secondary",
  refuse: "destructive",
  clarify: "outline",
};

export function RouteBadge({ route }: { route?: string | null }) {
  if (!route) return null;
  return <Badge variant={VARIANTS[route] ?? "outline"}>{LABELS[route] ?? route}</Badge>;
}
