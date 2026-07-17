import type { UIMessage } from "ai";

export function MessageBubble({ message }: { message: UIMessage }) {
  const text = message.parts
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("");

  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[80%] rounded-lg px-3 py-2 text-sm ${
          isUser ? "bg-primary text-primary-foreground" : "bg-muted"
        }`}
      >
        {text || <span className="italic text-muted-foreground">...</span>}
      </div>
    </div>
  );
}
