"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { AlertTriangle, SendHorizontal, Sparkles, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { MessageBubble } from "@/components/MessageBubble";
import { Alert, AlertAction, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { getToken } from "@/lib/auth";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const SUGGESTED_PROMPTS = [
  "Apple's net income from 2022 to 2025",
  "Compare Google and Meta's revenue structure and strategy in 2025",
  "Which had higher revenue growth in 2024–2025, Apple or Meta, and why?",
  "What does Microsoft's 10-K say about its cloud strategy?",
];

export function ChatWindow() {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const { messages, sendMessage, status, error, regenerate, clearError } = useChat({
    transport: new DefaultChatTransport({
      api: `${API_URL}/api/chat`,
      // Function form -> re-read the token fresh on every request, not just once.
      headers: () => ({ Authorization: `Bearer ${getToken() ?? ""}` }),
    }),
  });

  const busy = status === "streaming" || status === "submitted";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, status]);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || busy) return;
    sendMessage({ text: input });
    setInput("");
  }

  function handlePromptClick(prompt: string) {
    if (busy) return;
    setInput(prompt);
  }

  return (
    <div className="flex h-[calc(100vh-8rem)] w-full max-w-3xl flex-col overflow-hidden rounded-xl border bg-card shadow-sm">
      <ScrollArea className="flex-1 p-4 sm:p-6">
        <div className="flex flex-col gap-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center gap-4 py-10 text-center">
              <div className="flex size-12 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-primary/60 text-primary-foreground">
                <Sparkles className="size-6" />
              </div>
              <div className="space-y-1">
                <p className="text-sm font-medium">Ask a financial question to get started</p>
                <p className="text-sm text-muted-foreground">
                  Grounded in SQL financials and FY2025 10-K filings — I&apos;ll say so if the data isn&apos;t available.
                </p>
              </div>
              <div className="flex flex-wrap justify-center gap-2 pt-2">
                {SUGGESTED_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => handlePromptClick(prompt)}
                    className="rounded-full border bg-background px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((message, index) => (
            <MessageBubble
              key={message.id}
              message={message}
              isStreaming={busy && index === messages.length - 1}
            />
          ))}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>
      <div className="border-t p-3 sm:p-4">
        {status === "error" && (
          <Alert variant="destructive" className="mb-3">
            <AlertTriangle />
            <AlertTitle>Something went wrong</AlertTitle>
            <AlertDescription>
              {error?.message || "The request failed mid-stream. You can try again."}
            </AlertDescription>
            <AlertAction className="flex gap-1">
              <Button type="button" variant="ghost" size="xs" onClick={() => regenerate()}>
                Try again
              </Button>
              <Button
                type="button"
                variant="ghost"
                size="icon-xs"
                aria-label="Dismiss"
                onClick={() => clearError()}
              >
                <X className="size-3" />
              </Button>
            </AlertAction>
          </Alert>
        )}
        <form onSubmit={handleSubmit} className="flex items-center gap-2 rounded-full border bg-background pr-1.5 pl-3 focus-within:ring-3 focus-within:ring-ring/50">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a financial question..."
            disabled={busy}
            className="h-10 border-none bg-transparent px-0 shadow-none focus-visible:ring-0"
          />
          <Button
            type="submit"
            size="icon"
            disabled={busy || !input.trim()}
            aria-label="Send"
            className="rounded-full"
          >
            <SendHorizontal className="size-4" />
          </Button>
        </form>
      </div>
    </div>
  );
}
