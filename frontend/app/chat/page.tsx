"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { ChatWindow } from "@/components/ChatWindow";
import { getToken } from "@/lib/auth";

export default function ChatPage() {
  const router = useRouter();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
      return;
    }
    setReady(true);
  }, [router]);

  if (!ready) return null;

  return (
    <div className="flex flex-1 items-center justify-center">
      <ChatWindow />
    </div>
  );
}
