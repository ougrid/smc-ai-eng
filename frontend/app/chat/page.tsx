"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { AppHeader } from "@/components/AppHeader";
import { ChatWindow } from "@/components/ChatWindow";
import { Skeleton } from "@/components/ui/skeleton";
import { getToken } from "@/lib/auth";

export default function ChatPage() {
  const router = useRouter();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
      return;
    }
    // localStorage only exists client-side, so this can't be derived at
    // render time (SSR) -- the effect is the hydration-safe check itself.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setReady(true);
  }, [router]);

  if (!ready) {
    return (
      <div className="flex flex-1 items-center justify-center p-4">
        <Skeleton className="h-[calc(100vh-8rem)] w-full max-w-3xl rounded-xl" />
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col">
      <AppHeader />
      <div className="flex flex-1 items-center justify-center p-4">
        <ChatWindow />
      </div>
    </div>
  );
}
