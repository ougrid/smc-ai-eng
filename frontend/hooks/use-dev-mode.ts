"use client";

import { useSyncExternalStore } from "react";

import { getDevMode, subscribeDevMode } from "@/lib/devmode";

// Server snapshot is always false (localStorage is client-only); the store
// reconciles on hydration. Every subscriber re-renders when the flag flips.
export function useDevMode(): boolean {
  return useSyncExternalStore(subscribeDevMode, getDevMode, () => false);
}
