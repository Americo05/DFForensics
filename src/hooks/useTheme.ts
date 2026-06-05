"use client";

import { useCallback, useSyncExternalStore } from 'react';

// External store: a single boolean ("is dark") backed by localStorage and
// fanned out via a window CustomEvent so every useTheme caller stays in
// sync without a Context Provider.
//
// useSyncExternalStore is React's blessed way to consume mutable external
// state with hydration safety: getServerSnapshot is used on the server
// (always dark), getSnapshot is used on the client (reads the cache).
// React reconciles the two without a setState-in-effect dance — which is
// what the previous useState+useEffect version tripped over.

const THEME_CHANGE_EVENT = 'app-theme-change';
const STORAGE_KEY = 'theme';

// Cached client-side snapshot. useSyncExternalStore requires getSnapshot
// to return the SAME reference when nothing changed (otherwise it bails
// out as a tearing risk). Reading localStorage on every call returns a
// fresh string each time — fine for booleans but defensive caching here
// keeps the contract crystal-clear.
let clientSnapshot: boolean | null = null;

function getSnapshot(): boolean {
  if (clientSnapshot !== null) return clientSnapshot;
  if (typeof window === 'undefined') return true;
  clientSnapshot = localStorage.getItem(STORAGE_KEY) !== 'light';
  return clientSnapshot;
}

function getServerSnapshot(): boolean {
  // Server has no storage; pick a stable default. The client either
  // agrees (already dark) or flips on the first post-mount commit.
  return true;
}

function subscribe(callback: () => void): () => void {
  const handler = () => {
    // Invalidate cache so the next getSnapshot reads fresh state.
    clientSnapshot = null;
    callback();
  };
  window.addEventListener(THEME_CHANGE_EVENT, handler);
  return () => window.removeEventListener(THEME_CHANGE_EVENT, handler);
}

export function useTheme() {
  const isDark = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);

  const toggle = useCallback(() => {
    const next = !getSnapshot();
    // Update the cache + persisted preference + DOM attribute eagerly so
    // the next render (triggered by the broadcast below) sees a fresh
    // snapshot.
    clientSnapshot = next;
    localStorage.setItem(STORAGE_KEY, next ? 'dark' : 'light');
    if (next) {
      document.documentElement.removeAttribute('data-theme');
    } else {
      document.documentElement.setAttribute('data-theme', 'light');
    }
    // Broadcast: every mounted useTheme re-renders with the new snapshot.
    window.dispatchEvent(new Event(THEME_CHANGE_EVENT));
  }, []);

  return { isDark, toggle };
}
