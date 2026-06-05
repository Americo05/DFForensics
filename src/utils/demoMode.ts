/**
 * demoMode.ts — runtime detection of "is this the public site (no engine)
 * or a local install (engine running on localhost)".
 *
 * Why hostname-based: the engine is a Python service that only ever runs
 * on the user's own machine (Docker Compose). Anything hosted publicly
 * (Vercel, custom domain, preview deploy) inherently has no engine and
 * should show the "demo" experience pointing visitors at the download.
 *
 * Treating localhost-style hostnames as "not demo" means a clone+npm-dev
 * developer keeps the full UI without needing to flip any env var.
 *
 * Returns `true` during SSR so the first paint matches the "demo" branch
 * for the public site — same hydration-safety logic as useTheme.
 */

const LOCAL_HOSTNAMES = new Set([
  "localhost",
  "127.0.0.1",
  "0.0.0.0",
  "[::1]",
]);

export function isDemoMode(): boolean {
  if (typeof window === "undefined") return true;
  const host = window.location.hostname;
  if (LOCAL_HOSTNAMES.has(host)) return false;
  // Common LAN ranges — when someone runs the dashboard on their laptop and
  // visits it from a phone on the same network, the hostname is an IP like
  // 192.168.1.42 or 10.0.0.5. Treat those as local too.
  if (/^192\.168\./.test(host)) return false;
  if (/^10\./.test(host)) return false;
  if (/^172\.(1[6-9]|2[0-9]|3[01])\./.test(host)) return false;
  return true;
}
