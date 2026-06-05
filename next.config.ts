import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emit a self-contained build under .next/standalone with only the
  // runtime files needed to serve the app. Cuts the production Docker
  // image from ~1.2 GB (full node_modules) to ~200 MB. No effect on
  // `npm run dev` or local `npm run build`.
  output: "standalone",
};

export default nextConfig;
