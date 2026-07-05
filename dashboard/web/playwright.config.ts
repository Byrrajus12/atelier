import { defineConfig } from "@playwright/test";

// Drives the dashboard against dashboard/web/mock-feed.mjs (a canned,
// looping converging run), not the real orchestrator. Confirms the
// event-stream -> reducer -> UI wiring; the real reconnect/bootstrap proof
// is opening this dashboard against scripts/live_run.py, which only the
// person running that script can do.
export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  use: {
    baseURL: "http://localhost:5173",
  },
  webServer: [
    {
      command: "node mock-feed.mjs",
      port: 8765,
      reuseExistingServer: !process.env.CI,
      timeout: 10_000,
    },
    {
      command: "npm run dev",
      port: 5173,
      reuseExistingServer: !process.env.CI,
      timeout: 10_000,
    },
  ],
});
