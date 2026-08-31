import { defineConfig, devices } from "@playwright/test";
import path from "path";

// The real running system: real FastAPI serving the real trained models
// under models/, real Next.js dashboard on top of it. No mocks -- that's
// the point of an E2E test as opposed to the unit/integration coverage
// everywhere else in this repo.
const REPO_ROOT = path.resolve(__dirname, "..");
const API_PORT = 8010;
const DASHBOARD_PORT = 3010;

export default defineConfig({
  testDir: "./__tests__/e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  reporter: "list",
  use: {
    baseURL: `http://localhost:${DASHBOARD_PORT}`,
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: [
    {
      // python3 must already have requirements.txt installed -- same
      // assumption the README's own Quickstart makes.
      command: `python3 -m uvicorn src.serving.app:app --port ${API_PORT}`,
      cwd: REPO_ROOT,
      url: `http://localhost:${API_PORT}/health`,
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
    },
    {
      command: "npm run dev",
      cwd: __dirname,
      url: `http://localhost:${DASHBOARD_PORT}`,
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
      env: {
        NEXT_PUBLIC_API_BASE_URL: `http://localhost:${API_PORT}`,
        PORT: String(DASHBOARD_PORT),
      },
    },
  ],
});
