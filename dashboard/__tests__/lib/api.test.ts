import { describe, it, expect, vi, afterEach } from "vitest";

const originalFetch = global.fetch;

afterEach(() => {
  global.fetch = originalFetch;
  vi.unstubAllEnvs();
  vi.resetModules();
});

function mockFetchOnce(response: { ok: boolean; status?: number; body?: unknown }) {
  global.fetch = vi.fn().mockResolvedValue({
    ok: response.ok,
    status: response.status,
    json: async () => response.body,
  });
}

describe("getUseCases", () => {
  it("fetches /use-cases and returns the parsed list", async () => {
    const useCases = [
      {
        name: "contact_score",
        display_name: "Contact Score",
        description: "Probability a contact converts.",
        entity: "contact",
        label_column: "contact_converted",
      },
      {
        name: "gtm_fit",
        display_name: "GTM Fit",
        description: "Static ICP fit score.",
        entity: "account",
        label_column: "gtm_fit_label",
      },
    ];
    mockFetchOnce({ ok: true, body: useCases });
    const { getUseCases } = await import("@/lib/api");

    const result = await getUseCases();

    expect(result).toEqual(useCases);
    expect(global.fetch).toHaveBeenCalledWith("http://localhost:8000/use-cases");
  });

  it("throws a descriptive error when the response is not ok", async () => {
    mockFetchOnce({ ok: false, status: 500 });
    const { getUseCases } = await import("@/lib/api");

    await expect(getUseCases()).rejects.toThrow("Failed to load use cases (500)");
  });

  it("respects NEXT_PUBLIC_API_BASE_URL when set", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "https://api.example.com");
    vi.resetModules();
    mockFetchOnce({ ok: true, body: [] });
    const { getUseCases } = await import("@/lib/api");

    await getUseCases();

    expect(global.fetch).toHaveBeenCalledWith("https://api.example.com/use-cases");
  });
});

describe("getLeaderboard", () => {
  it("fetches the leaderboard with the default limit and no auth header when NEXT_PUBLIC_API_KEY is unset", async () => {
    mockFetchOnce({ ok: true, body: { use_case: "gtm_fit", count: 0, results: [] } });
    const { getLeaderboard } = await import("@/lib/api");

    const result = await getLeaderboard("gtm_fit");

    expect(result).toEqual({ use_case: "gtm_fit", count: 0, results: [] });
    expect(global.fetch).toHaveBeenCalledWith(
      "http://localhost:8000/score/gtm_fit/leaderboard?limit=25",
      { headers: {} }
    );
  });

  it("passes a custom limit through", async () => {
    mockFetchOnce({ ok: true, body: { use_case: "gtm_fit", count: 0, results: [] } });
    const { getLeaderboard } = await import("@/lib/api");

    await getLeaderboard("gtm_fit", 10);

    expect(global.fetch).toHaveBeenCalledWith(
      "http://localhost:8000/score/gtm_fit/leaderboard?limit=10",
      { headers: {} }
    );
  });

  it("attaches X-API-Key when NEXT_PUBLIC_API_KEY is set", async () => {
    vi.stubEnv("NEXT_PUBLIC_API_KEY", "secret");
    vi.resetModules();
    mockFetchOnce({ ok: true, body: { use_case: "gtm_fit", count: 0, results: [] } });
    const { getLeaderboard } = await import("@/lib/api");

    await getLeaderboard("gtm_fit");

    expect(global.fetch).toHaveBeenCalledWith(
      "http://localhost:8000/score/gtm_fit/leaderboard?limit=25",
      { headers: { "X-API-Key": "secret" } }
    );
  });

  it("throws a descriptive error naming the use case and status when the response is not ok", async () => {
    mockFetchOnce({ ok: false, status: 404 });
    const { getLeaderboard } = await import("@/lib/api");

    await expect(getLeaderboard("nope")).rejects.toThrow(
      "Failed to load leaderboard for nope (404)"
    );
  });
});
