import { test, expect } from "@playwright/test";

// Real system, no mocks: playwright.config.ts boots the actual FastAPI app
// (real trained models under models/) and the actual Next.js dashboard,
// and this walks the one real user journey the app has -- pick a use case,
// see it ranked, click a row, see why it scored the way it did.

test("user can pick a use case, see the leaderboard, and click a row to inspect its factors", async ({
  page,
}) => {
  await page.goto("/");

  const useCaseSelect = page.locator("select.use-case-select");
  await expect(useCaseSelect.locator("option")).not.toHaveCount(0);
  await useCaseSelect.selectOption("gtm_fit");

  const rows = page.locator("table.leaderboard tbody tr");
  await expect(rows.first()).toBeVisible({ timeout: 15_000 });
  expect(await rows.count()).toBeGreaterThan(1); // need a second row to click

  const secondRowId = await rows.nth(1).locator("td").nth(1).innerText();
  await rows.nth(1).click();

  await expect(page.getByRole("heading", { level: 2, name: secondRowId })).toBeVisible();
  await expect(page.locator(".score")).toContainText("%");

  const shapRows = page.locator(".shap-row");
  await expect(shapRows.first()).toBeVisible();
  expect(await shapRows.count()).toBeGreaterThan(0);

  // The click also updates the row's own selected styling.
  await expect(rows.nth(1)).toHaveClass(/selected/);
});

test("switching the use-case picker loads a different leaderboard without error", async ({ page }) => {
  await page.goto("/");

  const useCaseSelect = page.locator("select.use-case-select");
  await expect(useCaseSelect.locator("option")).not.toHaveCount(0);

  await useCaseSelect.selectOption("gtm_fit");
  const rows = page.locator("table.leaderboard tbody tr");
  await expect(rows.first()).toBeVisible({ timeout: 15_000 });

  await useCaseSelect.selectOption("ptb_prospect");
  await expect(rows.first()).toBeVisible({ timeout: 15_000 });

  await expect(page.locator(".error")).toHaveCount(0);
  await expect(useCaseSelect).toHaveValue("ptb_prospect");
});
