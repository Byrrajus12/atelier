import { expect, test } from "@playwright/test";

test("dashboard connects to the mock feed, descends the curve, fills the log, and shows the terminal badge", async ({
  page,
}) => {
  await page.goto("/");

  await expect(page.locator(".run-header__connection")).toContainText("connected", {
    timeout: 10_000,
  });

  await expect(page.locator(".stroke-log__row").first()).toBeVisible({ timeout: 10_000 });

  // The mock feed's canned run always ends at this reason/value regardless of
  // when this client joined mid-loop, so this is a stable assertion that the
  // whole pipeline (reducer join + terminal handling) rendered correctly.
  await expect(page.locator(".run-header__badge")).toContainText("converged", {
    timeout: 20_000,
  });
  await expect(page.locator(".run-header__metric-value")).toHaveText("0.0186");

  const line = page.locator(".error-curve__line");
  await expect(line).toBeVisible();
  await expect(line).toHaveAttribute("d", /M.+L.+/);

  const rowCount = await page.locator(".stroke-log__row").count();
  expect(rowCount).toBeGreaterThan(0);
});
