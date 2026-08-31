import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// Not using vitest's `globals: true`, so @testing-library/react's automatic
// afterEach cleanup never gets registered on its own -- without this, DOM
// output from one test's render() leaks into the next test in the same file.
afterEach(() => {
  cleanup();
});
