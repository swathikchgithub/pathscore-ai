import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import ShapBarChart from "@/components/ShapBarChart";
import type { Factor } from "@/lib/types";

describe("ShapBarChart", () => {
  it("renders one row per factor with its value formatted to 3 decimals", () => {
    const factors: Factor[] = [
      { feature: "industry", impact: 0.482 },
      { feature: "company_size", impact: -0.12 },
    ];
    render(<ShapBarChart factors={factors} />);

    expect(screen.getByText("industry")).toBeInTheDocument();
    expect(screen.getByText("0.482")).toBeInTheDocument();
    expect(screen.getByText("company_size")).toBeInTheDocument();
    expect(screen.getByText("-0.120")).toBeInTheDocument();
  });

  it("marks positive impacts and negative impacts with distinct classes", () => {
    const factors: Factor[] = [
      { feature: "a", impact: 0.5 },
      { feature: "b", impact: -0.5 },
    ];
    const { container } = render(<ShapBarChart factors={factors} />);

    const bars = container.querySelectorAll(".shap-bar");
    expect(bars[0]).toHaveClass("positive");
    expect(bars[1]).toHaveClass("negative");
  });

  it("scales each bar's width relative to the largest absolute impact", () => {
    const factors: Factor[] = [
      { feature: "big", impact: 1.0 },
      { feature: "small", impact: -0.25 },
    ];
    const { container } = render(<ShapBarChart factors={factors} />);

    const bars = container.querySelectorAll<HTMLElement>(".shap-bar");
    expect(bars[0].style.width).toBe("100%");
    expect(bars[1].style.width).toBe("25%");
  });

  it("renders a zero-width bar for a zero impact instead of NaN%", () => {
    const factors: Factor[] = [{ feature: "flat", impact: 0 }];
    const { container } = render(<ShapBarChart factors={factors} />);

    const bar = container.querySelector<HTMLElement>(".shap-bar");
    expect(bar).toHaveClass("positive"); // impact >= 0
    expect(bar?.style.width).toBe("0%");
  });

  it("renders nothing for an empty factor list without crashing", () => {
    const { container } = render(<ShapBarChart factors={[]} />);
    expect(container.querySelectorAll(".shap-row")).toHaveLength(0);
  });
});
