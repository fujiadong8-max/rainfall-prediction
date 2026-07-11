"""
Taxi dynamic-pricing demo driven by rainfall prediction probabilities.

Reference model semantics:
- https://github.com/rohitl17/cab-dynamic-pricing describes a cab-price
  estimation app using trip features, surge multiplier, and weather inputs.

This file does not copy external project code. It implements a small,
reproducible demo layer for this repository: rainfall prediction probabilities
are converted into a weather multiplier and compared with a no-rain
counterfactual price.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs/mlp_predictions.csv")
DEFAULT_OUTPUT = Path("outputs/taxi_pricing_demo.csv")
DEFAULT_REPORT = Path("outputs/taxi_pricing_demo_report.html")
DEFAULT_SUMMARY = Path("outputs/taxi_pricing_demo_summary.json")


@dataclass(frozen=True)
class PricingConfig:
    base_flag_fare: float = 4.20
    per_km_rate: float = 1.35
    per_minute_rate: float = 0.42
    peak_hour_markup: float = 0.18
    location_pressure_scale: float = 0.22
    max_weather_markup: float = 0.55
    price_cap_multiplier: float = 3.00
    currency: str = "AUD"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def load_rainfall_predictions(path: str | Path) -> pd.DataFrame:
    input_path = resolve_path(path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Cannot find rainfall prediction file: {input_path}. "
            "Run predict.py first or pass --predictions."
        )

    df = pd.read_csv(input_path)
    require_columns(df, ["RainTomorrow_probability"])

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["Date"] = pd.NaT

    if "Location" not in df.columns:
        df["Location"] = "Unknown"

    df = df.dropna(subset=["RainTomorrow_probability"]).copy()
    df["RainTomorrow_probability"] = df["RainTomorrow_probability"].clip(0, 1)
    if df.empty:
        raise ValueError("No usable rainfall probabilities found in the input file.")
    return df


def rainfall_risk_label(probability: float) -> str:
    if probability >= 0.80:
        return "very_high"
    if probability >= 0.60:
        return "high"
    if probability >= 0.35:
        return "moderate"
    return "low"


def rain_multiplier(probability: pd.Series, max_weather_markup: float) -> pd.Series:
    """Map P(rain tomorrow) to a bounded weather multiplier.

    The transformation is monotonic and conservative: low risk produces no
    material markup, while high probabilities gradually approach the configured
    maximum markup.
    """
    adjusted = ((probability - 0.20) / 0.80).clip(lower=0, upper=1)
    return 1.0 + max_weather_markup * adjusted.pow(1.35)


def build_demo_trips(
    predictions: pd.DataFrame,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    """Create reproducible synthetic trip contexts from weather rows.

    This project has weather observations but no taxi order log. The generated
    trip fields are therefore scenario variables for a demo, not empirical taxi
    behavior data.
    """
    n = min(sample_size, len(predictions))
    sampled = predictions.sample(n=n, random_state=random_state).reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    hour_weights = np.array(
        [
            0.020,
            0.010,
            0.010,
            0.010,
            0.015,
            0.030,
            0.055,
            0.075,
            0.075,
            0.040,
            0.035,
            0.035,
            0.040,
            0.040,
            0.040,
            0.045,
            0.060,
            0.075,
            0.080,
            0.065,
            0.050,
            0.040,
            0.025,
            0.020,
        ]
    )
    hour_weights = hour_weights / hour_weights.sum()
    hours = rng.choice(np.arange(24), size=n, p=hour_weights)

    distances = rng.gamma(shape=2.4, scale=2.2, size=n).clip(1.0, 28.0)
    speed_km_h = rng.normal(loc=28.0, scale=6.0, size=n).clip(12.0, 48.0)
    durations = (distances / speed_km_h * 60.0 + rng.normal(4.0, 1.5, size=n)).clip(4.0, 90.0)

    location_codes = pd.factorize(sampled["Location"].astype(str), sort=True)[0]
    location_pressure = (location_codes % 7) / 6.0
    peak_hour = np.isin(hours, [7, 8, 9, 16, 17, 18, 19]).astype(float)

    demo = pd.DataFrame(
        {
            "Date": sampled["Date"],
            "Location": sampled["Location"].astype(str),
            "trip_hour": hours,
            "trip_distance_km": distances,
            "trip_duration_min": durations,
            "location_pressure_index": location_pressure,
            "peak_hour": peak_hour.astype(int),
            "rain_probability": sampled["RainTomorrow_probability"].to_numpy(),
        }
    )

    if "RainTomorrow_pred_label" in sampled.columns:
        demo["rain_pred_label"] = sampled["RainTomorrow_pred_label"].astype(str).to_numpy()
    else:
        demo["rain_pred_label"] = np.where(demo["rain_probability"] >= 0.5, "Yes", "No")

    demo["rain_risk_level"] = demo["rain_probability"].map(rainfall_risk_label)
    return demo


def apply_pricing(trips: pd.DataFrame, config: PricingConfig) -> pd.DataFrame:
    priced = trips.copy()

    priced["base_price"] = (
        config.base_flag_fare
        + config.per_km_rate * priced["trip_distance_km"]
        + config.per_minute_rate * priced["trip_duration_min"]
    )
    priced["demand_supply_multiplier"] = (
        1.0
        + config.peak_hour_markup * priced["peak_hour"]
        + config.location_pressure_scale * priced["location_pressure_index"]
    ).clip(0.85, 1.75)
    priced["rain_multiplier"] = rain_multiplier(
        priced["rain_probability"],
        max_weather_markup=config.max_weather_markup,
    )
    priced["total_multiplier"] = (
        priced["demand_supply_multiplier"] * priced["rain_multiplier"]
    ).clip(upper=config.price_cap_multiplier)

    priced["price_without_rain_signal"] = (
        priced["base_price"] * priced["demand_supply_multiplier"]
    )
    priced["price_with_rain_signal"] = priced["base_price"] * priced["total_multiplier"]
    priced["rain_increment"] = (
        priced["price_with_rain_signal"] - priced["price_without_rain_signal"]
    )
    priced["rain_increment_pct"] = (
        priced["rain_increment"] / priced["price_without_rain_signal"] * 100.0
    )

    money_columns = [
        "trip_distance_km",
        "trip_duration_min",
        "rain_probability",
        "base_price",
        "demand_supply_multiplier",
        "rain_multiplier",
        "total_multiplier",
        "price_without_rain_signal",
        "price_with_rain_signal",
        "rain_increment",
        "rain_increment_pct",
    ]
    priced[money_columns] = priced[money_columns].round(3)
    return priced


def summarize(priced: pd.DataFrame) -> dict:
    by_risk = (
        priced.groupby("rain_risk_level", observed=True)
        .agg(
            trips=("rain_probability", "size"),
            avg_rain_probability=("rain_probability", "mean"),
            avg_rain_multiplier=("rain_multiplier", "mean"),
            avg_price_without_rain_signal=("price_without_rain_signal", "mean"),
            avg_price_with_rain_signal=("price_with_rain_signal", "mean"),
            avg_rain_increment=("rain_increment", "mean"),
            avg_rain_increment_pct=("rain_increment_pct", "mean"),
        )
        .reset_index()
    )

    order = {"low": 0, "moderate": 1, "high": 2, "very_high": 3}
    by_risk["_order"] = by_risk["rain_risk_level"].map(order)
    by_risk = by_risk.sort_values("_order").drop(columns="_order")

    overall = {
        "rows": int(len(priced)),
        "avg_price_without_rain_signal": float(priced["price_without_rain_signal"].mean()),
        "avg_price_with_rain_signal": float(priced["price_with_rain_signal"].mean()),
        "avg_rain_increment": float(priced["rain_increment"].mean()),
        "avg_rain_increment_pct": float(priced["rain_increment_pct"].mean()),
        "max_total_multiplier": float(priced["total_multiplier"].max()),
    }

    return {
        "overall": {key: round(value, 4) for key, value in overall.items()},
        "by_risk": by_risk.round(4).to_dict(orient="records"),
    }


def write_outputs(
    priced: pd.DataFrame,
    summary: dict,
    output_path: str | Path,
    report_path: str | Path,
    summary_path: str | Path,
    config: PricingConfig,
) -> None:
    output_path = resolve_path(output_path)
    report_path = resolve_path(report_path)
    summary_path = resolve_path(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    priced.to_csv(output_path, index=False, encoding="utf-8-sig")

    payload = {"config": asdict(config), **summary}
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    by_risk = pd.DataFrame(summary["by_risk"])
    top_rows = priced.sort_values("rain_increment", ascending=False).head(20)

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Rainfall-driven taxi pricing demo</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .note {{ color: #555; max-width: 960px; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f4f6f8; }}
    .metric {{ display: inline-block; margin: 12px 24px 12px 0; }}
    .metric b {{ display: block; font-size: 22px; }}
  </style>
</head>
<body>
  <h1>Rainfall-driven taxi pricing demo</h1>
  <p class="note">
    This is a scenario demo. Trip distance, duration, hour, and local pressure
    are synthetic variables. The empirical input from this repository is the
    rainfall prediction probability. The core comparison is the counterfactual
    price without a rainfall signal versus the price after adding the rainfall
    risk multiplier.
  </p>
  <div class="metric"><span>Rows</span><b>{summary["overall"]["rows"]}</b></div>
  <div class="metric"><span>Average no-rain price</span><b>{summary["overall"]["avg_price_without_rain_signal"]:.2f} {config.currency}</b></div>
  <div class="metric"><span>Average rain-informed price</span><b>{summary["overall"]["avg_price_with_rain_signal"]:.2f} {config.currency}</b></div>
  <div class="metric"><span>Average rain increment</span><b>{summary["overall"]["avg_rain_increment"]:.2f} {config.currency}</b></div>

  <h2>Summary by rainfall risk</h2>
  {by_risk.to_html(index=False, escape=True)}

  <h2>Largest rainfall-driven price changes</h2>
  {top_rows.to_html(index=False, escape=True)}
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")


def run_demo(args: argparse.Namespace) -> dict:
    config = PricingConfig(
        max_weather_markup=args.max_weather_markup,
        price_cap_multiplier=args.price_cap_multiplier,
    )
    predictions = load_rainfall_predictions(args.predictions)
    trips = build_demo_trips(
        predictions=predictions,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    priced = apply_pricing(trips, config)
    summary = summarize(priced)
    write_outputs(
        priced=priced,
        summary=summary,
        output_path=args.output,
        report_path=args.report,
        summary_path=args.summary,
        config=config,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a rainfall-informed taxi dynamic-pricing demo."
    )
    parser.add_argument(
        "--predictions",
        default=str(DEFAULT_INPUT),
        help="CSV produced by predict.py with RainTomorrow_probability.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output priced-trip CSV.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Output HTML report.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Output JSON summary.")
    parser.add_argument("--sample-size", type=int, default=2500, help="Number of scenario trips.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-weather-markup",
        type=float,
        default=0.55,
        help="Maximum weather-only markup when rain probability approaches 1.",
    )
    parser.add_argument(
        "--price-cap-multiplier",
        type=float,
        default=3.0,
        help="Upper cap for total multiplier after demand and rain adjustments.",
    )
    return parser.parse_args()


def main() -> None:
    summary = run_demo(parse_args())
    print("Taxi pricing demo complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
