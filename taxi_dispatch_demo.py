"""
Rainfall-aware taxi driver dispatch demo.

Reference scaffold:
- https://github.com/lanshu-cy/Optimizing-Ride-Requests-for-Drivers-using-Reinforcement-Learning
  frames taxi operation as a sequential decision problem with wait, accept, and
  move actions. This file uses that state/action/reward framing, but keeps the
  implementation local, small, and reproducible for this rainfall project.

This is not a validated reinforcement-learning experiment. It is a decision
environment demo that shows how RainTomorrow_probability can enter driver
acceptance and repositioning decisions. A DQN/Dueling-DQN model should only be
added after this environment is made empirically defensible with real trip
requests, driver availability, and weather joins.
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
DEFAULT_OUTPUT = Path("outputs/taxi_dispatch_demo.csv")
DEFAULT_REPORT = Path("outputs/taxi_dispatch_demo_report.html")
DEFAULT_SUMMARY = Path("outputs/taxi_dispatch_demo_summary.json")


ACTION_COLUMNS = {
    "wait": "score_wait",
    "accept": "score_accept",
    "relocate": "score_relocate",
}


@dataclass(frozen=True)
class DispatchConfig:
    base_fare: float = 4.20
    per_km_fare: float = 1.35
    per_minute_fare: float = 0.42
    vehicle_cost_per_km: float = 0.42
    driver_time_cost_per_min: float = 0.20
    relocation_cost_per_km: float = 0.58
    rain_demand_lift: float = 0.32
    rain_pickup_penalty: float = 0.18
    wait_minutes: float = 15.0
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


def time_bucket_label(bucket: int) -> str:
    return {
        0: "night",
        1: "morning",
        2: "afternoon",
        3: "evening",
    }[int(bucket)]


def best_action(scores: pd.DataFrame) -> pd.Series:
    return scores[list(ACTION_COLUMNS.values())].idxmax(axis=1).map(
        {column: action for action, column in ACTION_COLUMNS.items()}
    )


def score_for_action(scores: pd.DataFrame, actions: pd.Series) -> pd.Series:
    values = np.empty(len(scores), dtype=float)
    for action, column in ACTION_COLUMNS.items():
        mask = actions == action
        values[mask.to_numpy()] = scores.loc[mask, column].to_numpy()
    return pd.Series(values, index=scores.index)


def build_dispatch_scenarios(
    predictions: pd.DataFrame,
    sample_size: int,
    random_state: int,
    config: DispatchConfig,
) -> pd.DataFrame:
    n = min(sample_size, len(predictions))
    sampled = predictions.sample(n=n, random_state=random_state).reset_index(drop=True)
    rng = np.random.default_rng(random_state)

    zone_codes = pd.factorize(sampled["Location"].astype(str), sort=True)[0]
    service_zone = zone_codes + 1
    zone_pressure = (zone_codes % 11) / 10.0

    time_buckets = rng.choice(
        np.array([0, 1, 2, 3]),
        size=n,
        p=np.array([0.18, 0.27, 0.23, 0.32]),
    )
    time_pressure = np.take(np.array([0.18, 0.58, 0.35, 0.72]), time_buckets)

    rain_probability = sampled["RainTomorrow_probability"].to_numpy()
    rain_lift = config.rain_demand_lift * np.power(rain_probability, 1.25)
    base_demand_index = (
        0.22
        + 0.38 * time_pressure
        + 0.25 * zone_pressure
        + rng.normal(0.0, 0.06, size=n)
    ).clip(0.05, 0.95)
    demand_index = (base_demand_index + rain_lift).clip(0.05, 1.25)

    weekday = sampled["Date"].dt.dayofweek.fillna(0).astype(int).to_numpy()
    trip_distance = rng.gamma(shape=2.2, scale=2.1, size=n).clip(0.8, 26.0)
    trip_duration = (trip_distance / rng.normal(28.0, 5.5, size=n).clip(12.0, 46.0) * 60.0)
    trip_duration = (trip_duration + rng.normal(4.0, 1.8, size=n)).clip(4.0, 90.0)

    pickup_distance_no_rain = rng.gamma(shape=1.8, scale=0.55, size=n).clip(0.1, 5.0)
    pickup_distance = (
        pickup_distance_no_rain * (1.0 + config.rain_pickup_penalty * rain_probability)
    ).clip(0.1, 6.5)

    current_request_fare = (
        config.base_fare
        + config.per_km_fare * trip_distance
        + config.per_minute_fare * trip_duration
    )
    future_request_fare = current_request_fare * rng.normal(1.04, 0.15, size=n).clip(0.65, 1.45)
    target_zone_demand = (
        base_demand_index
        + 0.18
        + 0.16 * time_pressure
        + 0.22 * np.power(rain_probability, 1.15)
        + rng.normal(0.0, 0.06, size=n)
    ).clip(0.05, 1.35)
    relocation_distance = rng.gamma(shape=2.0, scale=1.1, size=n).clip(0.5, 12.0)

    demo = pd.DataFrame(
        {
            "Date": sampled["Date"],
            "Location": sampled["Location"].astype(str),
            "service_zone": service_zone,
            "weekday": weekday,
            "time_bucket": time_buckets,
            "time_bucket_label": [time_bucket_label(x) for x in time_buckets],
            "rain_probability": rain_probability,
            "rain_risk_level": [rainfall_risk_label(x) for x in rain_probability],
            "base_demand_index": base_demand_index,
            "demand_index_with_rain": demand_index,
            "target_zone_demand_with_rain": target_zone_demand,
            "trip_distance_km": trip_distance,
            "trip_duration_min": trip_duration,
            "pickup_distance_km": pickup_distance,
            "relocation_distance_km": relocation_distance,
            "current_request_fare": current_request_fare,
            "future_request_fare": future_request_fare,
        }
    )
    return demo


def compute_action_scores(
    scenarios: pd.DataFrame,
    config: DispatchConfig,
    use_rain_signal: bool,
) -> pd.DataFrame:
    scores = scenarios.copy()

    if use_rain_signal:
        demand_index = scores["demand_index_with_rain"]
        target_demand = scores["target_zone_demand_with_rain"]
        pickup_distance = scores["pickup_distance_km"]
    else:
        demand_index = scores["base_demand_index"]
        target_demand = (scores["base_demand_index"] + 0.18 + 0.16 * (scores["time_bucket"] == 3)).clip(
            0.05,
            1.10,
        )
        pickup_distance = scores["pickup_distance_km"] / (
            1.0 + config.rain_pickup_penalty * scores["rain_probability"]
        )

    accept_operating_cost = (
        config.vehicle_cost_per_km * (pickup_distance + scores["trip_distance_km"])
        + config.driver_time_cost_per_min * scores["trip_duration_min"]
    )
    scores["score_accept"] = scores["current_request_fare"] - accept_operating_cost

    wait_success_probability = (0.25 + 0.65 * demand_index).clip(0.05, 0.98)
    expected_wait_time_cost = config.driver_time_cost_per_min * config.wait_minutes
    scores["score_wait"] = (
        wait_success_probability * scores["future_request_fare"] * 0.78
        - expected_wait_time_cost
    )

    relocation_success_probability = (0.18 + 0.70 * target_demand).clip(0.05, 0.98)
    relocation_cost = (
        config.relocation_cost_per_km * scores["relocation_distance_km"]
        + config.driver_time_cost_per_min * (scores["relocation_distance_km"] / 22.0 * 60.0)
    )
    scores["score_relocate"] = (
        relocation_success_probability * scores["future_request_fare"] * 0.86
        - relocation_cost
    )
    return scores[["score_wait", "score_accept", "score_relocate"]]


def apply_dispatch_policy(scenarios: pd.DataFrame, config: DispatchConfig) -> pd.DataFrame:
    rain_scores = compute_action_scores(scenarios, config, use_rain_signal=True)
    no_rain_scores = compute_action_scores(scenarios, config, use_rain_signal=False)

    policy_with_rain = best_action(rain_scores)
    policy_without_rain = best_action(no_rain_scores)
    reward_with_rain = score_for_action(rain_scores, policy_with_rain)
    reward_no_rain_policy_under_rain = score_for_action(rain_scores, policy_without_rain)

    result = scenarios.copy()
    for column in ACTION_COLUMNS.values():
        result[column] = rain_scores[column]
        result[f"no_rain_{column}"] = no_rain_scores[column]

    result["action_with_rain_signal"] = policy_with_rain
    result["action_without_rain_signal"] = policy_without_rain
    result["action_changed_by_rain_signal"] = (
        result["action_with_rain_signal"] != result["action_without_rain_signal"]
    )
    result["expected_reward_with_rain_policy"] = reward_with_rain
    result["expected_reward_no_rain_policy_under_rain"] = reward_no_rain_policy_under_rain
    result["rain_policy_reward_gain"] = (
        result["expected_reward_with_rain_policy"]
        - result["expected_reward_no_rain_policy_under_rain"]
    )

    numeric_cols = [
        "rain_probability",
        "base_demand_index",
        "demand_index_with_rain",
        "target_zone_demand_with_rain",
        "trip_distance_km",
        "trip_duration_min",
        "pickup_distance_km",
        "relocation_distance_km",
        "current_request_fare",
        "future_request_fare",
        "score_wait",
        "score_accept",
        "score_relocate",
        "no_rain_score_wait",
        "no_rain_score_accept",
        "no_rain_score_relocate",
        "expected_reward_with_rain_policy",
        "expected_reward_no_rain_policy_under_rain",
        "rain_policy_reward_gain",
    ]
    result[numeric_cols] = result[numeric_cols].round(4)
    return result


def summarize(result: pd.DataFrame) -> dict:
    action_counts = (
        result["action_with_rain_signal"].value_counts().rename_axis("action").reset_index(name="count")
    )
    changed = result["action_changed_by_rain_signal"].mean()
    overall = {
        "rows": int(len(result)),
        "action_changed_share": float(changed),
        "avg_reward_with_rain_policy": float(result["expected_reward_with_rain_policy"].mean()),
        "avg_reward_no_rain_policy_under_rain": float(
            result["expected_reward_no_rain_policy_under_rain"].mean()
        ),
        "avg_rain_policy_reward_gain": float(result["rain_policy_reward_gain"].mean()),
    }

    by_risk = (
        result.groupby("rain_risk_level", observed=True)
        .agg(
            scenarios=("rain_probability", "size"),
            avg_rain_probability=("rain_probability", "mean"),
            action_changed_share=("action_changed_by_rain_signal", "mean"),
            avg_reward_with_rain_policy=("expected_reward_with_rain_policy", "mean"),
            avg_reward_no_rain_policy_under_rain=(
                "expected_reward_no_rain_policy_under_rain",
                "mean",
            ),
            avg_rain_policy_reward_gain=("rain_policy_reward_gain", "mean"),
        )
        .reset_index()
    )
    order = {"low": 0, "moderate": 1, "high": 2, "very_high": 3}
    by_risk["_order"] = by_risk["rain_risk_level"].map(order)
    by_risk = by_risk.sort_values("_order").drop(columns="_order")

    by_action_and_risk = (
        result.groupby(["rain_risk_level", "action_with_rain_signal"], observed=True)
        .size()
        .reset_index(name="count")
    )

    return {
        "overall": {key: round(value, 4) for key, value in overall.items()},
        "action_counts": action_counts.to_dict(orient="records"),
        "by_risk": by_risk.round(4).to_dict(orient="records"),
        "by_action_and_risk": by_action_and_risk.to_dict(orient="records"),
    }


def write_outputs(
    result: pd.DataFrame,
    summary: dict,
    output_path: str | Path,
    report_path: str | Path,
    summary_path: str | Path,
    config: DispatchConfig,
) -> None:
    output_path = resolve_path(output_path)
    report_path = resolve_path(report_path)
    summary_path = resolve_path(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    payload = {"config": asdict(config), **summary}
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    by_risk = pd.DataFrame(summary["by_risk"])
    action_counts = pd.DataFrame(summary["action_counts"])
    changed_rows = result[result["action_changed_by_rain_signal"]].sort_values(
        "rain_policy_reward_gain",
        ascending=False,
    )
    top_changed = changed_rows.head(20)

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Rainfall-aware taxi dispatch demo</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .note {{ color: #555; max-width: 980px; line-height: 1.45; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f4f6f8; }}
    .metric {{ display: inline-block; margin: 12px 24px 12px 0; }}
    .metric b {{ display: block; font-size: 22px; }}
  </style>
</head>
<body>
  <h1>Rainfall-aware taxi dispatch demo</h1>
  <p class="note">
    This is a scenario demo for driver acceptance and repositioning decisions.
    The rain probability is the empirical input from this repository. Trip
    request variables are simulated to make the decision interface explicit.
    The defensible scientific claim is limited to workflow feasibility until
    real taxi request logs and weather joins are used.
  </p>
  <div class="metric"><span>Rows</span><b>{summary["overall"]["rows"]}</b></div>
  <div class="metric"><span>Action changed by rain signal</span><b>{summary["overall"]["action_changed_share"]:.1%}</b></div>
  <div class="metric"><span>Avg rain-policy reward gain</span><b>{summary["overall"]["avg_rain_policy_reward_gain"]:.2f} {config.currency}</b></div>

  <h2>Action counts</h2>
  {action_counts.to_html(index=False, escape=True)}

  <h2>Summary by rainfall risk</h2>
  {by_risk.to_html(index=False, escape=True)}

  <h2>Largest action changes driven by rainfall signal</h2>
  {top_changed.to_html(index=False, escape=True)}
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")


def run_demo(args: argparse.Namespace) -> dict:
    config = DispatchConfig()
    predictions = load_rainfall_predictions(args.predictions)
    scenarios = build_dispatch_scenarios(
        predictions=predictions,
        sample_size=args.sample_size,
        random_state=args.random_state,
        config=config,
    )
    result = apply_dispatch_policy(scenarios, config)
    summary = summarize(result)
    write_outputs(
        result=result,
        summary=summary,
        output_path=args.output,
        report_path=args.report,
        summary_path=args.summary,
        config=config,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a rainfall-aware taxi driver dispatch decision demo."
    )
    parser.add_argument(
        "--predictions",
        default=str(DEFAULT_INPUT),
        help="CSV produced by predict.py with RainTomorrow_probability.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output dispatch CSV.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Output HTML report.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Output JSON summary.")
    parser.add_argument("--sample-size", type=int, default=2500, help="Number of scenarios.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    summary = run_demo(parse_args())
    print("Taxi dispatch demo complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
