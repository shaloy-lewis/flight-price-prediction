"""Training-partition EDA and full-source quality reporting; no notebook state required."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.components.data_ingestion import holdout_split, leakage_groups, load_dataset
from src.config import ExperimentConfig
from src.reporting import save_figure, write_json


def run_analysis(data_dir: Path, output: Path, config: ExperimentConfig) -> dict:
    dataset = load_dataset(data_dir)
    frame = dataset.frame
    groups = leakage_groups(frame)
    train_idx, _ = holdout_split(frame, groups, config.seed, config.holdout_fraction)
    train = frame.iloc[train_idx].copy()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "data_quality.json", dataset.quality)
    write_json(output / "sources.json", dataset.sources)
    source = f"Source: Kaggle original raw files; training partition only (n={len(train):,}). Fares in INR."
    findings = {
        "partition": "training only",
        "rows": len(train),
        "scope": "Cross-sectional associations; no booking timestamps.",
        "price_by_cabin": train.groupby("cabin_class")
        .price.agg(["count", "median", "mean"])
        .to_dict("index"),
        "price_by_airline_and_cabin": train.groupby(["airline", "cabin_class"])
        .price.agg(["count", "median"])
        .reset_index()
        .to_dict("records"),
        "price_by_route_and_cabin": train.groupby(
            ["source_city", "destination_city", "cabin_class"]
        )
        .price.agg(["count", "median"])
        .reset_index()
        .to_dict("records"),
        "numeric_correlations": train[["duration_minutes", "stops", "price"]].corr().to_dict(),
        "travel_date_coverage": "11 February to 31 March 2022; travel dates do not establish observation chronology.",
    }
    write_json(output / "eda_summary.json", findings)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cabin, group in train.groupby("cabin_class"):
        axes[0].hist(group.price, bins=50, alpha=0.5, label=cabin)
        axes[1].hist(np.log(group.price), bins=50, alpha=0.5, label=cabin)
    for axis in axes:
        axis.set_ylabel("Number of offers")
        axis.legend()
    axes[0].set(xlabel="Observed fare (INR)", title="Fare distribution by cabin")
    axes[1].set(xlabel="Natural log of fare (INR)", title="Log-fare distribution by cabin")
    save_figure(fig, output / "figures/price_distribution.png", source)
    pivot = train.pivot_table(
        index="airline", columns="cabin_class", values="price", aggfunc="median"
    )
    fig, axis = plt.subplots(figsize=(9, 4))
    pivot.plot.bar(ax=axis)
    axis.set(
        title="Median observed fare by airline and cabin",
        xlabel="Historical airline",
        ylabel="Median fare (INR)",
    )
    axis.tick_params(axis="x", rotation=25)
    save_figure(fig, output / "figures/airline_cabin.png", source)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sample = train.sample(min(4000, len(train)), random_state=config.seed)
    for cabin, group in sample.groupby("cabin_class"):
        axes[0].scatter(group.duration_minutes / 60, group.price, s=5, alpha=0.3, label=cabin)
    axes[0].set(
        title="Duration and fare (sample)",
        xlabel="Total itinerary duration (hours)",
        ylabel="Observed fare (INR)",
    )
    axes[0].legend()
    train.pivot_table(
        index="stops", columns="cabin_class", values="price", aggfunc="median"
    ).plot.bar(ax=axes[1])
    axes[1].set(
        title="Stops and median fare", xlabel="Stops (2 = two or more)", ylabel="Median fare (INR)"
    )
    save_figure(fig, output / "figures/duration_stops.png", source)
    fig, axis = plt.subplots(figsize=(10, 4))
    route = (
        train[train.cabin_class == "economy"]
        .groupby(["source_city", "destination_city"])
        .price.median()
        .unstack()
    )
    image = axis.imshow(route.to_numpy(), cmap="viridis")
    axis.set_xticks(range(len(route.columns)), route.columns, rotation=25)
    axis.set_yticks(range(len(route.index)), route.index)
    axis.set(title="Economy route median fares", xlabel="Destination", ylabel="Origin")
    fig.colorbar(image, ax=axis, label="Median observed fare (INR)")
    save_figure(fig, output / "figures/routes.png", source)
    fig, axis = plt.subplots(figsize=(10, 4))
    train.pivot_table(
        index="travel_date", columns="cabin_class", values="price", aggfunc="median"
    ).plot(ax=axis)
    axis.set(
        title="Fare by travel date (not an observation-time series)",
        xlabel="Travel date in 2022",
        ylabel="Median observed fare (INR)",
    )
    save_figure(fig, output / "figures/travel_dates.png", source)
    return findings
