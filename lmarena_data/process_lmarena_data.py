"""Plot the committed provider summary or aggregate user-supplied Arena snapshots."""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROVIDERS = {
    "Google": ("Google",),
    "OpenAI": ("OpenAI",),
    "Anthropic": ("Anthropic",),
    "DeepSeek": ("DeepSeek",),
    "xAI": ("xAI",),
    "Mistral AI": ("Mistral AI", "Mistral"),
}
COLORS = ["#2563EB", "#087F6D", "#A65C32", "#7C3AED", "#455468", "#C03965"]


def validate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject malformed summaries; preserve gaps instead of filling them."""
    required = {"Date", *PROVIDERS}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing summary columns: {', '.join(sorted(missing))}")
    frame = frame[["Date", *PROVIDERS]].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    if frame.empty or frame["Date"].isna().any() or frame["Date"].duplicated().any():
        raise ValueError("Summary must contain unique, non-empty dates.")
    for provider in PROVIDERS:
        frame[provider] = pd.to_numeric(frame[provider], errors="raise")
        values = frame[provider].dropna()
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite scores for {provider}.")
    if not frame[list(PROVIDERS)].notna().any().any():
        raise ValueError("Summary has no observed scores.")
    return frame.sort_values("Date").reset_index(drop=True)


def aggregate_snapshots(directory: Path) -> pd.DataFrame:
    """Take each provider's maximum score in each monthly CSV; never forward-fill."""
    files = sorted(directory.glob("monthly_*.csv"))
    if not files:
        raise ValueError(f"No monthly_YYYY.MM.csv files in {directory}.")
    rows = []
    for path in files:
        match = re.fullmatch(r"monthly_(\d{4})\.(\d{2})\.csv", path.name)
        if not match:
            raise ValueError(f"Unexpected snapshot filename: {path.name}")
        date = pd.Timestamp(year=int(match[1]), month=int(match[2]), day=1)
        frame = pd.read_csv(path)
        if not {"organization", "arena_score"}.issubset(frame.columns):
            raise ValueError(f"{path.name} requires organization and arena_score columns.")
        if frame.empty:
            raise ValueError(f"Empty snapshot: {path.name}")
        frame["arena_score"] = pd.to_numeric(frame["arena_score"], errors="raise")
        if not np.isfinite(frame["arena_score"]).all():
            raise ValueError(f"Missing or non-finite scores in {path.name}.")
        organizations = frame["organization"].astype("string").str.strip()
        row = {"Date": date}
        for provider, aliases in PROVIDERS.items():
            row[provider] = frame.loc[organizations.isin(aliases), "arena_score"].max()
        rows.append(row)
    result = validate_summary(pd.DataFrame(rows))
    # Missing months remain explicit gaps in the plotted time series.
    months = pd.date_range(result.Date.min(), result.Date.max(), freq="MS")
    return result.set_index("Date").reindex(months).rename_axis("Date").reset_index()


def plot_summary(frame: pd.DataFrame, output: Path, *, archived: bool) -> None:
    """Use common scales and separate panels to keep close scores readable."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex=True, sharey=True)
    fig.patch.set_facecolor("#FAFAF8")
    fig.subplots_adjust(left=.065, right=.97, top=.77, bottom=.19, hspace=.42, wspace=.16)
    fig.text(.065, .94, "LM ARENA  /  PROVIDER TRENDS", fontsize=11, color="#68717C", weight="bold")
    fig.text(.065, .882, "How did leading model providers compare?", fontsize=25, weight="bold", color="#17212B")
    period = f"{frame.Date.min():%b %Y} – {frame.Date.max():%b %Y}"
    fig.text(.065, .833, f"Highest recorded Arena score per provider · {period}", fontsize=12, color="#59636E")
    values = frame[list(PROVIDERS)]
    lower, upper = values.min().min() - 25, values.max().max() + 35
    for ax, (provider, color) in zip(axes.flat, zip(PROVIDERS, COLORS)):
        ax.set_facecolor("#FAFAF8")
        ax.plot(frame.Date, frame[provider], color=color, linewidth=2.5, marker="o", markersize=4)
        ax.set_title(provider, loc="left", fontsize=14, fontweight="bold", color=color, pad=13)
        ax.set_ylim(lower, upper)
        ax.grid(axis="y", color="#E1E5E8", linewidth=.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0, labelsize=9, colors="#68717C", pad=8)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        last = frame.iloc[-1][provider]
        if pd.notna(last):
            ax.annotate(f"{last:,.0f}", (frame.Date.iloc[-1], last), xytext=(-3, 10),
                        textcoords="offset points", ha="right", color=color, weight="bold", fontsize=11)
    fig.text(.02, .48, "Arena score · shared scale", rotation=90, va="center", fontsize=10, color="#68717C")
    note = "Archived summary; original monthly snapshots are not included." if archived else "Computed from supplied monthly snapshots; missing observations remain gaps."
    fig.text(.065, .10, note, fontsize=10, color="#59636E")
    fig.text(.065, .065, "Source: LM Arena text leaderboard · Scores reflect human preferences; model coverage and voting populations can change.", fontsize=9, color="#68717C")
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "lmarena_top_scores_by_provider.csv")
    parser.add_argument("--snapshots-dir", type=Path, help="Rebuild from monthly_YYYY.MM.csv files instead of the summary.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    try:
        frame = aggregate_snapshots(args.snapshots_dir) if args.snapshots_dir else validate_summary(pd.read_csv(args.summary))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output_dir / "provider_scores.csv", index=False, date_format="%Y-%m-%d")
        plot_summary(frame, args.output_dir / "provider_trends.png", archived=not bool(args.snapshots_dir))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved provider_scores.csv and provider_trends.png to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
