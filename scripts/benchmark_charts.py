"""Draw the forget-versus-rebuild charts from docs/benchmark/results.json.

Run: python scripts/benchmark_charts.py
Writes a light and a dark PNG of each chart next to the results.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

FOLDER = Path(__file__).resolve().parent.parent / "docs" / "benchmark"
TRAINED_LENGTH = 512

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#1f1f1e", "muted": "#6b6b68", "grid": "#e6e5e1",
        "band": "#f0efeb", "subtract": "#2a78d6", "rebuild": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ecebe7", "muted": "#a3a29e", "grid": "#33332f",
        "band": "#252523", "subtract": "#3987e5", "rebuild": "#d95926",
    },
}


def milliseconds(value, _):
    if value >= 1000:
        return f"{value / 1000:.1f} s".replace(".0 s", " s")
    return f"{value:.0f} ms"


def style(ax, theme):
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=10, length=0, pad=6)
    ax.tick_params(which="minor", length=0)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def figure(theme):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    fig.patch.set_facecolor(theme["surface"])
    style(ax, theme)
    return fig, ax


def title(fig, theme, heading, subheading):
    fig.text(0.07, 0.95, heading, fontsize=14, weight="bold", color=theme["ink"], va="top")
    fig.text(0.07, 0.885, subheading, fontsize=10, color=theme["muted"], va="top")


def length_chart(results, theme, path):
    rows = results["lengths"]
    lengths = [row["length"] for row in rows]
    fig, ax = figure(theme)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvspan(TRAINED_LENGTH, lengths[-1] * 1.6, color=theme["band"], zorder=0, linewidth=0)
    ax.text(
        TRAINED_LENGTH * 1.08, 230, "Longer than training conversations:\ntiming and state only, not recall",
        fontsize=9, color=theme["muted"], va="top",
    )
    for key, label in (("rebuild", "Rebuild"), ("subtract", "Subtract")):
        values = [row[f"{key}_ms"] for row in rows]
        ax.plot(lengths, values, color=theme[key], linewidth=2, marker="o", markersize=7,
                markeredgecolor=theme["surface"], markeredgewidth=2, zorder=3)
        ax.annotate(f"{label}  {milliseconds(values[-1], None)}", (lengths[-1], values[-1]),
                    xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=10, color=theme["ink"])
    ax.set_xlim(lengths[0] / 1.2, lengths[-1] * 1.6)
    ax.set_ylim(15, 12000)
    ax.set_xticks(lengths)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax.yaxis.set_major_formatter(FuncFormatter(milliseconds))
    ax.set_xlabel("Conversation length (bytes)", color=theme["muted"], fontsize=10)
    title(fig, theme, "Time to forget a card number",
          f"19 bytes erased, median of {rows[0]['cases'] * 3} runs, laptop CPU at {results['threads']} threads")
    fig.subplots_adjust(left=0.1, right=0.83, top=0.8, bottom=0.14)
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def size_chart(results, theme, path):
    rows = results["erased_size"]
    sizes = [row["bytes_erased"] for row in rows]
    values = [row["subtract_ms"] for row in rows]
    reference = next(row for row in results["lengths"] if row["length"] > 1600)
    fig, ax = figure(theme)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(reference["rebuild_ms"], color=theme["rebuild"], linewidth=2, linestyle=(0, (4, 3)))
    ax.text(sizes[0], reference["rebuild_ms"] * 1.15,
            f"Rebuild, whatever the size: about {milliseconds(reference['rebuild_ms'], None)}",
            fontsize=10, color=theme["ink"])
    ax.plot(sizes, values, color=theme["subtract"], linewidth=2, marker="o", markersize=7,
            markeredgecolor=theme["surface"], markeredgewidth=2, zorder=3)
    ax.annotate(f"Subtract  {milliseconds(values[-1], None)}", (sizes[-1], values[-1]),
                xytext=(10, 0), textcoords="offset points", va="center", fontsize=10, color=theme["ink"])
    ax.set_xlim(sizes[0] / 1.3, sizes[-1] * 1.4)
    ax.set_ylim(5, 5000)
    ax.set_xticks(sizes)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax.yaxis.set_major_formatter(FuncFormatter(milliseconds))
    ax.set_xlabel("Bytes erased", color=theme["muted"], fontsize=10)
    title(fig, theme, "Subtract costs grow with what is erased",
          f"In a {reference['length']:,}-byte conversation, median of 5 runs each")
    fig.subplots_adjust(left=0.1, right=0.83, top=0.8, bottom=0.14)
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def main():
    results = json.loads((FOLDER / "results.json").read_text(encoding="utf-8"))
    for name, theme in THEMES.items():
        length_chart(results, theme, FOLDER / f"time_by_length_{name}.png")
        size_chart(results, theme, FOLDER / f"time_by_erased_size_{name}.png")


if __name__ == "__main__":
    main()
