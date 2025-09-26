import json
from pathlib import Path
from collections import defaultdict
import os

RESULTS_ROOT = Path(__file__).parent / "results" / "results"

# Collect all JSON result files recursively
json_files = sorted(RESULTS_ROOT.glob("**/results_*.json"))

if not json_files:
    print("No results found.")
    raise SystemExit(0)

# Aggregation per model
# Structure: model_name -> list of dicts with metrics
per_model = defaultdict(list)

for jf in json_files:
    try:
        with open(jf, "r") as f:
            data = json.load(f)
    except Exception:
        continue

    full_model_name = (
        data.get("config_general", {})
        .get("model_config", {})
        .get("model_name")
        or data.get("config_general", {}).get("model_name")
        or str(jf.parent)
    )
    display_model = full_model_name.split("/")[-1] if full_model_name else str(jf.parent).split("/")[-1]

    results = data.get("results", {})
    # Prefer the overall aggregate if present
    task_or_all = results.get("all")
    if not task_or_all:
        # If not present, pick the first task entry
        if results:
            task_or_all = next(iter(results.values()))
        else:
            task_or_all = {}

    trad = task_or_all.get("trad_score")
    trad_stderr = task_or_all.get("trad_score_stderr")
    idk_score = task_or_all.get("idk_score")
    idk_score_stderr = task_or_all.get("idk_score_stderr")
    # Accept only idk_freq (disregard idk_frequency)
    idk_freq = task_or_all.get("idk_freq")
    idk_freq_stderr = task_or_all.get("idk_freq_stderr")

    extract_fail = task_or_all.get("extract_fail")
    extract_fail_stderr = task_or_all.get("extract_fail_stderr")

    per_model[display_model].append(
        {
            "trad_score": trad,
            "trad_score_stderr": trad_stderr,
            "idk_score": idk_score,
            "idk_score_stderr": idk_score_stderr,
            "idk_freq": idk_freq,
            "idk_freq_stderr": idk_freq_stderr,
            "extract_fail": extract_fail,
            "extract_fail_stderr": extract_fail_stderr,
            "path": str(jf),
        }
    )

# For each model, keep the most recent or best entry; here we choose the latest by filename
rows = []
for model, entries in per_model.items():
    entry = sorted(entries, key=lambda e: e["path"])[-1]
    rows.append(
        (
            model,
            entry.get("trad_score"),
            entry.get("idk_score"),
            entry.get("idk_freq"),
            entry.get("extract_fail"),
            entry.get("trad_score_stderr"),
            entry.get("idk_score_stderr"),
            entry.get("idk_freq_stderr"),
            entry.get("extract_fail_stderr"),
        )
    )

# Build table cells first, then compute column widths to align pipes
header = ["Model", "trad_score ± se", "idk_score ± se", "idk_freq ± se", "extract_fail ± se"]

def fmt_val_se(v, se):
    def fmt_num(n):
        if isinstance(n, (int, float)):
            return f"{n*100:.2f}"
        return None

    v_str = fmt_num(v)
    if v_str is None:
        return "-" if v is None else str(v)

    se_str = fmt_num(se)
    return f"{v_str} ± {se_str}" if se_str is not None else v_str

table_rows = []
for (
    model,
    trad,
    idk_s,
    idk_f,
    ext_f,
    trad_se,
    idk_se,
    idk_f_se,
    ext_f_se,
) in rows:
    table_rows.append([
        str(model),
        fmt_val_se(trad, trad_se),
        fmt_val_se(idk_s, idk_se),
        fmt_val_se(idk_f, idk_f_se),
        fmt_val_se(ext_f, ext_f_se),
    ])

# Compute widths per column
col_widths = [len(h) for h in header]
for r in table_rows:
    for i, cell in enumerate(r):
        if len(cell) > col_widths[i]:
            col_widths[i] = len(cell)

def pad(cell: str, width: int, align: str) -> str:
    if align == "left":
        return cell.ljust(width)
    return cell.rjust(width)

# Alignment: left for Model, right for numeric columns
aligns = ["left", "right", "right", "right", "right"]

# Header
header_line = "| " + " | ".join(pad(h, col_widths[i], aligns[i]) for i, h in enumerate(header)) + " |"
print(header_line)

# Alignment row with pipes aligned
align_line_parts = []
for i, w in enumerate(col_widths):
    if aligns[i] == "left":
        # left align: :-----
        if w >= 3:
            align_line_parts.append(":" + "-" * (w - 1))
        else:
            align_line_parts.append(":" + "-" * 2)
    else:
        # right align: -----:
        if w >= 3:
            align_line_parts.append("-" * (w - 1) + ":")
        else:
            align_line_parts.append("-" * 2 + ":")
align_line = "| " + " | ".join(align_line_parts) + " |"
print(align_line)

# Data rows
for r in table_rows:
    line = "| " + " | ".join(pad(r[i], col_widths[i], aligns[i]) for i in range(len(r))) + " |"
    print(line)

# --- Visualization: Bar chart of performance drop (trad vs idk) ---
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

if plt is not None:
    # Filter to selected models only
    desired_models = {
        "gemini-2.5-pro": "Gemini 2.5 Pro",
        "gpt-5": "GPT-5",
        "gpt-5-nano": "GPT-5 Nano",
        "gemini-2.5-flash": "Gemini 2.5 Flash",
        "gpt-4.1": "GPT-4.1",
    }

    # Build mapping from model -> metrics
    latest_by_model = {m: None for m in desired_models.keys()}
    for (
        model,
        trad,
        idk_s,
        idk_f,
        _ext_f,
        trad_se,
        idk_se,
        idk_f_se,
        _ext_f_se,
    ) in rows:
        if model in latest_by_model:
            latest_by_model[model] = {
                "trad": trad,
                "idk": idk_s,
                "trad_se": trad_se,
                "idk_se": idk_se,
                "idk_freq": idk_f,
                "idk_freq_se": idk_f_se,
            }

    # Keep only present models, preserving a preferred order
    ordered_keys = [k for k in ["gemini-2.5-pro", "gpt-5", "gpt-5-nano", "gemini-2.5-flash", "gpt-4.1"] if latest_by_model.get(k) is not None]
    if ordered_keys:
        x_labels = [desired_models[k] for k in ordered_keys]
        trad_vals = [latest_by_model[k]["trad"] * 100 if latest_by_model[k]["trad"] is not None else None for k in ordered_keys]
        idk_vals = [latest_by_model[k]["idk"] * 100 if latest_by_model[k]["idk"] is not None else None for k in ordered_keys]
        trad_se_vals = [latest_by_model[k]["trad_se"] * 100 if latest_by_model[k]["trad_se"] is not None else None for k in ordered_keys]
        idk_se_vals = [latest_by_model[k]["idk_se"] * 100 if latest_by_model[k]["idk_se"] is not None else None for k in ordered_keys]
        idk_freq_vals = [latest_by_model[k]["idk_freq"] * 100 if latest_by_model[k]["idk_freq"] is not None else None for k in ordered_keys]
        idk_freq_se_vals = [latest_by_model[k]["idk_freq_se"] * 100 if latest_by_model[k]["idk_freq_se"] is not None else None for k in ordered_keys]

        # Replace None with 0 for plotting, but we'll annotate as N/A
        trad_plot = [v if v is not None else 0.0 for v in trad_vals]
        idk_plot = [v if v is not None else 0.0 for v in idk_vals]
        trad_yerr = [se if se is not None else 0.0 for se in trad_se_vals]
        idk_yerr = [se if se is not None else 0.0 for se in idk_se_vals]

        x = range(len(x_labels))
        width = 0.36

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(
            [i - width/2 for i in x],
            trad_plot,
            width=width,
            label="Traditional score",
            color="#4e79a7",
            yerr=trad_yerr,
            capsize=3,
            error_kw={"elinewidth": 1, "capthick": 1, "ecolor": "#2f5c8a"},
        )
        ax.bar(
            [i + width/2 for i in x],
            idk_plot,
            width=width,
            label="IDK score",
            color="#f28e2b",
            yerr=idk_yerr,
            capsize=3,
            error_kw={"elinewidth": 1, "capthick": 1, "ecolor": "#b3621e"},
        )

        # Compute y-limit to accommodate error bars and labels
        bars_max = 0.0
        if trad_plot:
            bars_max = max(bars_max, max(tp + ty for tp, ty in zip(trad_plot, trad_yerr)))
        if idk_plot:
            bars_max = max(bars_max, max(ip + iy for ip, iy in zip(idk_plot, idk_yerr)))
        y_upper = max(100.0, bars_max * 1.12)
        ax.set_ylim(0, y_upper)
        label_offset = max(0.5, 0.02 * y_upper)

        # Annotate values above error bars
        for i, v in enumerate(trad_vals):
            if v is None:
                ax.text(i - width/2, 0.5, "N/A", ha="center", va="bottom", fontsize=9, rotation=90)
            else:
                ax.text(
                    i - width/2,
                    trad_plot[i] + trad_yerr[i] + label_offset,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        for i, v in enumerate(idk_vals):
            if v is None:
                ax.text(i + width/2, 0.5, "N/A", ha="center", va="bottom", fontsize=9, rotation=90)
            else:
                ax.text(
                    i + width/2,
                    idk_plot[i] + idk_yerr[i] + label_offset,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # Secondary axis: IDK frequency line with error bars
        ax2 = ax.twinx()
        idk_freq_plot = [v if v is not None else 0.0 for v in idk_freq_vals]
        idk_freq_yerr = [se if se is not None else 0.0 for se in idk_freq_se_vals]
        ax2.errorbar(
            list(x),
            idk_freq_plot,
            yerr=idk_freq_yerr,
            fmt="-o",
            color="#59a14f",
            linewidth=2,
            markersize=5,
            capsize=3,
            label="IDK frequency",
        )
        for i, v in enumerate(idk_freq_vals):
            if v is None:
                ax2.text(i, 2.0, "N/A", ha="center", va="bottom", fontsize=9, rotation=90, color="#59a14f")

        # Title emphasizes drop
        ax.set_title("GPQA Diamond Performance drop: Traditional vs IDK score (higher is better)")
        ax.set_ylabel("Score (%)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(x_labels)
        # y-limits already set above to account for labels; keep as-is
        ax2.set_ylabel("IDK frequency (%)")
        ax2.set_ylim(0, 100)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.5)

        # Save
        fig_dir = Path(__file__).parent / "results" / "figures"
        os.makedirs(fig_dir, exist_ok=True)
        out_path = fig_dir / "score_drop_barchart.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        print(f"Saved bar chart to: {out_path}")
else:
    print("matplotlib not available; skipping chart generation.")
