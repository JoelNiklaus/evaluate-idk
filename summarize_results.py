import json
from pathlib import Path
from collections import defaultdict

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
