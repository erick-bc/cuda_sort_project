import csv
import io
import re
import argparse
from collections import defaultdict

# AI Disclosure:
# This code was made in part
# with Google Gemini. It basically
# helps print out the results of throughput
# with Nsight output.

RADIX_START_KERNEL = "DeviceRadixSortHistogramKernel"
RADIX_SORT_PREFIX  = "DeviceRadixSort"

KERNEL_MAP = [
    ("block_sort_kernel",      "Student Merge"),
    ("merge_stage_kernel_v12", "Student Merge"),
    ("padded_kernel",          "Bitonic"),
    ("bitonic_sort_gpu",       "Bitonic"),
]

DISPLAY_NAMES = {
    "Thrust Radix":  "Thrust Sort (Radix)",
    "Thrust Merge":  "Thrust Sort (Merge)",
    "Student Merge": "Merge",
    "Bitonic":       "Bitonic",
}

ALGO_ORDER = ["Thrust Radix", "Thrust Merge", "Student Merge", "Bitonic"]

_SIZE_RE = re.compile(r'Sorting for size:\s*(\d+)')


def load_profile(filepath):
    """
    Read a raw ncu --csv output file that may contain benchmark stdout and
    ==PROF== messages before the actual CSV data.

    Returns (sizes, reader) where:
      sizes  – list of int, one per "Sorting for size: N" line found before
               the CSV header (order matches the benchmark's size loop)
      reader – csv.DictReader positioned at the first data row
    """
    with open(filepath, newline="", encoding="utf-8") as f:
        lines = f.readlines()

    sizes = []
    csv_start = None
    for i, line in enumerate(lines):
        m = _SIZE_RE.search(line)
        if m:
            sizes.append(int(m.group(1)))
        if csv_start is None and line.lstrip().startswith('"ID"'):
            csv_start = i
            break

    if csv_start is None:
        raise SystemExit(
            "ERROR: No CSV header found in file. "
            "Is this a valid ncu --csv output?"
        )

    reader = csv.DictReader(io.StringIO("".join(lines[csv_start:])))
    return sizes, reader


def classify_kernel(kernel_name: str, radix_inv: int) -> str:
    lower = kernel_name.lower()
    if RADIX_SORT_PREFIX.lower() in lower:
        # Within each size group radix_inv is reset so 1=Thrust Radix, 2=Thrust Merge
        return "Thrust Radix" if radix_inv <= 1 else "Thrust Merge"
    for substring, label in KERNEL_MAP:
        if substring.lower() in lower:
            return label
    return "Unknown"


def parse_value(val: str) -> float:
    try:
        return float(val.replace(",", "").strip())
    except ValueError:
        return 0.0


def _empty_group():
    return {
        "bytes_read":  defaultdict(float),
        "bytes_write": defaultdict(float),
        "time_ns":     defaultdict(float),
    }


def parse_metrics(reader):
    """
    Walk the CSV rows and split them into per-size groups.

    Each size run launches exactly 2 Thrust sorts (2 HistogramKernel IDs).
    When a third HistogramKernel is encountered the previous size group is
    complete and a new one begins.  Returns a list of groups, one per size.
    """
    groups   = []
    current  = _empty_group()
    radix_inv    = 0   # resets to 1 at the start of each new size group
    last_seen_id = None

    for row in reader:
        kernel    = row.get("Kernel Name", "")
        kernel_id = row.get("ID", "")
        metric    = row.get("Metric Name", "").strip()
        unit      = row.get("Metric Unit", "").strip()
        value     = row.get("Metric Value", "0")

        if not kernel or not metric:
            continue

        if kernel_id != last_seen_id:
            last_seen_id = kernel_id
            if RADIX_START_KERNEL.lower() in kernel.lower():
                radix_inv += 1
                if radix_inv > 2:
                    # 3rd Histogram = Thrust 1 of the next size
                    groups.append(current)
                    current   = _empty_group()
                    radix_inv = 1   # this row belongs to the new group's Thrust Radix

        algo = classify_kernel(kernel, radix_inv)
        val  = parse_value(value)

        if metric == "dram__bytes_read.sum" and unit == "byte":
            current["bytes_read"][algo] += val
        elif metric == "dram__bytes_write.sum" and unit == "byte":
            current["bytes_write"][algo] += val
        elif metric == "gpu__time_duration.sum" and unit == "ns":
            current["time_ns"][algo] += val

    groups.append(current)   # commit the final (or only) group
    return groups


def print_table(size, group, peak_gbps):
    SEP  = "=" * 57
    DASH = "-" * 57
    print(SEP)
    print(f"Hardware Throughput for size: {size:,}")
    print(DASH)
    print(f"{'Kernel':<20} {'GB/s':>15} {'Peak bandwidth %':>18}")
    print(DASH)
    for algo in ALGO_ORDER:
        total_bytes = group["bytes_read"][algo] + group["bytes_write"][algo]
        total_ns    = group["time_ns"][algo]
        if total_ns > 0:
            gbps     = total_bytes / total_ns
            pct_peak = (gbps / peak_gbps) * 100.0
        else:
            gbps = pct_peak = 0.0
        display = DISPLAY_NAMES.get(algo, algo)
        print(f"{display:<20} {gbps:>15.2f} {pct_peak:>17.2f}%")
    print(SEP)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Print hardware throughput table from an ncu CSV profile."
    )
    parser.add_argument("csv_file", help="Path to profile_out.csv")
    parser.add_argument("--peak-gbps", type=float, default=760.0,
                        help="Peak DRAM bandwidth in GB/s (default: 760 for RTX 3080)")
    args = parser.parse_args()

    sizes, reader = load_profile(args.csv_file)
    groups = parse_metrics(reader)

    if len(sizes) != len(groups):
        print(f"Warning: {len(sizes)} size(s) in benchmark output but "
              f"{len(groups)} group(s) in CSV — using CSV group count.")
        # Pad with 0 if benchmark output was missing some sizes
        while len(sizes) < len(groups):
            sizes.append(0)
        sizes = sizes[:len(groups)]

    for size, group in zip(sizes, groups):
        print_table(size, group, args.peak_gbps)


if __name__ == "__main__":
    main()
