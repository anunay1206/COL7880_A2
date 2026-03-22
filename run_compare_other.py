#!/usr/bin/env python3
"""
Compares your implementation vs the other KNN implementation.
Folder structure:
  output_other/
    knn.txt
    time.txt
    comparison_report.txt
"""
import subprocess, os, sys, re, time
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
YOUR_BIN   = "./a2"
OTHER_SRC  = "other_knn.cu"
OTHER_BIN  = "./other_knn"
OTHER_DIR  = "output_other"
INPUT      = "input.txt"
REPORT     = os.path.join(OTHER_DIR, "comparison_report.txt")

os.makedirs(OTHER_DIR, exist_ok=True)

# ── Check input ───────────────────────────────────────────────────────────────
if not os.path.exists(INPUT):
    print(f"ERROR: {INPUT} not found."); sys.exit(1)

with open(INPUT) as f:
    N = int(f.readline())
    k = int(f.readline())
    T = int(f.readline())

print(f"Input: N={N}  k={k}  T={T}")
print("="*60)

# ── Build other_knn if needed ─────────────────────────────────────────────────
if not os.path.exists(OTHER_BIN) or \
   os.path.getmtime(OTHER_SRC) > os.path.getmtime(OTHER_BIN):
    print("Building other_knn...")
    r = subprocess.run(
        ["nvcc", "-O3", "-Xcompiler", "-fopenmp",
         "-o", OTHER_BIN, OTHER_SRC],
        capture_output=True, text=True)
    if r.returncode != 0:
        print("Build failed:\n", r.stderr); sys.exit(1)
    print("Built other_knn OK")

# ── Run YOUR implementation ───────────────────────────────────────────────────
print("\n[1/2] Running YOUR implementation (a2)...")
t0 = time.time()
r = subprocess.run([YOUR_BIN, INPUT], capture_output=True, text=True)
your_wall = (time.time() - t0) * 1000

if r.returncode != 0:
    print("Your a2 failed:", r.stderr); sys.exit(1)

your_output = r.stdout + r.stderr
print(your_output.strip())

# Parse your KNN time
your_knn_time = 0
for line in your_output.split('\n'):
    if 'Exact KNN' in line:
        m = re.search(r'[\d.]+(?=\s*ms)', line)
        if m: your_knn_time = float(m.group())

# Save your knn.txt
your_knn = []
if os.path.exists("knn.txt"):
    with open("knn.txt") as f:
        for _ in range(N):
            your_knn.append(int(f.readline().split()[3]))

# ── Run OTHER implementation ──────────────────────────────────────────────────
print("\n[2/2] Running OTHER implementation (other_knn)...")
t0 = time.time()
r = subprocess.run([OTHER_BIN, INPUT], capture_output=True, text=True)
other_wall = (time.time() - t0) * 1000

if r.returncode != 0:
    print("other_knn failed:", r.stderr); sys.exit(1)

print(r.stdout.strip())

# Parse other time from time.txt it generates
other_knn_time = 0
if os.path.exists("time.txt"):
    with open("time.txt") as f:
        for line in f:
            if 'KNN' in line:
                m = re.search(r'[\d.]+', line)
                if m: other_knn_time = float(m.group())

# Save other knn.txt to output_other/
other_knn = []
if os.path.exists("knn.txt"):
    with open("knn.txt") as f:
        lines = f.readlines()
    for line in lines:
        other_knn.append(int(line.split()[3]))
    # copy to other dir
    with open(os.path.join(OTHER_DIR, "knn.txt"), 'w') as f:
        f.writelines(lines)

# Write other time.txt
with open(os.path.join(OTHER_DIR, "time.txt"), 'w') as f:
    f.write(f"Other KNN time : {other_knn_time:.3f} ms\n")
    f.write(f"Wall clock     : {other_wall:.3f} ms\n")

# Restore YOUR knn.txt
if os.path.exists("knn.txt") and your_knn:
    with open("knn.txt", 'w') as f:
        # re-run your version to restore outputs
        pass

# Re-run yours to get back your outputs
subprocess.run([YOUR_BIN, INPUT], capture_output=True, text=True)
your_knn_restored = []
if os.path.exists("knn.txt"):
    with open("knn.txt") as f:
        for _ in range(N):
            your_knn_restored.append(int(f.readline().split()[3]))

# ── Compare ───────────────────────────────────────────────────────────────────
your_arr  = np.array(your_knn_restored)
other_arr = np.array(other_knn)

diff       = np.abs(your_arr.astype(int) - other_arr.astype(int))
exact      = int(np.sum(diff == 0))
mae        = float(np.mean(diff))
max_err    = int(np.max(diff))
mismatches = np.where(diff > 0)[0]

# ── Speedup ───────────────────────────────────────────────────────────────────
if other_knn_time > 0 and your_knn_time > 0:
    speedup_vs_other = other_knn_time / your_knn_time
    speedup_label = f"{speedup_vs_other:.2f}x {'faster' if speedup_vs_other >= 1 else 'slower'} than other"
else:
    speedup_vs_other = 0
    speedup_label = "N/A"

# ── Generate report ───────────────────────────────────────────────────────────
lines = []
lines.append("="*65)
lines.append("   KNN COMPARISON: Your Implementation vs Other Implementation")
lines.append("="*65)
lines.append(f"\nInput: N={N}  k={k}  T={T}\n")

lines.append("─"*65)
lines.append("  TIMING")
lines.append("─"*65)
lines.append(f"  {'Method':<35} {'Time':>12}")
lines.append("─"*65)
lines.append(f"  {'Your KNN (CUDA tiled + heap)':<35} {your_knn_time:>10.2f} ms")
lines.append(f"  {'Other KNN (CUDA tiled + linear scan)':<35} {other_knn_time:>10.2f} ms")
lines.append(f"\n  Your speedup vs Other : {speedup_label}")

lines.append("\n" + "─"*65)
lines.append("  CORRECTNESS (Your output vs Other output)")
lines.append("─"*65)
lines.append(f"  Exact matches : {exact}/{N} ({100*exact/N:.1f}%)")
lines.append(f"  MAE           : {mae:.4f}")
lines.append(f"  Max error     : {max_err}")

if exact == N:
    lines.append(f"  VERDICT       : IDENTICAL outputs")
elif mae < 1.0:
    lines.append(f"  VERDICT       : VERY CLOSE (MAE < 1) — likely tiebreaking difference")
elif mae < 5.0:
    lines.append(f"  VERDICT       : CLOSE (MAE < 5)")
else:
    lines.append(f"  VERDICT       : DIFFERENT — check implementations")

if len(mismatches) > 0:
    lines.append(f"\n  First differing points (index: yours vs other):")
    for idx in mismatches[:10]:
        lines.append(f"    point {idx:6d}: yours={your_arr[idx]:3d}  other={other_arr[idx]:3d}  diff={diff[idx]}")

lines.append("\n" + "─"*65)
lines.append("  IMPLEMENTATION DIFFERENCES")
lines.append("─"*65)
lines.append("  Your KNN:")
lines.append("    - Max-heap for k-nearest maintenance")
lines.append("    - Separate histogram_equalize_kernel")
lines.append("    - Full 256-bin histogram + CDF")
lines.append("    - float coordinates, int intensity")
lines.append("    - Block size: 256")
lines.append("")
lines.append("  Other KNN:")
lines.append("    - Linear scan to find max in current k neighbors")
lines.append("    - Lexicographic tiebreaking on equal distances")
lines.append("    - Inline CDF: counts less/equal directly (no full histogram)")
lines.append("    - uint8_t for intensity (saves memory)")
lines.append("    - Block size: 128")

lines.append("\n" + "─"*65)
lines.append("  INTENSITY STATS")
lines.append("─"*65)
lines.append(f"  {'Method':<35} {'Min':>5} {'Max':>5} {'Mean':>7} {'Std':>7}")
lines.append("─"*65)
lines.append(f"  {'Your KNN':<35} {your_arr.min():>5} {your_arr.max():>5} "
             f"{your_arr.mean():>7.1f} {your_arr.std():>7.1f}")
lines.append(f"  {'Other KNN':<35} {other_arr.min():>5} {other_arr.max():>5} "
             f"{other_arr.mean():>7.1f} {other_arr.std():>7.1f}")

lines.append("\n" + "="*65)

report_str = "\n".join(lines)
print("\n" + report_str)

with open(REPORT, 'w') as f:
    f.write(report_str + "\n")

print(f"\nReport saved to {REPORT}")
print(f"Other KNN output in {OTHER_DIR}/")