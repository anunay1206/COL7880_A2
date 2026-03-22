#!/usr/bin/env python3
import subprocess, os, sys, time, re
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_BIN   = "./sequential"
CUDA_BIN  = "./a2"
INPUT     = "input.txt"
SEQ_DIR   = "output_sequential"
CUDA_DIR  = "output_cuda"
REPORT    = "comparison_report.txt"

# ── Setup dirs ────────────────────────────────────────────────────────────────
os.makedirs(SEQ_DIR,  exist_ok=True)
os.makedirs(CUDA_DIR, exist_ok=True)

# ── Build sequential if needed ────────────────────────────────────────────────
if not os.path.exists(SEQ_BIN) or \
   os.path.getmtime("sequential.cpp") > os.path.getmtime(SEQ_BIN):
    print("Building sequential...")
    r = subprocess.run(["g++", "-O3", "-o", SEQ_BIN, "sequential.cpp", "-lm"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("Build failed:", r.stderr); sys.exit(1)
    print("Built sequential OK")
else:
    print("Sequential already up to date.")

# ── Check input ───────────────────────────────────────────────────────────────
if not os.path.exists(INPUT):
    print(f"ERROR: {INPUT} not found. Generate it first.")
    sys.exit(1)

with open(INPUT) as f:
    N = int(f.readline())
    k = int(f.readline())
    T = int(f.readline())

print(f"\nInput: N={N}  k={k}  T={T}")
print("="*60)

# ── Run sequential ────────────────────────────────────────────────────────────
print("\n[1/2] Running SEQUENTIAL version...")
r = subprocess.run([SEQ_BIN, INPUT, SEQ_DIR], capture_output=True, text=True)
if r.returncode != 0:
    print("Sequential failed:", r.stderr); sys.exit(1)
print(r.stdout.strip())

# Parse sequential times from time.txt
seq_times = {}
time_file = os.path.join(SEQ_DIR, "time.txt")
if os.path.exists(time_file):
    with open(time_file) as f:
        for line in f:
            m = re.search(r'[\d.]+(?=\s*ms)', line)
            if not m: continue
            val = float(m.group())
            if 'Approx' in line:
                seq_times['approx_knn'] = val
            elif 'KNN' in line:
                seq_times['knn'] = val
            elif 'K-means' in line:
                seq_times['kmeans'] = val

# ── Run CUDA — one mode at a time ─────────────────────────────────────────────
print("\n[2/2] Running CUDA+OpenMP version...")

cuda_times = {}
modes  = ['knn', 'approx_knn', 'kmeans']
labels = ['KNN', 'Approx KNN', 'K-means']

for mode in modes:
    r = subprocess.run([CUDA_BIN, INPUT, mode], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"CUDA {mode} failed:", r.stderr); sys.exit(1)
    output = r.stdout + r.stderr
    print(output.strip())

    # Parse time from TIME_IT output line
    for line in output.split('\n'):
        m = re.search(r'[\d.]+(?=\s*ms)', line)
        if m and ('KNN' in line or 'K-means' in line or 'means' in line or 'Exact' in line or 'Approx' in line):
            if 'converged' not in line:
                cuda_times[mode] = float(m.group())

# Copy CUDA outputs to cuda dir
for mode in modes:
    fname = f"{mode}.txt"
    if os.path.exists(fname):
        with open(fname) as f: data = f.read()
        with open(os.path.join(CUDA_DIR, fname), 'w') as f: f.write(data)

# Write CUDA time.txt
with open(os.path.join(CUDA_DIR, "time.txt"), 'w') as f:
    for mode, label in zip(modes, labels):
        f.write(f"CUDA+OMP {label:15s}: {cuda_times.get(mode, 0):.3f} ms\n")

# ── Load outputs for comparison ───────────────────────────────────────────────
def load_intensities(path, N):
    out = []
    with open(path) as f:
        for _ in range(N):
            line = f.readline().split()
            out.append(int(line[3]))
    return np.array(out)

seq_out  = {}
cuda_out = {}
for mode in modes:
    fname = f"{mode}.txt"
    sp = os.path.join(SEQ_DIR, fname)
    cp = os.path.join(CUDA_DIR, fname)
    if os.path.exists(sp):  seq_out[mode]  = load_intensities(sp, N)
    if os.path.exists(cp):  cuda_out[mode] = load_intensities(cp, N)

# ── Generate comparison report ────────────────────────────────────────────────
print("\n" + "="*60)
print("Generating comparison report...")

lines = []
lines.append("="*65)
lines.append("   COMPARISON REPORT — CUDA+OpenMP vs Sequential")
lines.append("="*65)
lines.append(f"\nInput: N={N}  k={k}  T={T}\n")

# ── Timing section ────────────────────────────────────────────────────────────
lines.append("─"*65)
lines.append("  EXECUTION TIMES")
lines.append("─"*65)
lines.append(f"  {'Method':<23} {'Sequential':>12} {'CUDA+OMP':>12} {'Speedup':>10}")
lines.append("─"*65)

speedups = {}
for mode, label in zip(modes, labels):
    st = seq_times.get(mode, 0)
    ct = cuda_times.get(mode, 0)
    sp = st / ct if ct > 0 else 0
    speedups[mode] = sp
    lines.append(f"  {label:<23} {st:>10.1f}ms {ct:>10.1f}ms {sp:>9.2f}x")

lines.append("─"*65)
knn_t    = cuda_times.get('knn', 0)
approx_t = cuda_times.get('approx_knn', 1)
lines.append(f"\n  Approx KNN speedup over CUDA KNN: {knn_t/approx_t:.2f}x\n")

# ── Correctness section ───────────────────────────────────────────────────────
lines.append("─"*65)
lines.append("  CORRECTNESS — CUDA output vs Sequential output")
lines.append("─"*65)

for mode, label in zip(modes, labels):
    lines.append(f"\n  [{label}]")
    if mode not in seq_out or mode not in cuda_out:
        lines.append("    Could not compare — file missing")
        continue

    s = seq_out[mode]
    c = cuda_out[mode]
    diff        = np.abs(s.astype(int) - c.astype(int))
    exact_match = int(np.sum(diff == 0))
    mae         = float(np.mean(diff))
    mmax        = int(np.max(diff))
    mismatches  = np.where(diff > 0)[0]

    lines.append(f"    Exact matches : {exact_match}/{N} "
                 f"({'100%' if exact_match==N else f'{100*exact_match/N:.1f}%'})")
    lines.append(f"    MAE           : {mae:.4f}")
    lines.append(f"    Max error     : {mmax}")

    if exact_match == N:
        lines.append(f"    VERDICT       : PERFECT MATCH")
    elif mae < 2.0:
        lines.append(f"    VERDICT       : VERY CLOSE (MAE < 2)")
    elif mae < 10.0:
        lines.append(f"    VERDICT       : ACCEPTABLE (MAE < 10)")
    else:
        lines.append(f"    VERDICT       : HIGH ERROR — check implementation")

    if 0 < len(mismatches) <= 5:
        lines.append(f"    Mismatches at : {list(mismatches)}")
    elif len(mismatches) > 5:
        lines.append(f"    First 5       : {list(mismatches[:5])}")

# ── Intensity stats ───────────────────────────────────────────────────────────
lines.append("\n" + "─"*65)
lines.append("  INTENSITY STATISTICS")
lines.append("─"*65)
lines.append(f"  {'Method':<28} {'Min':>5} {'Max':>5} {'Mean':>7} {'Std':>7}")
lines.append("─"*65)

orig = []
with open(INPUT) as f:
    f.readline(); f.readline(); f.readline()
    for _ in range(N):
        orig.append(int(f.readline().split()[3]))
orig = np.array(orig)
lines.append(f"  {'Original input':<28} {orig.min():>5} {orig.max():>5} "
             f"{orig.mean():>7.1f} {orig.std():>7.1f}")

for mode, label in zip(modes, labels):
    if mode in seq_out:
        s = seq_out[mode]
        lines.append(f"  {'Seq '+label:<28} {s.min():>5} {s.max():>5} "
                     f"{s.mean():>7.1f} {s.std():>7.1f}")
    if mode in cuda_out:
        c = cuda_out[mode]
        lines.append(f"  {'CUDA '+label:<28} {c.min():>5} {c.max():>5} "
                     f"{c.mean():>7.1f} {c.std():>7.1f}")

# ── Summary ───────────────────────────────────────────────────────────────────
lines.append("\n" + "─"*65)
lines.append("  SUMMARY")
lines.append("─"*65)
for mode, label in zip(modes, labels):
    sp = speedups.get(mode, 0)
    verdict = "FAST" if sp > 10 else "MODERATE" if sp > 3 else "SLOW"
    lines.append(f"  {label}: {sp:.2f}x speedup — {verdict}")

lines.append("\n" + "="*65)

report_str = "\n".join(lines)
print("\n" + report_str)

with open(REPORT, 'w') as f:
    f.write(report_str + "\n")

print(f"\nReport saved to      : {REPORT}")
print(f"Sequential outputs   : {SEQ_DIR}/")
print(f"CUDA outputs         : {CUDA_DIR}/")