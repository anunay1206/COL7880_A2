import numpy as np
import os

def load_input(path):
    with open(path) as f:
        N = int(f.readline())
        k = int(f.readline())
        T = int(f.readline())
        data = []
        for _ in range(N):
            vals = f.readline().split()
            data.append([float(vals[0]), float(vals[1]), float(vals[2]), int(vals[3])])
    return np.array(data), N, k, T

def load_output(path, N):
    out = []
    with open(path) as f:
        for _ in range(N):
            vals = f.readline().split()
            out.append(int(vals[3]))
    return np.array(out)

def compute_expected_knn(points, N, k):
    coords    = points[:, :3]
    intensity = points[:, 3].astype(int)
    expected  = np.zeros(N, dtype=int)

    for i in range(N):
        diff  = coords - coords[i]
        dists = (diff**2).sum(axis=1)
        dists[i] = np.inf

        order     = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], dists))
        neighbors = order[:k]

        hist = np.bincount(intensity[neighbors], minlength=256)
        cdf  = np.cumsum(hist)

        nonzero = np.where(cdf > 0)[0]
        if len(nonzero) == 0:
            expected[i] = intensity[i]
            continue

        cmin = int(cdf[nonzero[0]])
        m    = k

        if m == cmin:
            expected[i] = intensity[i]
        else:
            val = int((int(cdf[intensity[i]]) - cmin) / (m - cmin) * 255)
            expected[i] = max(0, min(255, val))

        if i % 500 == 0:
            print(f"  computing... {i}/{N}", end='\r')

    print()
    return expected

def verify_file(name, got, expected, N):
    mismatches = np.where(got != expected)[0]
    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    if len(mismatches) == 0:
        print(f"  PASSED — all {N}/{N} correct")
    else:
        print(f"  FAILED — {N-len(mismatches)}/{N} correct ({len(mismatches)} errors)")
        for idx in mismatches[:10]:
            print(f"  point {idx}: expected {expected[idx]}, got {got[idx]}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches)-10} more")

def verify_approx(got_approx, got_knn, N):
    mae = np.mean(np.abs(got_approx.astype(float) - got_knn.astype(float)))
    print(f"\n{'='*45}")
    print(f"  approx_knn.txt vs knn.txt")
    print(f"{'='*45}")
    print(f"  MAE         = {mae:.4f}")
    print(f"  Max error   = {np.max(np.abs(got_approx - got_knn))}")
    print(f"  Exact match = {np.sum(got_approx == got_knn)}/{N}")
    if mae < 10:
        print(f"  LIKELY CORRECT (MAE < 10)")
    else:
        print(f"  WARNING: MAE seems high")

# ── Main ──────────────────────────────────────────────────────────────────────
print("Loading input...")
points, N, k, T = load_input('input.txt')
print(f"N={N}  k={k}  T={T}")

# Load only files that exist
got_knn    = load_output('knn.txt', N)        if os.path.exists('knn.txt')        else None
got_approx = load_output('approx_knn.txt', N) if os.path.exists('approx_knn.txt') else None
got_kmeans = load_output('kmeans.txt', N)     if os.path.exists('kmeans.txt')     else None

print(f"  knn.txt        : {'found' if got_knn    is not None else 'MISSING'}")
print(f"  approx_knn.txt : {'found' if got_approx is not None else 'MISSING'}")
print(f"  kmeans.txt     : {'found' if got_kmeans is not None else 'MISSING'}")

# Compute expected only if needed and N is small enough
expected_knn = None
if (got_knn is not None or got_approx is not None) and N <= 10000:
    print(f"\nComputing expected KNN output for all {N} points...")
    expected_knn = compute_expected_knn(points, N, k)

# Verify KNN
if got_knn is not None:
    if expected_knn is not None:
        verify_file('knn.txt', got_knn, expected_knn, N)
    else:
        print(f"\nN={N} too large for full KNN verify — skipping")

# Verify Approx KNN
if got_approx is not None:
    if expected_knn is not None:
        verify_file('approx_knn.txt (vs exact)', got_approx, expected_knn, N)
    if got_knn is not None:
        verify_approx(got_approx, got_knn, N)
    else:
        print("\napprox_knn.txt found but knn.txt missing — cannot compute MAE")

# Verify K-means
if got_kmeans is not None:
    print(f"\n{'='*45}")
    print(f"  kmeans.txt — intensity stats")
    print(f"{'='*45}")
    intensity_orig = points[:, 3].astype(int)
    print(f"  Original  mean={intensity_orig.mean():.1f}  std={intensity_orig.std():.1f}  min={intensity_orig.min()}  max={intensity_orig.max()}")
    print(f"  K-means   mean={got_kmeans.mean():.1f}  std={got_kmeans.std():.1f}  min={got_kmeans.min()}  max={got_kmeans.max()}")
    if got_kmeans.min() >= 0 and got_kmeans.max() <= 255:
        print(f"  RANGE OK (0-255)")
    else:
        print(f"  ERROR: values out of 0-255 range")

if got_knn is None and got_approx is None and got_kmeans is None:
    print("\nNo output files found. Run ./a2 input.txt [knn|approx_knn|kmeans] first.")