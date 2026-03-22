import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
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
    data = []
    with open(path) as f:
        for _ in range(N):
            vals = f.readline().split()
            data.append(int(vals[3]))
    return np.array(data)

def plot_cloud(ax, x, y, z, intensity, title, cmap='plasma'):
    sc = ax.scatter(x, y, z, c=intensity, cmap=cmap, s=5, alpha=0.7, vmin=0, vmax=255)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel('X', fontsize=8); ax.set_ylabel('Y', fontsize=8); ax.set_zlabel('Z', fontsize=8)
    ax.tick_params(labelsize=7)
    return sc

def plot_hist(ax, intensity, title, color):
    ax.hist(intensity, bins=64, range=(0,255), color=color, alpha=0.8, edgecolor='none')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Intensity', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.tick_params(labelsize=7)
    mean = np.mean(intensity)
    std  = np.std(intensity)
    ax.axvline(mean, color='red', linewidth=1.2, linestyle='--', label=f'mean={mean:.1f}')
    ax.legend(fontsize=7)

# ── Load data ──────────────────────────────────────────────────────────────
base = os.path.dirname(os.path.abspath(__file__))
inp  = os.path.join(base, 'input.txt')

if not os.path.exists(inp):
    print("input.txt not found next to this script."); sys.exit(1)

data, N, k, T = load_input(inp)
x, y, z, I_orig = data[:,0], data[:,1], data[:,2], data[:,3].astype(int)

outputs = {}
for name in ['knn', 'approx_knn', 'kmeans']:
    path = os.path.join(base, f'{name}.txt')
    if os.path.exists(path):
        outputs[name] = load_output(path, N)
    else:
        print(f"Warning: {name}.txt not found, skipping.")

n_methods = len(outputs)
if n_methods == 0:
    print("No output files found. Run ./a2 input.txt first."); sys.exit(1)

# ── Figure 1: 3D point clouds ──────────────────────────────────────────────
fig1 = plt.figure(figsize=(5*(n_methods+1), 5))
fig1.suptitle('3D Point Cloud — Intensity Coloring', fontsize=13, fontweight='bold')

ax = fig1.add_subplot(1, n_methods+1, 1, projection='3d')
sc = plot_cloud(ax, x, y, z, I_orig, 'Original input')
plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.1, label='Intensity')

colors = {'knn':'#378add', 'approx_knn':'#1D9E75', 'kmeans':'#D85A30'}
titles = {'knn':'After KNN equalization', 'approx_knn':'After approx KNN', 'kmeans':'After K-means'}
for idx, (name, I_out) in enumerate(outputs.items()):
    ax2 = fig1.add_subplot(1, n_methods+1, idx+2, projection='3d')
    sc2 = plot_cloud(ax2, x, y, z, I_out, titles[name])
    plt.colorbar(sc2, ax=ax2, shrink=0.5, pad=0.1, label='Intensity')

plt.tight_layout()
plt.savefig(os.path.join(base, 'viz_3d.png'), dpi=130, bbox_inches='tight')
print("Saved viz_3d.png")

# ── Figure 2: Intensity histograms ─────────────────────────────────────────
fig2, axes = plt.subplots(1, n_methods+1, figsize=(5*(n_methods+1), 4))
fig2.suptitle('Intensity Distribution — Before vs After', fontsize=13, fontweight='bold')

plot_hist(axes[0], I_orig, 'Original', '#888780')
for idx, (name, I_out) in enumerate(outputs.items()):
    plot_hist(axes[idx+1], I_out, titles[name], colors[name])

plt.tight_layout()
plt.savefig(os.path.join(base, 'viz_hist.png'), dpi=130, bbox_inches='tight')
print("Saved viz_hist.png")

# ── Figure 3: Before vs After scatter per method ───────────────────────────
fig3, axes3 = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
if n_methods == 1: axes3 = [axes3]
fig3.suptitle('Original intensity → Remapped intensity', fontsize=13, fontweight='bold')

for idx, (name, I_out) in enumerate(outputs.items()):
    ax3 = axes3[idx]
    ax3.scatter(I_orig, I_out, s=1, alpha=0.3, color=colors[name])
    ax3.plot([0,255],[0,255], 'r--', linewidth=1, label='no change')
    ax3.set_xlabel('Original intensity', fontsize=9)
    ax3.set_ylabel('Output intensity', fontsize=9)
    ax3.set_title(titles[name], fontsize=10)
    ax3.legend(fontsize=8)
    mae = np.mean(np.abs(I_orig - I_out))
    ax3.text(10, 230, f'MAE vs orig: {mae:.1f}', fontsize=9, color='black',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(os.path.join(base, 'viz_scatter.png'), dpi=130, bbox_inches='tight')
print("Saved viz_scatter.png")

# ── Figure 4: MAE between methods ─────────────────────────────────────────
if 'knn' in outputs and 'approx_knn' in outputs:
    fig4, axes4 = plt.subplots(1, 2, figsize=(10, 4))
    fig4.suptitle('Approx KNN vs Exact KNN accuracy', fontsize=13, fontweight='bold')

    diff = np.abs(outputs['knn'] - outputs['approx_knn'])
    mae  = np.mean(diff)

    axes4[0].hist(diff, bins=40, color='#1D9E75', alpha=0.8, edgecolor='none')
    axes4[0].set_title(f'|KNN - ApproxKNN| per point   (MAE = {mae:.2f})', fontsize=10)
    axes4[0].set_xlabel('Absolute error', fontsize=9)
    axes4[0].set_ylabel('Count', fontsize=9)

    axes4[1].scatter(outputs['knn'], outputs['approx_knn'], s=1, alpha=0.3, color='#1D9E75')
    axes4[1].plot([0,255],[0,255],'r--',linewidth=1,label='perfect match')
    axes4[1].set_xlabel('Exact KNN intensity', fontsize=9)
    axes4[1].set_ylabel('Approx KNN intensity', fontsize=9)
    axes4[1].set_title('Exact vs Approx output', fontsize=10)
    axes4[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(base, 'viz_accuracy.png'), dpi=130, bbox_inches='tight')
    print("Saved viz_accuracy.png")
    print(f"\nApprox KNN MAE vs Exact KNN: {mae:.4f}")

# ── Print summary stats ────────────────────────────────────────────────────
print("\n── Intensity stats ──────────────────────────────")
print(f"{'Method':<20} {'Min':>5} {'Max':>5} {'Mean':>7} {'Std':>7}")
print(f"{'Original':<20} {I_orig.min():>5} {I_orig.max():>5} {I_orig.mean():>7.1f} {I_orig.std():>7.1f}")
for name, I_out in outputs.items():
    print(f"{titles[name]:<20} {I_out.min():>5} {I_out.max():>5} {I_out.mean():>7.1f} {I_out.std():>7.1f}")

plt.show()