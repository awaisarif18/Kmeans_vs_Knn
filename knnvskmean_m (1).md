# KNN vs K-means on ResNet Embeddings — the M metric (Implementation Plan)

> **For Claude Code:** build a **fresh Jupyter notebook** from this spec. It supersedes the
> raw-pixel version. **No logistic regression.** The one big change from the previous notebook is
> the front end: images are turned into **pretrained-ResNet feature vectors** instead of flattened
> pixels. Everything downstream (K-means, KNN, M, plots) operates on those vectors and is otherwise
> the same analysis. Include **every** visual listed in section 8, including the new bipartite
> mapping figure whose code is given verbatim.

---

## 1. Goal in one line

Split the images two ways — **K-means** (unsupervised, k = 2) and **KNN** (supervised, 2 groups) —
on **ResNet embeddings**, then measure how much the two splits disagree with the normalized metric

$$M = \frac{1}{N}\sum_{i=1}^{N}\big(\text{label}^{\text{KMeans}}_i - \text{label}^{\text{KNN}}_i\big)^2 \in [0,1],
\qquad \text{Agreement}=1-M.$$

M = 0 ⇒ the two partitions are identical. Expect M to be **much lower** than the raw-pixel version,
because the embeddings carry semantic (cat-vs-dog) information rather than brightness/layout.

---

## 2. Config (top cell)

```python
IMG_SIZE      = 224     # ResNet input size (was 64)
N_PER_CLASS   = 1000    # 1000 cats + 1000 dogs -> 2000 rows
K_NEIGHBORS   = 5       # k for KNN
RANDOM_STATE  = 42
CV_FOLDS      = 5       # folds for honest KNN predictions
EMB_CACHE     = 'resnet_embeddings.npz'   # cache file for embeddings + labels + paths
```

---

## 3. Stage 1 — Feature extraction with a pretrained ResNet backbone  ← the key change

Load images as **RGB at 224×224**, run them through a **frozen ResNet50** (ImageNet weights, classifier
head removed, global-average-pooled) to get **2048-D embeddings**, then **L2-normalize** and **cache** them.

**Critical details:** use `preprocess_input` (NOT manual `/255`); keep the file paths aligned with the
rows of `X` (needed later to display sample images, since you can't reshape a 2048-D embedding into a
picture); cache to `.npz` so the slow forward pass runs only once.

```python
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as kimage

DATA_DIR = Path('microsoft-catsvsdogs-dataset') / 'PetImages'
CAT_DIR, DOG_DIR = DATA_DIR / 'Cat', DATA_DIR / 'Dog'

def list_images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == '.jpg')

def image_to_tensor(path):
    img = kimage.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))  # RGB
    return preprocess_input(kimage.img_to_array(img))             # (224,224,3)

def build_dataset():
    paths, y = [], []
    for files, label in [(list_images(CAT_DIR), 0), (list_images(DOG_DIR), 1)]:
        kept = 0
        for p in files:
            if kept >= N_PER_CLASS:
                break
            try:
                image_to_tensor(p); paths.append(p); y.append(label); kept += 1
            except Exception:
                continue  # skip corrupt JPEGs
    return paths, np.array(y)

if Path(EMB_CACHE).exists():
    d = np.load(EMB_CACHE, allow_pickle=True)
    X, y, paths = d['X'], d['y'], list(d['paths'])
else:
    paths, y = build_dataset()
    backbone = ResNet50(weights='imagenet', include_top=False, pooling='avg')  # -> 2048-D
    batch = np.stack([image_to_tensor(p) for p in paths])        # (N, 224, 224, 3)
    X = backbone.predict(batch, batch_size=32, verbose=1)        # (N, 2048)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)             # L2-normalize
    np.savez(EMB_CACHE, X=X, y=y, paths=np.array([str(p) for p in paths], dtype=object))

N = len(y)
print('X (embeddings):', X.shape, '| y:', y.shape, '| cats/dogs:', (y==0).sum(), (y==1).sum())
```

---

## 4. Stage 2 — 2-D projection for plotting

Embeddings are already L2-normalized, so no `StandardScaler` needed for modelling. Compute a **2-D PCA**
purely to draw the scatter plots. (Optional: also make a 50-D PCA `X_pca50` as a speed/denoise variant —
not required.)

```python
from sklearn.decomposition import PCA
X_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
```

---

## 5. Stage 3 & 4 — the two splits

```python
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict

# K-means (unsupervised) — never sees y
km_labels = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE).fit_predict(X)

# KNN (supervised) — honest labels via cross-validation (every point predicted by others)
knn_labels = cross_val_predict(KNeighborsClassifier(n_neighbors=K_NEIGHBORS), X, y, cv=CV_FOLDS)
knn_acc_vs_true = (knn_labels == y).mean()
```

---

## 6. Stage 5 & 6 — align (flip) and compute normalized M

K-means cluster IDs are arbitrary, so align to KNN by trying both mappings and keeping the better one,
then M = disagreements / N.

```python
def align_kmeans_to_knn(reference, clusters):
    asis = (clusters != reference).sum()
    flip = ((1 - clusters) != reference).sum()
    if flip < asis:
        return (1 - clusters), int(flip), True
    return clusters, int(asis), False

km_aligned, M_raw, flipped = align_kmeans_to_knn(knn_labels, km_labels)
M = M_raw / N
agreement = 1 - M
print(f'flip={flipped} | M={M:.4f} | agreement={agreement:.4f}')
```

---

## 7. Stage 7 — flip-proof cross-checks

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ari = adjusted_rand_score(knn_labels, km_aligned)
nmi = normalized_mutual_info_score(knn_labels, km_aligned)
km_acc_vs_true = max((km_labels == y).mean(), ((1 - km_labels) == y).mean())
```

---

## 8. Stage 8 — Visuals & tables (implement ALL of these)

Use `matplotlib` + `pandas` only (no seaborn). Each gets its own cell with a short markdown intro.

**8.1 Contingency table + heatmap** — `confusion_matrix(km_aligned, knn_labels)`, rows = K-means (aligned),
cols = KNN; show as a `pandas` DataFrame and as an annotated `imshow` (Blues), titled with the agreement.

**8.2 Three-panel PCA scatter** — same `X_2d` points three times, coloured by **true label**, **K-means
(aligned)**, **KNN**; `cmap='coolwarm'`, `s=8, alpha=0.5`. If the splits matched, panels 2 and 3 look identical.

**8.3 Agreement map** — PCA scatter, green where `km_aligned == knn_labels`, red where they disagree
(red points are exactly what M counts); legend shows the two counts.

**8.4 Representative images per K-means cluster** — for each cluster, find the rows whose **embedding** is
closest to that cluster's `kmeans.cluster_centers_[c]` (Euclidean in embedding space), then **load and show
the original RGB images from `paths[i]`** (do NOT reshape the embedding). Show ~8 per cluster, title each
with `cat`/`dog`. This reveals what the clusters latched onto.

**8.5 M bar + full summary table** — a horizontal stacked bar (green = agreement, red = M) and a `pandas`
table with: M raw (`{M_raw}/{N}`), M normalized, agreement, ARI, NMI, KNN acc vs true, K-means acc vs true,
flip applied.

**8.6 NEW — Sankey-style bipartite M-mapping (hero figure).** Place this after `km_aligned`, `knn_labels`,
`M`, `agreement`, `N` exist. It shows M as a big hero number over a bipartite ribbon diagram (2 K-means
circles ↔ 2 KNN circles; green ribbons = agree, red = counted in M; ribbon widths ∝ counts). **Add these
imports** at the top of the notebook: `from matplotlib.path import Path as MPath` and
`from matplotlib.patches import PathPatch, Circle, Patch`. Implement this cell **exactly**:

```python
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Circle, Patch

SURFACE, INK, INK_2 = '#fcfcfb', '#0b0b0b', '#52514e'
NODE_FILL, NODE_RING = '#86b6ef', '#1c5cab'
AGREE, DISAGREE      = '#0ca30c', '#d03b3b'

ct = confusion_matrix(km_aligned, knn_labels)      # rows = K-means (aligned), cols = KNN
km_sizes, knn_sizes = ct.sum(axis=1), ct.sum(axis=0)

fig = plt.figure(figsize=(11, 7.2), facecolor=SURFACE)
gs  = fig.add_gridspec(2, 1, height_ratios=[1, 3.2], hspace=0.02)

# ---- hero figure: M -------------------------------------------------------
axh = fig.add_subplot(gs[0]); axh.set_facecolor(SURFACE); axh.axis('off')
axh.text(0.5, 0.80, f'M = {M:.3f}', ha='center', va='center',
         fontsize=52, fontweight='bold', color=INK)
axh.text(0.5, 0.30, f'K-means and KNN disagree on {ct[0, 1] + ct[1, 0]} of {N} images '
                    f'({M*100:.1f}%)      ·      agreement (1 − M) = {agreement:.3f}',
         ha='center', va='center', fontsize=11.5, color=INK_2)

# ---- bipartite mapping: 2 K-means circles <-> 2 KNN circles ---------------
ax = fig.add_subplot(gs[1]); ax.set_facecolor(SURFACE)
ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.set_aspect('equal'); ax.axis('off')

x_km, x_knn = 2.3, 7.7
ys = [3.05, 1.35]                                   # group 0 on top, group 1 below
r_of = lambda n: 0.25 + 0.45 * np.sqrt(n / N)       # area-ish scaling
km_r  = [r_of(km_sizes[i])  for i in (0, 1)]
knn_r = [r_of(knn_sizes[j]) for j in (0, 1)]

def ribbon(xa, ya0, ya1, xb, yb0, yb1, color):
    """Bezier band from the left node's edge to the right node's edge."""
    cx = (xa + xb) / 2
    verts = [(xa, ya0), (cx, ya0), (cx, yb0), (xb, yb0), (xb, yb1),
             (cx, yb1), (cx, ya1), (xa, ya1), (xa, ya0)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor=SURFACE,
                           lw=1.5, alpha=0.55, zorder=1))

def centerline(t, xa, ya, xb, yb):
    """Point at parameter t along the ribbon's centre Bezier — used to place labels."""
    cx = (xa + xb) / 2
    u = 1 - t
    return (u**3 * xa + 3*u**2*t * cx + 3*u*t**2 * cx + t**3 * xb,
            u**3 * ya + 3*u**2*t * ya + 3*u*t**2 * yb + t**3 * yb)

# stack the flows top-to-bottom on each node, Sankey-style
lcur = {i: ys[i] + 0.75 * km_r[i]  for i in (0, 1)}
rcur = {j: ys[j] + 0.75 * knn_r[j] for j in (0, 1)}

for i in (0, 1):
    for j in (0, 1):
        c = ct[i, j]
        ya0, ya1 = lcur[i], lcur[i] - 1.5 * km_r[i]  * c / km_sizes[i]
        yb0, yb1 = rcur[j], rcur[j] - 1.5 * knn_r[j] * c / knn_sizes[j]
        lcur[i], rcur[j] = ya1, yb1
        xa, xb = x_km + 0.55 * km_r[i], x_knn - 0.55 * knn_r[j]
        ribbon(xa, ya0, ya1, xb, yb0, yb1, AGREE if i == j else DISAGREE)
        # the two disagree ribbons cross at the centre, so park their labels
        # off-centre (one early, one late) instead of on top of each other
        t = 0.5 if i == j else (0.24 if i < j else 0.76)
        lx, ly = centerline(t, xa, (ya0 + ya1) / 2, xb, (yb0 + yb1) / 2)
        ax.text(lx, ly, f'{c}\n{c/N*100:.1f}%', ha='center', va='center',
                fontsize=9.5, color=INK, linespacing=1.35, zorder=4,
                bbox=dict(boxstyle='round,pad=0.30', fc=SURFACE, ec='none', alpha=0.88))

for x, radii, sizes, name in [(x_km, km_r, km_sizes, 'K-means'),
                              (x_knn, knn_r, knn_sizes, 'KNN')]:
    for g in (0, 1):
        ax.add_patch(Circle((x, ys[g]), radii[g], facecolor=NODE_FILL,
                            edgecolor=NODE_RING, lw=1.8, zorder=3))
        ax.text(x, ys[g] + 0.13, f'{name}\ngroup {g}', ha='center', va='center',
                fontsize=9.5, color=INK, linespacing=1.3, zorder=4)
        ax.text(x, ys[g] - 0.26, f'n = {sizes[g]}', ha='center', va='center',
                fontsize=10, fontweight='bold', color=NODE_RING, zorder=4)

ax.text(x_km,  4.15, 'K-MEANS  (unsupervised, aligned)', ha='center', fontsize=11,
        fontweight='bold', color=INK_2)
ax.text(x_knn, 4.15, 'KNN  (supervised)', ha='center', fontsize=11,
        fontweight='bold', color=INK_2)
ax.legend(handles=[Patch(facecolor=AGREE, alpha=0.55, label='same group  → agree'),
                   Patch(facecolor=DISAGREE, alpha=0.55, label='different group  → counted in M')],
          loc='lower center', bbox_to_anchor=(0.5, -0.04), ncol=2,
          frameon=False, fontsize=10.5)

plt.show()
```

---

## 9. Interpretation cell (markdown, at the end)

- **M is the inaccuracy** between the two splits: 0 = identical; larger = more disagreement.
- Since KNN closely tracks the true labels, M ≈ **how well K-means recovers cat vs dog on its own**.
- On **ResNet embeddings** expect M to be **much lower** than on raw pixels (agreement often 0.85–0.95+):
  same algorithms, better representation. State the actual M, whether a flip was applied, and how it
  compares to the raw-pixel run.

---

## 10. Implementation notes for Claude Code

- Requires `tensorflow`; first run downloads ImageNet weights (~100 MB) and does one forward pass over
  2000 images — minutes on CPU, seconds on GPU. **Cache** to `resnet_embeddings.npz` and reuse.
- Keep `paths` row-aligned with `X` so visual 8.4 can load the original images.
- The backbone is **frozen** (feature extractor only) — no training anywhere.
- Standard deps: `tensorflow`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `Pillow`.
- Structure the notebook with a markdown intro before each code cell, matching the clean, explanatory
  style of the earlier cats-vs-dogs notebook.
