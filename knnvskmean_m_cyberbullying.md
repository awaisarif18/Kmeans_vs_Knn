# KNN vs K-means — the M metric on cyberbullying tweets (Plan)

Companion to `knnvskmean_m.md`. Same metric, same procedure — different data and, critically,
a **representation that actually works**, which is what makes M mean what we claim it means.

**Dataset:** `cyberbullying_age_vs_not.csv` — 15,929 tweets, already deduped and binarised
(`age → 1` = 7,992, `not_cyberbullying → 0` = 7,937). Near-perfectly balanced, so the 0.50
coin-flip baseline is the honest reference point.

**Representation:** BERT sentence embeddings — `all-MiniLM-L6-v2`, 384-dim, L2-normalised —
exactly as in `knn_cyberbullying_bert.ipynb` and `kmeans_cyberbullying_bert.ipynb`.

**Output notebook:** `knn_vs_kmeans_M_cyberbullying.ipynb` (new file; the cats/dogs notebook is
left untouched).

---

## 0. Why this run exists

The cats/dogs run produced **M = 0.471 with KNN accuracy 0.537**. KNN was barely above chance, so
M was measuring the disagreement between two near-arbitrary partitions — the number was real but
the *interpretation* in section 13 ("M ≈ how well K-means recovers the true classes") did not hold,
because KNN wasn't tracking the true classes either.

On BERT embeddings, KNN reaches ~0.81 across 6 classes and should be **well above 0.90 on this
binary pair** (`age` was the easiest class in the 6-way run: F1 0.908, recall 0.975). That restores
the plan's central assumption — **KNN ≈ the true labels** — and only then is M legitimately
readable as "how well does K-means recover the real split on its own."

So this notebook is not just "the same thing on new data." It is the **control case** that makes
the cats/dogs result interpretable, and the two together tell the actual story: *M is a property
of the representation, not of the algorithms.*

---

## 1. The M metric (unchanged)

$$M = \frac{1}{N}\sum_{i=1}^{N}\big(\text{label}^{\text{KMeans}}_i - \text{label}^{\text{KNN}}_i\big)^2$$

Labels are 0/1 → each term is 1 on disagreement, 0 on agreement. M = disagreement rate ∈ [0,1];
agreement = 1 − M. **Label-flip alignment applies as before** (K-means cluster IDs are arbitrary —
try as-is and flipped, keep the better mapping).

---

## 2. Pipeline

1. **Load** `cyberbullying_age_vs_not.csv` → `tweet_text`, `cyberbullying_type`, `label`; N = 15,929.
2. **Clean** — strip URLs and `@mentions`, collapse whitespace; keep casing and punctuation
   (BERT uses them). Identical `clean_text` function to the two existing BERT notebooks.
3. **Embed** — `all-MiniLM-L6-v2`, `batch_size=64`, `normalize_embeddings=True` → `X` (15929 × 384).
   Cache to **`bert_age_vs_not.npz`**. *(The existing `bert_embeddings.npz` cannot be reused — it
   holds the 6-class 80/20 split in a different row order with no recoverable row identity.)*
   First run ≈ 2–4 min on CPU; cached thereafter.
4. **No StandardScaler.** — see §3, this is a deliberate deviation from the cats/dogs plan.
5. **PCA-2D for plotting only**; report explained variance.
6. **K-means (k = 2)**, `n_init=10`, on the 384-dim embeddings → one cluster label per tweet.
7. **Tune k for KNN** by 5-fold CV using the `NearestNeighbors`-once-reuse-for-all-k trick from
   `knn_cyberbullying_bert.ipynb` (fit `max_k=30` neighbours per fold, take the first k and vote).
8. **KNN labels** via 5-fold `cross_val_predict` at the tuned k — every tweet predicted by *other*
   tweets, no self-memorisation. Record KNN accuracy vs true labels.
9. **Align** K-means to KNN (flip check), then **compute M**, agreement, and the flip flag.
10. **Cross-check** with ARI and NMI (flip-proof by construction); also report K-means accuracy vs
    true labels and silhouette score.

---

## 3. The one deliberate deviation: no StandardScaler

The cats/dogs plan standardised because raw pixels have wildly unequal scales. BERT embeddings are
already **L2-normalised to the unit sphere**, where Euclidean distance is a monotone function of
cosine similarity — which is the geometry the model was trained to produce. `StandardScaler`
per-dimension re-weighting destroys that unit norm and inflates low-variance noise dimensions.

Rather than assert this, the notebook **measures it**: a short ablation running the whole M
computation both ways and reporting both M values side by side. If standardising makes M worse, we
have shown it rather than claimed it.

---

## 4. Sections & visuals

Each numbered section = a markdown cell + code cell, matching the cats/dogs notebook's rhythm.

| # | Section | Visual |
|---|---------|--------|
| 1 | Setup & imports | — |
| 2 | Load + class balance | **Bar of the two class counts** (already balanced — one glance, then move on) |
| 3 | Clean text | before/after table of 5 tweets |
| 4 | BERT embeddings (cached) | shape printout + **tweet-length distribution** by class |
| 5 | See the embedding space | **PCA-2D scatter coloured by true label** + **t-SNE on a 5k sample** — if the two classes form visible regions, KNN has something to work with |
| 6 | K-means (k=2) | **PCA scatter coloured by cluster** + cluster size bar |
| 7 | Choosing k for KNN | **CV accuracy vs k line chart**, best k marked; top-5 k table |
| 8 | KNN labels via cross-val | accuracy vs true; **confusion matrix vs true labels** |
| 9 | Align & compute **M** | **the headline number** |
| 10 | Contingency table | **K-means × KNN heatmap** — big diagonal ⇒ agreement |
| 11 | Three-panel PCA | same points coloured by **true / K-means / KNN**, side by side — if they matched, panels 2 and 3 would be identical |
| 12 | Agreement map | **PCA scatter, agree (green) vs disagree (red)** — the red points are literally what M counts |
| 13 | What each cluster caught | the **10 tweets closest to each centroid**, as a table — shows *what* K-means grouped on |
| 14 | Cross-checks | **summary table**: M, agreement, ARI, NMI, KNN acc, K-means acc, silhouette, flip flag |
| 15 | **M at a glance** | the **hero-figure + bipartite ribbon diagram** ported from section 15 of the cats/dogs notebook — 2 K-means circles ↔ 2 KNN circles, ribbon width = tweet count, green = agree / red = counted in M |
| 16 | **Representation ablation** | **grouped bar of M across representations**: TF-IDF · BERT · BERT+StandardScaler · (cats/dogs raw pixels as the reference bar) — the single chart that carries the whole lesson |
| 17 | **Cats/dogs vs tweets, side by side** | **paired scatter/dumbbell of (KNN accuracy, M)** for both runs — makes explicit that a low M only *means* something when KNN accuracy is high |
| 18 | Conclusion | — |

Sections 16–17 are the new material; 1–15 mirror the cats/dogs notebook so the two are directly
comparable.

**Chart conventions** (carried over): status green `#0ca30c` / red `#d03b3b` reserved for
agree/disagree and always shipped with labels, never colour alone; sequential blue for magnitude;
direct labels on ≤4 series; text in ink tokens, never series colour; every figure rendered and
eyeballed for label collisions before it ships.

---

## 5. What to expect

- **KNN accuracy ~0.93–0.97.** `age` vs `not_cyberbullying` is the easy pair — age-related bullying
  has strong lexical markers (school, bully, high school, kids) that BERT encodes cleanly.
- **M substantially lower than 0.471** — plausibly **0.05–0.20**. K-means on BERT embeddings should
  find something close to the real semantic split, because in this space the classes *are* the
  dominant source of variance.
- **ARI and NMI clearly above 0** (unlike cats/dogs, where they hovered near zero), agreeing with M.
- **TF-IDF arm lands between** raw pixels and BERT — the representation ladder made visible.

If M comes out high anyway, that is still a result, and section 17 is where we say what it means.

---

## 6. Reading the final number

- M is the **disagreement rate** between the two splits; agreement = 1 − M.
- **The claim "M ≈ how well K-means recovers the true classes" is only licensed when KNN accuracy is
  high.** This notebook reports KNN accuracy *next to* M everywhere, so the reader can always check
  that the licence holds. That is the fix for the flaw in the cats/dogs write-up.
- Report M with agreement, ARI, NMI, and KNN accuracy together — never M alone.

---

## 7. Execution notes

- **Use the `.venv` kernel**, not system Python — `sentence_transformers` 5.6.0 and `torch` 2.13.0+cpu
  are installed there; the cats/dogs notebook ran on system Python, which has neither.
- Build the notebook programmatically with `nbformat`, execute with `nbconvert --execute --inplace`,
  then extract and *look at* each figure. (After execution the file will be ~1–2 MB of embedded PNGs
  and should not be read directly.)
- `bert_age_vs_not.npz` (~24 MB) and the executed notebook are new untracked files.
