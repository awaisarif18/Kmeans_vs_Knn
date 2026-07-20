> **This document has two parts:**
>
> 1. **Previous learnings** (below) — K-Means & KNN on cyberbullying tweets, TF-IDF vs BERT.
> 2. **[Today's learnings — 20 July 2026](#todays-learnings--20-july-2026)** — KNN vs K-means and the
>    **M metric** on Cats vs Dogs: raw pixels vs pretrained ResNet50 embeddings.

---

# Part 1 — Previous learnings

# Clustering Cyberbullying Tweets with K-Means — TF-IDF vs BERT

A step-by-step guide to what we built, in what order, and *why* each step exists.
The goal of both notebooks is the same: take tweets labelled `age` and
`not_cyberbullying`, run **K-Means with k=2**, and check whether the two clusters
K-Means finds line up with the two real categories.

- `kmeans_cyberbullying.ipynb` — the **TF-IDF** version
- `kmeans_cyberbullying_bert.ipynb` — the **BERT** version

---

## 0. The one idea to keep in mind

**K-Means is unsupervised.** It groups points by how close they are to each other —
it *never sees the labels* while clustering. We keep the `age=1` / `not_cyberbullying=0`
labels only to **evaluate** the clusters afterward (did the unsupervised split match
reality?). That is the whole experiment.

K-Means only works on **numbers**. Tweets are text. So the real work in both notebooks
is *turning text into numeric vectors*. That single step — the "vectorizer" — is the
only thing that differs between the two notebooks. Everything around it is the same.

```
raw tweets ->  [ VECTORIZER ]  -> numeric matrix X -> K-Means -> clusters -> evaluate
                    ^
          TF-IDF  or  BERT  (this is the only difference)
```

---

## Part A — Shared pipeline (identical in both notebooks)

### Step 1: Load the data
```python
import pandas as pd
df = pd.read_csv('cyberbullying_tweets.csv')
```
The CSV has two columns: `tweet_text` and `cyberbullying_type`. The full file has
~47.7k rows across 6 classes.

### Step 2: Keep only the two classes we care about
```python
df = df[df['cyberbullying_type'].isin(['age', 'not_cyberbullying'])].copy()
```
Drops `religion`, `ethnicity`, `gender`, `other_cyberbullying`. Leaves ~15,937 rows.
The `.copy()` avoids pandas "SettingWithCopy" warnings when we add columns later.

### Step 3: Remove duplicate rows
```python
df = df.drop_duplicates()
```
Removed 8 duplicates → **15,929 rows**. Duplicates would bias the clusters (the same
tweet counted twice pulls a centroid toward it).

### Step 4: Turn labels into binary numbers
```python
df['label'] = df['cyberbullying_type'].map({'age': 1, 'not_cyberbullying': 0})
```
`age -> 1`, `not_cyberbullying -> 0`. Again: this column is for **evaluation only**.

### Step 5: Clean the text
Strip noise so the vectorizer sees signal, not junk.
- **TF-IDF version (aggressive):** lowercase, remove URLs, `@mentions`, the `#` symbol,
  and all punctuation — keep letters only. Bag-of-words doesn't understand punctuation
  or casing, so we throw it away.
- **BERT version (light):** only remove URLs and `@mentions`. **Keep casing and
  punctuation**, because BERT was trained on natural text and uses those cues.

```python
# TF-IDF: keep letters only
text = re.sub(r'[^a-z\s]', ' ', text.lower())

# BERT: just remove URLs + mentions, leave the rest
text = re.sub(r'http\S+|@\w+', ' ', text)
```

This difference matters: **match your preprocessing to your vectorizer.**

---

## Part B — The vectorizer (the ONE real difference)

### B1. TF-IDF (`kmeans_cyberbullying.ipynb`)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])   # shape: (15929, 5000)
```

**What it does:** builds a vocabulary of the 5,000 most useful words, then gives each
tweet a 5,000-long vector. Each number = "how important is this word in this tweet,
relative to the whole corpus" (Term Frequency × Inverse Document Frequency).

- **Strengths:** fast, no downloads, fully explainable (you can read the vocabulary).
- **Weakness:** it only knows *word overlap*. "kid" and "child" are unrelated to it.
  Two tweets meaning the same thing with different words look completely different.
- The matrix is **sparse** (mostly zeros) and **high-dimensional** (5,000 columns).

### B2. BERT (`kmeans_cyberbullying_bert.ipynb`)

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
X = model.encode(df['clean_text'].tolist(),
                 batch_size=64,
                 normalize_embeddings=True)       # shape: (15929, 384)
```

**What it does:** runs each tweet through a pre-trained Sentence-BERT model that outputs
a 384-dimensional **embedding** — a dense vector that encodes the tweet's *meaning*.
Semantically similar tweets land near each other even with zero shared words.

- **Strengths:** captures meaning/context; far better clusters (see results below).
- **Cost:** downloads a ~90 MB model once; encoding ~16k tweets takes ~1–2 min on CPU;
  needs the `sentence-transformers` + `torch` libraries.
- The matrix is **dense** (real numbers everywhere) and lower-dimensional (384 columns).
- `normalize_embeddings=True` makes vectors unit-length so Euclidean distance behaves
  like cosine similarity — the right notion of "close" for text embeddings.

---

## Part C — Clustering (identical again)

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=23, n_init=10)
df['cluster'] = kmeans.fit_predict(X)
```
- `n_clusters=2` → the "two groups" (k=2).
- `random_state=23` → reproducible runs.
- `n_init=10` → K-Means restarts 10 times from different seeds and keeps the best,
  guarding against a bad random start.

---

## Part D — Visualization & evaluation

### D1. Reduce to 2D so we can plot it
Both X matrices have hundreds/thousands of dimensions — impossible to plot directly.
So we **project to 2D purely for the picture** (this does not affect clustering).

```python
from sklearn.decomposition import PCA
X_2d = PCA(n_components=2, random_state=23).fit_transform(X)   # TF-IDF needs X.toarray()
```
Then scatter twice, side by side: colored by **cluster** vs colored by **true label**.
If the two pictures look alike, the unsupervised clustering matched reality.

The BERT notebook adds more visuals:
- **class-balance** bar + pie chart,
- **cluster-size vs true-size** bars,
- a **t-SNE** scatter (slower, on a 3,000-tweet sample — usually separates groups more
  clearly than PCA),
- a **seaborn confusion-matrix heatmap**.

### D2. Align clusters before scoring
K-Means labels clusters arbitrarily — its "cluster 1" might be your "label 0". Fix it
by flipping if that improves agreement:
```python
agreement = (df['cluster'] == df['label']).mean()
if agreement < 0.5:
    df['cluster_aligned'] = 1 - df['cluster']
else:
    df['cluster_aligned'] = df['cluster']
```

### D3. Score it
```python
accuracy = (df['cluster_aligned'] == df['label']).mean()   # matches the labels?
silhouette = silhouette_score(X, df['cluster'])            # are clusters well separated?
```
- **Agreement/accuracy** — how often the cluster matches the true label (0–100%).
- **Silhouette** (−1..1) — purely geometric: how tight/separated the clusters are.
  Text clouds overlap a lot, so this is usually low even when accuracy is high.

---

## Results — TF-IDF vs BERT

| Metric | TF-IDF | BERT |
|---|---|---|
| Feature vector | 5,000-dim sparse (word counts) | 384-dim dense (meaning) |
| Cluster-vs-label agreement | **77.1%** | **89.0%** |
| Cluster balance (sizes) | 11466 / 4463 (lopsided) | 7538 / 8391 (near the true 50/50) |
| Silhouette | — | 0.088 (low — clouds overlap) |
| Setup cost | none, instant | ~90 MB model, ~1–2 min encode |

**Why BERT won:** it understands that "kid", "child", "high school", and "teenager"
are related, so age-bullying tweets grouped together even when worded differently.
TF-IDF only sees exact words, so it split the age tweets and left half mixed in with
`not_cyberbullying`.

**Representative BERT clusters (sanity check):**
- Cluster 0 → reality-TV chatter: *"NOOOO! #MKR"*, *"TWITTER IS ABOUT TO EXPLODE! #mkr"*
- Cluster 1 → school bullying: *"every girl who bullied me in high school"*

---

## Takeaways

1. **Vectorization is the whole game.** K-Means, plotting, and scoring are identical;
   the quality of results is decided entirely by how you turn text into vectors.
2. **Match preprocessing to the vectorizer** — strip hard for bag-of-words, keep it
   natural for BERT.
3. **Meaning beats word-matching** for messy text: BERT's 89% vs TF-IDF's 77%.
4. **Unsupervised ≠ using labels.** Labels only ever appear in the evaluation step.
5. **Dimensionality reduction (PCA/t-SNE) is for the eyes only**  never cluster on the
   2D projection; cluster on the full matrix.

---

## KNN note — why "training accuracy" can lie (and how we fix it)

*(This applies to the KNN classifier we build next, not to K-Means above — but it's the
single most important gotcha to understand, so it lives here.)*

**KNN doesn't really "learn" — it memorizes.** Its whole training step is: keep a copy
of every training tweet (as an embedding) plus its label. To classify a new tweet, it
finds the closest stored tweet(s) and copies their label. That's it.

**The trap.** If you measure accuracy *on the training data itself* with **k=1** (look at
the single nearest neighbor), each tweet's nearest neighbor is **itself** — distance
zero, a perfect match. So KNN copies its own label and is right every time →
**~100% "accuracy."**

That number is meaningless. It's like grading an open-book exam where the questions are
the exact flashcards the student is holding: scoring 100% only proves they can look
things up, not that they learned anything that transfers to *new* tweets.

**How we fix it — two safeguards:**

1. **Cross-validation when choosing k.** Split the training data into chunks ("folds").
   Put some folds into KNN's memory and score it on a *different* fold held **out** of
   memory. Now the tweet being scored isn't in the memory, so it can't find itself — the
   score is honest. This is also how we discover that a larger k (e.g. 15) generalizes
   better than k=1.

2. **Report the held-out test set.** The headline accuracy comes from the 20% of tweets
   the model **never stored**. They're strangers to it, so their accuracy is the real,
   trustworthy result.

**One-line summary:** k=1 on training data is a *mirror* (it just sees itself);
cross-validation and the separate test set are the *honest exams*.

---

## Two different "k"s (don't mix them up)

- **6 = number of classes/labels** — the categories we sort tweets into (`religion`,
  `age`, `gender`, `ethnicity`, `not_cyberbullying`, `other_cyberbullying`). Fixed by the
  dataset.
- **k = number of neighbors** in K**N**N — how many nearby training tweets get to *vote*
  on a prediction. A knob **we** tune.

They're unrelated — you can have 6 classes and k=9 neighbors. Whenever this guide says
"the value of k", it means the neighbor count, not the class count.

---

## How we pick k (the number of neighbors)

We don't guess k — we let the data choose it, using **cross-validation on the training
set only**. The test set stays locked away until the very end.

**Step A — split the training data into folds.** Chop the training set into 5 equal
parts ("folds").

**Step B — for each candidate k (we try k = 1, 2, 3, … 30):**

1. Train on 4 folds, test on the 1 held-out fold → get an accuracy.
2. Rotate so each fold is the held-out one exactly once → 5 accuracies.
3. Average them → **one honest score for this k**.

**Step C — plot and pick.** Plot *average CV accuracy vs k* and choose the k with the
highest score (`best_k`).

```text
CV accuracy
  ^
  |        .-•-.___
  |     •-'        '-•-•___        <- peak here = best_k
  |   •'                   '-•-.
  | •  (k=1: too noisy)          (large k: over-smoothed)
  +--------------------------------> k
    1   3   5   ...            30
```

**Why sweep instead of hard-coding a number?** The best k depends on the data:

- **k too small (like 1):** the prediction rides on a single neighbor → jumpy and
  noise-sensitive (and on training data it fakes ~100% by matching itself).
- **k too large:** you average over so many neighbors that the class boundaries blur and
  the model drifts toward just predicting the biggest class.
- The sweep finds the sweet spot **empirically**.

**The discipline that keeps it honest:** all k-tuning happens *inside the training data*.
Only after `best_k` is chosen do we train once on the full training set and report
accuracy on the untouched **test** set — no peeking, no leakage.

The KNN vote itself, once k is fixed: embed the tweet → find its k nearest training
tweets (Euclidean distance) → those neighbors' labels vote → the label with the most
votes among the 6 classes wins.

---

## Environment notes
- Everything runs in the project `.venv`.
- Extra packages installed for these notebooks: `pandas`, `sentence-transformers`
  (brings `torch` + `transformers`), `seaborn`.
- The BERT model (`all-MiniLM-L6-v2`) downloads once to the Hugging Face cache; after
  that it runs offline.

---
---

# Today's learnings — 20 July 2026

## The task

> Take the Cats vs Dogs images, split them **two different ways** — once with **K-means**
> (unsupervised, k=2) and once with **KNN** (supervised, 2 groups) — and measure **how much the
> two splits disagree** using a metric called **M**. Then do it again with the *only* change being
> how an image is turned into numbers: **raw pixels** first, then **pretrained ResNet50 embeddings**.

Notebooks produced today:

- `knn_vs_kmeans_M.ipynb` — the **raw-pixel** version (grayscale 64×64 → 4096 features)
- `knn_vs_kmeans_M_resnet.ipynb` — the **ResNet50 embedding** version (RGB 224×224 → 2048 features)

The design is a **controlled experiment**: K-means, KNN, and the M formula are byte-for-byte
identical between the two notebooks. Only the feature extractor changes. So any change in M is
caused by the **representation**, not the algorithms.

---

## 1. The M metric

$$M = \frac{1}{N}\sum_{i=1}^{N}\big(\text{label}^{\text{KMeans}}_i - \text{label}^{\text{KNN}}_i\big)^2$$

The labels are 0/1, so each squared term is **1 when the two methods disagree** on an image and
**0 when they agree**. The sum is therefore just *the number of disagreeing images*, and dividing
by N puts M in **[0, 1]**.

**M is the disagreement rate. Agreement = 1 − M.**

| M | Meaning |
|---|---|
| 0 | the two partitions are **identical** |
| 0.20 | they disagree on 20% of images (agreement 0.80) |
| 1 | they disagree on **every** image |

### The label-flip fix (mandatory)

K-means cluster IDs are **arbitrary** — its "cluster 0" might be the dogs. Counting disagreements
without fixing that would inflate M for a purely cosmetic reason.

```python
def align_kmeans_to_knn(reference, clusters):
    asis = (clusters != reference).sum()
    flip = ((1 - clusters) != reference).sum()
    if flip < asis:
        return (1 - clusters), int(flip), True
    return clusters, int(asis), False
```

Try both mappings, keep whichever agrees more. This is the same idea as `cluster_aligned` in
Part 1 — it shows up every time you compare an unsupervised labelling to anything else.

---

## 2. The results

| Metric | Raw pixels | **ResNet50** |
|---|---|---|
| Features | 4,096 (grayscale 64×64) | 2,048 (frozen ResNet50) |
| **M** | **0.4715** | **0.0450** |
| Agreement (1 − M) | 0.529 | **0.955** |
| Disagreeing images | 943 / 2000 | **90 / 2000** |
| KNN accuracy vs true labels | 0.537 | **0.987** |
| K-means accuracy vs true labels | — | **0.956** |
| ARI (flip-proof) | ≈ 0 | **0.828** |
| NMI (flip-proof) | ≈ 0 | **0.778** |
| Flip applied | no | no |

**M fell by 90%** — same algorithms, same data, same metric.

*(The raw-pixel ARI/NMI are recalled from that run, not re-verified — recompute before quoting.)*

### Three findings

**(a) The unsupervised method nearly recovered the true classes on its own.** K-means scored
**0.956** against the real cat/dog labels *without ever seeing a label*. On raw pixels this was
impossible — it grouped by brightness and background instead. The "closest to centroid" figure
shows it plainly: cluster 0's 8 most typical members are **8 cats**, cluster 1's are **8 dogs**.

**(b) The 90 disagreements are structural, not noise.** The contingency table is completely
one-sided:

|  | KNN group 0 | KNN group 1 |
|---|---|---|
| **K-means cluster 0** | 912 | **0** |
| **K-means cluster 1** | 90 | 998 |

That **zero** is the interesting part. K-means cluster 0 is a **pure subset** of the cats — it never
once claims a dog. K-means errs in one direction only: it sweeps 90 cats into its dog cluster. The
agreement map confirms it geometrically — the red (disagree) points form a tight vertical band right
at the PC1 boundary between the two lobes. These are genuinely ambiguous images, not random failures.

**(c) Two independent, flip-proof metrics agree.** ARI 0.828 and NMI 0.778 don't care what the
labels are *called*, so the alignment step can't fool them. They tell the same story as M, which
means M isn't an artefact of how we aligned.

---

## 3. ⚠️ The most important lesson: M is only interpretable if KNN is accurate

The headline reading of M is *"how well does K-means recover the true classes on its own?"*
**That reading is only valid if KNN actually tracks the true classes.**

M compares K-means to **KNN**, not to the truth. KNN is a *stand-in* for the truth — and a stand-in
is only as good as its accuracy.

- **Raw-pixel run: KNN scored 0.537 — barely above a coin flip (0.50).** So M = 0.47 there was
  **not** measuring "K-means failed to find cats and dogs." It was measuring the disagreement
  between **two near-arbitrary partitions**. The number was real; the interpretation was not.
  *(The first notebook's write-up made exactly that claim, and we corrected it.)*
- **ResNet run: KNN scored 0.987.** Now the licence holds and M can be read the intended way.

> **Rule: never report M alone. Always report M next to KNN accuracy**, because KNN accuracy is
> what tells you whether M means anything at all.

That is why the final chart puts the two panels side by side, with a dashed "coin flip" line at
0.50 on the accuracy panel.

---

## 4. The one change that mattered — pixels vs embeddings

### Raw pixels (notebook 1)
```python
img = Image.open(path).convert('L').resize((64, 64))   # grayscale
X_row = np.asarray(img).flatten() / 255.0              # 4096 numbers
```
Each feature is **one pixel's brightness**. Distance between two images = "do these two pictures
have similar light and dark in the same places?" That is a question about *photography*, not about
*animals*. Which is why K-means clustered by background and exposure.

### ResNet50 embeddings (notebook 2)
```python
backbone = ResNet50(weights='imagenet', include_top=False, pooling='avg')  # -> 2048-D
X = backbone.predict(batch)
X = X / np.linalg.norm(X, axis=1, keepdims=True)       # L2-normalize
```
Each feature is a **learned visual concept** (fur texture, ear shape, snout). Distance now means
"do these two images contain similar *things*?"

Three arguments doing real work:

- `weights='imagenet'` — load the pretrained weights. `None` would give a random, useless network.
- `include_top=False` — **chop off the 1000-class classifier.** We don't want ResNet's *guesses*,
  we want the internal representation it built *before* guessing.
- `pooling='avg'` — without the top, output is a `7×7×2048` feature *map*; averaging over the 7×7
  grid flattens it to one `2048` vector per image.

`backbone.trainable = False` — **nothing is trained anywhere.** The backbone is a pure feature
extractor. This keeps the comparison honest: we upgraded the *input*, not the *algorithms*.

---

## 5. ⚠️ The objection to raise yourself — ImageNet already knows cats and dogs

**ResNet50 was pretrained on ImageNet, which contains ~120 dog breeds and several cat classes.**
So the embedding space was *built* with supervised knowledge of exactly the distinction we're testing.

Presenting "unsupervised K-means recovers cats vs dogs at 95.6%" without this caveat looks like a
missed flaw. The honest framing:

> The K-means step is unsupervised **given the features**, but the features themselves carry
> supervision transferred from ImageNet. What we showed is not *"clustering can discover species
> from nothing."* It is: **once a representation encodes semantic structure, unsupervised similarity
> becomes nearly equivalent to semantic similarity — which on raw pixels it emphatically is not.**

Still a real result (it's the whole rationale for transfer learning), but state it precisely.
**Next step to close the gap:** rerun with a *self-supervised* backbone (SimCLR, DINO, MAE) that
never saw a class label. If M stays low, the claim gets much stronger.

---

## 6. Technical lessons worth keeping

### `preprocess_input`, never `/255`
ResNet50 expects **BGR** channel order with the ImageNet mean subtracted — not 0–1 floats, not
0–255. Using `/255` **doesn't crash**; it just silently degrades every embedding. Always import the
`preprocess_input` that ships with your specific backbone.

### L2-normalise, and then *don't* standardise
```python
X = X / np.linalg.norm(X, axis=1, keepdims=True)
```
`axis=1` computes each **row's** length; `keepdims=True` keeps the shape `(N, 1)` so the division
broadcasts row-wise. Every vector ends up length 1, on a unit sphere — only *direction* survives,
magnitude is discarded.

**This is why no `StandardScaler` is used.** On the unit sphere, Euclidean distance is a monotone
function of cosine similarity — the geometry these features are meant to be compared in.
Standardising per-dimension afterwards would destroy the unit norm and amplify low-variance noise
directions. *(Same reason `normalize_embeddings=True` was right for BERT in Part 1.)*

### `cross_val_predict`, not `fit` + `predict`
```python
knn_labels = cross_val_predict(KNeighborsClassifier(n_neighbors=5), X, y, cv=5)
```
If KNN predicts data it was fitted on, **every image finds itself as its own nearest neighbour** —
distance zero, guaranteed correct. This is the *exact same trap* documented in Part 1's "why
training accuracy can lie". `cross_val_predict` gives every point a label from a model trained on
the **other** folds, so the self-match is impossible and the accuracy is honest.

### PCA is for the eyes only
The 2-D projection explains just **15.4%** of the variance. All clustering and classification run on
the full **2,048** dimensions. Say this before anyone asks why the scatter plots look imperfect.
*(Same rule as Part 1: never cluster on the 2-D projection.)*

### Stream batches — don't stack the whole dataset
The original plan did `np.stack([image_to_tensor(p) for p in paths])`, which allocates
`2000 × 224 × 224 × 3 × 4 bytes ≈ 1.2 GB` **before the first prediction runs**, and decoded every
JPEG twice. Rewritten to embed 32 at a time: peak memory drops to tens of MB, each file is decoded
once, and the result is identical.

The batching loop also guarantees **row alignment** — a path is only recorded *after* its image
decoded successfully:
```python
try:
    arrays.append(image_to_tensor(p))   # if this throws...
    ok.append(p)                        # ...this never runs -> stays aligned
except Exception:
    continue                            # corrupt JPEG, skip
```
This matters because a 2048-D embedding **cannot be reshaped back into a picture** — to show
"what's in this cluster" you must load the original file from a path that still matches row *i*.
(The dataset has two zero-byte files: `Cat/666.jpg` and `Dog/11702.jpg`.)

### ⚠️ The cache trap
```python
if Path(EMB_CACHE).exists():   # only asks "does the file exist?"
```
It does **not** ask "was this built with the current settings." Change `N_PER_CLASS` from 1000 to
12500, hit run, and it happily loads the **old 2,000-row file** — no error, no warning, wrong data.

**Fix: put the settings in the filename**, so each config gets its own cache automatically:
```python
EMB_CACHE = f'resnet_embeddings_{N_PER_CLASS}_{IMG_SIZE}.npz'
```

### Sorting filenames is alphabetical, not numeric
`sorted()` gives `1.jpg, 10.jpg, 100.jpg, 1000.jpg, 10000.jpg…` — **not** 1, 2, 3. Taking "the first
1000" gives a scattered sample, not images 1–1000. Harmless here (still an arbitrary sample) but
worth knowing before it confuses you.

---

## 7. Debugging lessons from today

1. **A notebook that "doesn't work" is often one broken line.** The first notebook failed entirely
   because an f-string had been split mid-word across two source lines
   (`f'Dog fold` / `er not found...'`) → `SyntaxError`. Since that cell defined `X`, `y`, and `N`,
   *every* downstream cell died with cascading `NameError`s. **Read the first error, not the last.**
2. **Always render a figure and look at it.** A validator checks colour, not layout. Two defects
   were invisible in code and obvious on sight:
   - the two red "disagree" ribbons cross at dead centre, so their labels landed on the **identical
     point** — one number completely hidden under the other. Fixed by placing them at t=0.24 and
     t=0.76 along the curve instead of both at t=0.5.
   - when a flow count is **0**, the ribbon has zero width but the label still drew — a stray
     "0 / 0.0%" floating with nothing under it. Fixed by skipping zero-count flows.
3. **Don't let a download progress bar into a saved notebook.** The ImageNet weights download wrote
   ~200 KB of ANSI progress-bar junk into the stored cell output. Re-running once the weights and
   embeddings were cached produced a clean file.
4. **Never `Read` an executed notebook directly** — after execution it's 1–2.5 MB of base64 PNGs.
   Use `nbformat` in a script to inspect sources and text outputs, and extract images to disk to
   view them.

---

## 8. Takeaways

1. **The representation is the whole game.** Identical K-means, identical KNN, identical metric —
   M went from 0.47 to 0.045 purely by changing how an image becomes numbers. This is the **same
   lesson as Part 1** (TF-IDF 77% → BERT 89%), now demonstrated on images with a controlled
   before/after.
2. **A metric can be computed correctly and still be uninterpretable.** M was always "right"; at
   KNN = 0.537 it just didn't mean what we wanted it to mean. **Always check the assumption a
   metric rests on before reading it.**
3. **Unsupervised ≠ label-free.** K-means used no labels — but its *features* were built on
   ImageNet labels. Be precise about where supervision entered.
4. **Look at your errors, not just their count.** "90 disagreements" is a number; "all 90 in one
   cell, sitting on the decision boundary, K-means never mislabels a dog as a cat" is a *finding*.
5. **Guard against silent wrongness** — self-neighbour leakage, stale caches, `/255` instead of
   `preprocess_input`. None of these throw an error. They just quietly give you the wrong answer.

---

## Environment notes (today)
- `tensorflow` **2.21.0** installed into the project `.venv` (alongside the existing `torch`).
- **TensorFlow has no GPU support on native Windows for ≥ 2.11** — CPU only. Use WSL2 for GPU.
- ResNet50 ImageNet weights (~100 MB) download once to `~/.keras`, then run offline.
- `resnet_embeddings.npz` ≈ 16 MB for 2,000 images (≈ 205 MB if scaled to the full ~25,000).
- Notebooks are built programmatically with `nbformat` and executed with
  `nbconvert --execute --inplace`.
