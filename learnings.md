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
