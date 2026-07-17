# Plan — KNN Classification of Cyberbullying Tweets (all 6 classes) with BERT embeddings

## Objective

Build a **supervised text classifier** that predicts the cyberbullying category of a
tweet across **all 6 classes**, using:

- **BERT sentence embeddings** (`all-MiniLM-L6-v2`) to turn tweets into numeric vectors,
- a **K-Nearest-Neighbors (KNN)** classifier,
- a proper **train/test split**,
- **k tuned by cross-validation** (sweep k, pick the best),
- **Euclidean distance** on normalized embeddings,

and to report **accuracy** plus a rich set of **graphs and tables** for both the
training and test phases.

> Note on why this is clean: unlike K-Means (unsupervised), **KNN is supervised**.
> Train/test split and accuracy are the *native* way to evaluate it — no cluster-to-class
> mapping tricks needed. The labels are used directly for both training and scoring.

---

## The 6 classes

`religion`, `age`, `gender`, `ethnicity`, `not_cyberbullying`, `other_cyberbullying`
— roughly balanced (~7,800–8,000 tweets each, ~47.7k total before cleaning).

We will encode them as integers `0..5` and keep a mapping for readable plots/tables.

---

## Key design decisions (confirmed)

| Decision | Choice | Why |
|---|---|---|
| Classes | all 6 | full multiclass problem |
| Vectorizer | BERT `all-MiniLM-L6-v2` (384-dim, normalized) | captures meaning; already cached |
| Classifier | KNN | the task; pairs naturally with embeddings |
| k selection | sweep k = 1..30, cross-validated, pick best | shows accuracy-vs-k curve, avoids guessing |
| Distance | Euclidean | straight-line distance; on normalized embeddings it ranks neighbors the same as cosine |
| Split | stratified train/test (80/20) | preserves class balance in both sets |

---

## Pipeline — step by step

### Step 1 — Load & inspect
- `pd.read_csv('cyberbullying_tweets.csv')`.
- Print shape and `value_counts()` of `cyberbullying_type`.
- **Visual:** bar chart of the full class distribution.

### Step 2 — Clean & deduplicate
- Remove duplicate rows (`drop_duplicates`).
- Light cleaning for BERT: strip URLs and `@mentions`, collapse whitespace.
  Keep casing/punctuation (BERT uses them).
- Report how many duplicates were dropped.

### Step 3 — Encode labels
- Map the 6 class strings → integers `0..5` (store both directions).
- **Visual:** a small table showing the label ↔ integer mapping.

### Step 4 — Train/test split (BEFORE embedding-independent leakage concerns)
- `train_test_split(..., test_size=0.2, stratify=y, random_state=23)`.
- **Visual:** side-by-side bar chart of class counts in train vs test (to confirm the
  stratified split kept classes balanced).
- *Leakage note:* BERT is pre-trained and frozen, so encoding each tweet independently
  introduces no train→test leakage. KNN itself stores only the **training** set as its
  reference; the test set is never seen during "fitting".

### Step 5 — BERT embeddings
- Load `SentenceTransformer('all-MiniLM-L6-v2')`.
- Encode **train** and **test** texts separately with `normalize_embeddings=True`
  (unit-length vectors, so Euclidean distance ranks neighbors the same way cosine would).
- Result: `X_train` (n_train × 384), `X_test` (n_test × 384).
- **Visual:** 2D projections of the training embeddings —
  - **PCA** scatter colored by class,
  - **t-SNE** scatter (on a sample, e.g. 4–5k points) colored by class —
  to *see* whether the 6 classes are separable in embedding space.

### Step 6 — Tune k (cross-validation sweep)
- For k = 1..30, run `KNeighborsClassifier(n_neighbors=k, metric='euclidean')`
  with `cross_val_score` (e.g. 5-fold) on the **training set only**.
- Pick `best_k` = k with highest mean CV accuracy.
- **Visual:** line chart **CV accuracy vs k**, with the chosen k marked.
- **Table:** top few k values with their mean ± std CV accuracy.

### Step 7 — Train final model
- Fit `KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')` on the full
  training set.
- (KNN "training" = storing the training vectors + labels.)

### Step 8 — Evaluate on TRAIN (sanity / overfitting check)
- Predict on the training set.
- **Metrics:** train accuracy.
- **Visual:** confusion matrix heatmap (train).
- Purpose: compare with test to gauge over/under-fitting. (For k=1 train accuracy is
  ~100% by construction — we'll call this out.)

### Step 9 — Evaluate on TEST (the real result)
- Predict on the held-out test set.
- **Metrics:**
  - overall **accuracy**,
  - **macro & weighted precision / recall / F1**,
  - full **classification report** (per-class precision/recall/F1/support).
- **Visuals:**
  - **confusion matrix heatmap** (test) with class names,
  - **per-class F1 (or accuracy) bar chart** to see which categories are easy/hard,
  - a **normalized confusion matrix** (row %) to read misclassification patterns.

### Step 10 — Error analysis (qualitative)
- **Table:** a handful of misclassified test tweets with `true` vs `predicted` labels
  — shows *where and why* KNN struggles (e.g. `other_cyberbullying` vs a specific type).

### Step 11 — Summary dashboard
- One consolidated section with:
  - a **metrics summary table** (train vs test accuracy, macro-F1, best k),
  - the key charts side by side,
  - a short written takeaway.

---

## Visual & table inventory (what "better visual representation" means here)

**Graphs**
1. Full class-distribution bar chart.
2. Train vs test class-count grouped bars (stratification check).
3. PCA 2D scatter of embeddings by class.
4. t-SNE 2D scatter of embeddings by class.
5. CV accuracy-vs-k line chart (with chosen k marked).
6. Confusion matrix heatmap — train.
7. Confusion matrix heatmap — test (raw counts).
8. Normalized confusion matrix heatmap — test (row %).
9. Per-class F1 bar chart.

**Tables**
1. Label ↔ integer mapping.
2. Top-k CV results (k, mean acc, std).
3. Full classification report (per-class P/R/F1/support).
4. Metrics summary (train vs test).
5. Sample misclassified tweets (true vs predicted).

Styling: `seaborn` heatmaps, consistent colors, value annotations on bars,
titles/labels on every axis.

---

## Deliverables

- `knn_cyberbullying_bert.ipynb` — the notebook implementing all steps above, executed
  with outputs saved.
- (Optional) append a "KNN results" section to `learnings.md` mirroring the K-Means guide,
  so the repo tells the full **K-Means vs KNN** story its name promises.

---

## Gaps / risks I'm flagging up front

1. **`other_cyberbullying` is a fuzzy catch-all.** It likely overlaps semantically with
   the specific categories and will probably be the hardest class (lowest F1). The
   confusion matrix and error-analysis table are there specifically to expose this.
2. **KNN with k=1 memorizes the training set** → ~100% train accuracy that means nothing.
   That's exactly why we tune k by cross-validation and always report the **test** number.
3. **Runtime:** encoding ~48k tweets with BERT on CPU takes a few minutes; the k-sweep
   with 5-fold CV over 30 values adds compute. We'll keep it reasonable (KNN predict is
   the cost, and Euclidean on 384-dim is fine). t-SNE runs on a sample only.
4. **Class balance is good** (~even), so plain accuracy is a fair headline metric — but
   we still report macro-F1 so no class hides behind the average.
5. **Reproducibility:** fixed `random_state=23` on the split and t-SNE.
6. **BERT dimensionality reduction (PCA/t-SNE) is for the eyes only** — KNN runs on the
   full 384-dim embeddings, never on the 2D projection.

---

## Open options (say the word and I'll fold them in)

- Add a **baseline** run (KNN on TF-IDF, or a majority-class baseline) to quantify how
  much BERT actually helps.
- Add **UMAP** as a third, often-cleaner 2D projection (needs an extra install).
- Save the trained pipeline + embeddings to disk so the notebook can be re-run fast.
- Compare against the earlier **K-Means** results in a single table (the repo's namesake).

If this plan looks right, I'll build and execute `knn_cyberbullying_bert.ipynb`.
