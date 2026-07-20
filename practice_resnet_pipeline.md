# Build It Yourself — ResNet50 → K-means vs KNN → M

A line-by-line practice run. **Part 1 is the game**: every line is described, you write it.
**Part 2 is the answer key**: the same template, filled in, with an explanation of each line.

Rules of the game:

- Work top to bottom. Each cell depends on the ones before it.
- Every line gets **2 hints**. Hint 1 tells you *what tool* to reach for. Hint 2 is usually a
  literal **search query** — paste it into Google or the library docs.
- Don't scroll to Part 2 until a cell runs (or genuinely refuses to).

**This is a simplified build.** The real notebook (`knn_vs_kmeans_M_resnet.ipynb`) adds batching,
an embedding cache, and corrupt-file handling. All of that is stripped out here so you can see the
skeleton. We use **200 images per class instead of 1000**, so it runs in a couple of minutes and
fits in memory in one go.

Setup: a `.venv` with `tensorflow`, `scikit-learn`, `numpy`, `matplotlib`, and the
`microsoft-catsvsdogs-dataset/PetImages/{Cat,Dog}` folders next to your notebook.

> **A note on the expected outputs:** exact numbers depend on your sample. The ones shown are from
> a 200-per-class run and yours may differ by a few hundredths — the *shapes* and the *rough
> magnitude* are what you're checking against.

---
---

# PART 1 — THE GAME

---

## Cell 01: Settings

*One place for every knob, so you never hunt for a magic number later.*

**Line 1:** Make a variable holding the image size ResNet50 expects: `224`.
- *Hint 1:* Just a plain integer assignment. Name it in CAPS — the convention for a constant.
- *Hint 2:* Google: `resnet50 input size 224x224 why`

**Line 2:** Make a variable for how many images to take from each class. Use `200`.
- *Hint 1:* Another plain integer. Cats and dogs get the same number so the classes stay balanced.
- *Hint 2:* Think about why balance matters: if you had 900 cats and 100 dogs, what would a model
  that always guesses "cat" score?

**Line 3:** Make a variable for `k`, the number of neighbours KNN votes with. Use `5`.
- *Hint 1:* Integer again. This is the `n_neighbors` argument you'll pass later.
- *Hint 2:* Google: `sklearn KNeighborsClassifier n_neighbors`

**Line 4:** Make a variable for the random seed. Use `42`.
- *Hint 1:* K-means starts from random centroids; a fixed seed makes your run reproducible.
- *Hint 2:* Google: `sklearn random_state reproducibility`

**Expected output:** nothing prints. If you get no error, the cell worked.

---

## Cell 02: Imports

*Pull in everything up front so a missing package fails now, not 10 minutes in.*

**Line 1:** Import `numpy` under its usual short name.
- *Hint 1:* The alias everybody uses is two letters.
- *Hint 2:* Google: `import numpy as np convention`

**Line 2:** Import `matplotlib`'s plotting module under its usual short name.
- *Hint 1:* You want the `pyplot` submodule, not the top-level package.
- *Hint 2:* Google: `import matplotlib.pyplot as plt`

**Line 3:** Import the `Path` class for handling file paths.
- *Hint 1:* It lives in the standard library module `pathlib`. No install needed.
- *Hint 2:* Google: `python pathlib Path tutorial`

**Line 4:** From ResNet50's Keras module, import both the **model class** and its
**`preprocess_input`** function.
- *Hint 1:* They live together in `tensorflow.keras.applications.resnet50`. One import line, two
  names.
- *Hint 2:* Google: `tensorflow.keras.applications.resnet50 preprocess_input`

**Line 5:** Import the two helpers that load an image file and convert it to an array.
- *Hint 1:* Modern Keras puts them in `tensorflow.keras.utils`. They're named after exactly what
  they do — one loads an image, one turns an image into an array.
- *Hint 2:* Google: `keras utils load_img img_to_array`

**Line 6:** Import the K-means class from scikit-learn.
- *Hint 1:* It's in the `cluster` module. The class name is capitalised oddly — two capitals.
- *Hint 2:* Google: `sklearn.cluster KMeans`

**Line 7:** Import the KNN **classifier** class from scikit-learn.
- *Hint 1:* It's in the `neighbors` module (American spelling). Careful — there's also a
  `KNeighborsRegressor`; you want the classifier.
- *Hint 2:* Google: `sklearn.neighbors KNeighborsClassifier`

**Line 8:** Import the function that produces cross-validated predictions.
- *Hint 1:* It's in `model_selection`. Not `cross_val_score` — you want the one that returns a
  **prediction for every row**, not a list of scores.
- *Hint 2:* Google: `sklearn cross_val_predict vs cross_val_score`

**Line 9:** Import `PCA`.
- *Hint 1:* It's in the `decomposition` module.
- *Hint 2:* Google: `sklearn.decomposition PCA`

**Expected output:** nothing, or possibly a few TensorFlow startup warnings about CPU
instructions. Warnings are fine; a `ModuleNotFoundError` is not.

---

## Cell 03: Collect the file paths and the true labels

*Before any machine learning: get a list of files, and a matching list of 0/1 answers.*

**Line 1:** Build a `Path` to the folder holding the two class folders:
`microsoft-catsvsdogs-dataset/PetImages`.
- *Hint 1:* `Path` lets you join folders with the `/` operator, which handles Windows vs Linux
  slashes for you.
- *Hint 2:* Google: `pathlib join paths with slash operator`

**Line 2:** Write a small function that takes a folder and returns the first `N_PER_CLASS` `.jpg`
files inside it, sorted.
- *Hint 1:* Three pieces: list the folder's contents, keep only files ending in `.jpg`, sort them,
  then slice the first N. A sorted list comprehension plus `[:N_PER_CLASS]` does it.
- *Hint 2:* Google: `pathlib iterdir filter by suffix` — and check what `.suffix` returns for
  `cat.JPG` (hint: watch the case).

**Line 3:** Call it on the `Cat` folder, and again on the `Dog` folder. Join the two lists into one
list of paths, cats first.
- *Hint 1:* Two lists become one with `+`.
- *Hint 2:* The **order matters** — everything after this assumes the first N rows are cats.

**Line 4:** Build the label array: `0` repeated N times for cats, then `1` repeated N times for
dogs.
- *Hint 1:* NumPy has a function that makes an array filled with a single repeated value. Then
  glue the two arrays together with a concatenation function.
- *Hint 2:* Google: `numpy full` and `numpy concatenate`

**Line 5:** Print how many paths and how many labels you have, to confirm they match.
- *Hint 1:* `len()` on both. If these two numbers ever disagree, stop and fix it — everything
  downstream silently breaks.
- *Hint 2:* You should see 400 and 400.

**Expected output:**

```text
400 images, 400 labels
```

---

## Cell 04: Turn one image into a tensor

*Get one image working before you loop over 400.*

**Line 1:** Define a function `image_to_tensor(path)`.
- *Hint 1:* Standard `def`. It takes one argument and returns one array.
- *Hint 2:* Writing this as a function (not inline) means Cell 06 stays a single readable line.

**Line 2:** Inside it, load the image at `path`, resized to `IMG_SIZE` × `IMG_SIZE`.
- *Hint 1:* `load_img` has an argument that does the resizing for you — you don't need PIL's
  `.resize()`. It takes a **tuple** of (height, width).
- *Hint 2:* Google: `keras load_img target_size`

**Line 3:** Convert that image to a NumPy array, then run it through `preprocess_input`, and return
the result.
- *Hint 1:* Two function calls, one wrapped around the other. `img_to_array` first, then
  `preprocess_input`.
- *Hint 2:* Google: `why preprocess_input instead of dividing by 255 resnet` — this is the single
  easiest thing to get silently wrong in the whole notebook.

**Line 4:** (Outside the function) call it on the first path and print the result's `.shape`.
- *Hint 1:* Every NumPy array has a `.shape` attribute — no parentheses, it's not a method.
- *Hint 2:* You're expecting three numbers: height, width, and colour channels.

**Expected output:**

```text
(224, 224, 3)
```

---

## Cell 05: Build the frozen ResNet50 feature extractor

*The one cell that turns "a picture" into "a list of numbers about the picture".*

**Line 1:** Create a `ResNet50` with three arguments: pretrained ImageNet weights, **no** classifier
head, and average pooling.
- *Hint 1:* The three argument names are `weights`, `include_top`, and `pooling`. One takes the
  string `'imagenet'`, one takes `False`, one takes the string `'avg'`.
- *Hint 2:* Google: `keras ResNet50 include_top=False pooling='avg' feature extraction` — and while
  you're there, work out what shape you'd get *without* `pooling='avg'`.

**Line 2:** Freeze it.
- *Hint 1:* Every Keras model has a boolean attribute controlling whether its weights can be
  updated. Set it to `False`.
- *Hint 2:* Google: `keras model trainable False freeze`

**Line 3:** Print the output shape of the model so you can see the vector length.
- *Hint 1:* Keras models expose `.output_shape`.
- *Hint 2:* The first entry will be `None` — that's the batch dimension, meaning "any number of
  images". The second number is the one you care about.

**Expected output:** the first run downloads ~100 MB of weights, then:

```text
(None, 2048)
```

---

## Cell 06: Embed every image

*400 pictures in, a 400 × 2048 table of numbers out.*

**Line 1:** Build one big array of all 400 preprocessed images by calling `image_to_tensor` on
every path.
- *Hint 1:* A list comprehension inside NumPy's "stack a list of same-shaped arrays into one
  bigger array" function.
- *Hint 2:* Google: `numpy stack list of arrays` — check the resulting shape is
  `(400, 224, 224, 3)`.

**Line 2:** Push that array through the backbone to get the features.
- *Hint 1:* Keras models have a `.predict()` method. Pass `verbose=0` to silence the progress bar.
- *Hint 2:* Google: `keras model predict verbose=0`

**Line 3:** Print the shape of the result.
- *Hint 1:* `.shape` again.
- *Hint 2:* Two numbers now: how many images, and how many features each.

**Expected output:** takes 30–90 seconds on CPU, then:

```text
X: (400, 2048)
```

---

## Cell 07: L2-normalise the vectors

*Make every vector the same length, so "close" means "similar content", not "similar brightness".*

**Line 1:** Divide `X` by the length of each row, keeping the shape so it broadcasts.
- *Hint 1:* NumPy's norm function lives in the `linalg` submodule. You need two arguments beyond
  the array: one saying "compute along the rows", one saying "keep the result 2-D".
- *Hint 2:* Google: `numpy linalg norm axis=1 keepdims` — then work out *why* `keepdims=True` is
  needed by trying it without and reading the broadcasting error.

**Line 2:** Print the length of the first row, rounded, to confirm it's now 1.
- *Hint 1:* Same norm function, applied to a single row (`X[0]`), no axis needed.
- *Hint 2:* Expect `1.0`. If you get something like `23.7`, line 1 didn't take effect.

**Expected output:**

```text
norm of row 0: 1.0
```

---

## Cell 08: K-means (the unsupervised split)

*Split the 400 vectors into 2 groups. The labels are not used here — at all.*

**Line 1:** Create a `KMeans` with 2 clusters, 10 restarts, and your random seed, and fit it on `X`.
- *Hint 1:* The three argument names are `n_clusters`, `n_init`, `random_state`. You can chain
  `.fit(X)` straight onto the constructor.
- *Hint 2:* Google: `sklearn KMeans n_init why multiple initializations`

**Line 2:** Pull the cluster assignment for every image out of the fitted object.
- *Hint 1:* Fitted scikit-learn estimators store results in attributes ending with an underscore.
  You want the one called `labels_`.
- *Hint 2:* Google: `sklearn KMeans labels_ attribute`

**Line 3:** Print how many images landed in each cluster.
- *Hint 1:* NumPy has a function that counts occurrences of each non-negative integer in an array.
- *Hint 2:* Google: `numpy bincount`

**Expected output:** two numbers that add to 400, roughly balanced:

```text
K-means cluster sizes: [198 202]
```

---

## Cell 09: KNN (the supervised split)

*The honest way: every prediction comes from a model that never saw that image.*

**Line 1:** Use `cross_val_predict` with a `KNeighborsClassifier` to get a predicted label for
every image, using 5 folds.
- *Hint 1:* Four arguments: the estimator, `X`, `y`, and `cv=5`. It returns one array the same
  length as `y`.
- *Hint 2:* Google: `sklearn cross_val_predict example` — and read up on *why* calling
  `.fit(X, y)` then `.predict(X)` would give you a near-perfect and completely fake score.

**Line 2:** Compute the accuracy: the fraction of predictions matching the true labels.
- *Hint 1:* Comparing two NumPy arrays with `==` gives an array of `True`/`False`. The **mean** of
  booleans *is* the fraction that are true.
- *Hint 2:* No sklearn import needed for this — one comparison and `.mean()`.

**Line 3:** Print that accuracy to 4 decimal places.
- *Hint 1:* An f-string with a format spec.
- *Hint 2:* Google: `python f-string format float 4 decimal places`

**Expected output:** should be well above 0.90 — if it's near 0.50, something is wrong upstream.

```text
KNN accuracy vs true labels: 0.9825
```

---

## Cell 10: Align the clusters, then compute M

*K-means' "cluster 0" might be the dogs. Fix that before counting disagreements.*

**Line 1:** Count the disagreements between `km_labels` and `knn_labels` as they are.
- *Hint 1:* `!=` between the two arrays gives booleans; `.sum()` counts the `True`s.
- *Hint 2:* This is your "as-is" count.

**Line 2:** Count the disagreements if you flipped K-means' labels (0↔1).
- *Hint 1:* For a 0/1 array, `1 - array` flips every value. No `if` statement needed.
- *Hint 2:* Convince yourself on paper: `1 - 0 = 1`, `1 - 1 = 0`. That's the whole trick.

**Line 3:** Keep whichever version disagrees less — store the aligned labels and the disagreement
count.
- *Hint 1:* A simple `if flip < asis:` / `else:` block assigning two variables each way.
- *Hint 2:* Think about what M would be if you skipped this step entirely and K-means happened to
  number its clusters the other way round. (Answer: about 1 minus the right answer.)

**Line 4:** Divide the disagreement count by the number of images to get **M**.
- *Hint 1:* `len(y)` is your N.
- *Hint 2:* Careful in older Python — make sure you're doing float division, not integer division.

**Line 5:** Print the disagreement count, M, and agreement (1 − M).
- *Hint 1:* One f-string, or three prints.
- *Hint 2:* M should be small if both methods found the same structure.

**Expected output:**

```text
disagreements: 12 of 400
M            : 0.0300
agreement    : 0.9700
```

---

## Cell 11: Sanity checks

*Two extra numbers that stop you fooling yourself.*

**Line 1:** Compute K-means' accuracy against the **true** labels, taking the better of the two
label orderings.
- *Hint 1:* Same flip trick as Cell 10 — compute both means and take `max()`.
- *Hint 2:* This is the number that tells you whether unsupervised clustering actually found cats
  vs dogs.

**Line 2:** Import and compute the Adjusted Rand Index between the K-means and KNN labellings.
- *Hint 1:* It's in `sklearn.metrics`, named `adjusted_rand_score`. It takes two label arrays.
- *Hint 2:* Google: `adjusted rand index intuition` — the key property is that it doesn't care what
  the clusters are *named*, so it can't be fooled by your alignment step.

**Line 3:** Print both, plus the KNN accuracy from Cell 09, together.
- *Hint 1:* Three lines of f-string printing.
- *Hint 2:* **Always print M next to KNN accuracy.** M compares K-means to KNN — if KNN is barely
  better than a coin flip, M is comparing two bad answers and means nothing.

**Expected output:**

```text
K-means accuracy vs true : 0.9625
ARI (K-means vs KNN)     : 0.8834
KNN accuracy vs true     : 0.9825
```

---

## Cell 12: Look at it

*2048 dimensions squashed to 2, purely so your eyes can check the story.*

**Line 1:** Fit a 2-component PCA on `X` and transform it in one call.
- *Hint 1:* Construct `PCA(n_components=2, random_state=...)` then use the method that fits and
  transforms in one step.
- *Hint 2:* Google: `sklearn fit_transform vs fit then transform`

**Line 2:** Make a figure with 1 row and 3 columns of axes.
- *Hint 1:* `plt.subplots` returns two things: a figure and an array of axes. Pass `figsize` to
  make it wide.
- *Hint 2:* Google: `matplotlib subplots 1 row 3 columns figsize`

**Line 3:** On each of the three axes, scatter the 2-D points, coloured by (a) the true labels,
(b) the aligned K-means labels, (c) the KNN labels.
- *Hint 1:* A `for` loop zipping the axes together with the three label arrays and three titles is
  much cleaner than writing `ax.scatter` three times.
- *Hint 2:* Google: `matplotlib scatter c= array of labels colormap` — column 0 of `X_2d` is your
  x, column 1 is your y.

**Line 4:** Show the plot.
- *Hint 1:* One function call.
- *Hint 2:* In a notebook you often get the plot anyway, but call it explicitly so the code also
  works as a script.

**Expected output:** three scatter plots side by side, each showing two blobs. The three panels
should look nearly identical — that visual sameness *is* what a low M means. Note the blobs will
overlap somewhat; the 2-D view keeps only ~15% of the information.

---
---

# PART 2 — THE ANSWER KEY

Same template, now with the code and an explanation of every line.

---

## Cell 01: Settings

```python
IMG_SIZE     = 224
N_PER_CLASS  = 200
K_NEIGHBORS  = 5
RANDOM_STATE = 42
```

**Line 1 — `IMG_SIZE = 224`.** ResNet50 was trained on 224×224 crops, so its filters expect features
at that scale. Feed it 64×64 and it still runs, it just produces worse vectors.

**Line 2 — `N_PER_CLASS = 200`.** Equal numbers from each class. With balanced classes, "50%" is
the score of a coin flip, which gives you a clean baseline to compare against.

**Line 3 — `K_NEIGHBORS = 5`.** How many neighbours vote on each prediction. `k=1` is jumpy and
noise-sensitive; very large `k` blurs the boundary. 5 is a reasonable default.

**Line 4 — `RANDOM_STATE = 42`.** K-means picks random starting centroids, and cross-validation
shuffles. Fixing the seed means your numbers are the same every run.

**Output:** *(nothing)*

---

## Cell 02: Imports

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.decomposition import PCA
```

**Lines 1–3** are the standard trio: arrays, plots, paths. `Path` beats string concatenation
because `DATA_DIR / 'Cat'` works identically on Windows and Linux.

**Line 4** imports the model **and its matching preprocessing function together**. That pairing is
deliberate — each Keras backbone ships its own `preprocess_input`, and they are not interchangeable.
Importing them on one line makes it hard to forget.

**Line 5** — `load_img` opens and resizes a file; `img_to_array` converts the PIL image to NumPy.
On older Keras 2 these lived in `tensorflow.keras.preprocessing.image` instead.

**Lines 6–7** — the two algorithms being compared. Note the different modules: `cluster` for the
unsupervised one, `neighbors` for the supervised one.

**Line 8** — `cross_val_predict`, not `cross_val_score`. The score version gives you 5 numbers; you
need one *label per image* to compare against K-means row by row.

**Line 9** — PCA, used only for the picture at the end.

**Output:** *(nothing, possibly TF warnings)*

---

## Cell 03: Collect the file paths and the true labels

```python
DATA_DIR = Path('microsoft-catsvsdogs-dataset') / 'PetImages'

def list_images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == '.jpg')[:N_PER_CLASS]

paths = list_images(DATA_DIR / 'Cat') + list_images(DATA_DIR / 'Dog')

y = np.concatenate([np.full(N_PER_CLASS, 0), np.full(N_PER_CLASS, 1)])

print(f'{len(paths)} images, {len(y)} labels')
```

**Line 1** builds the base path. Nothing touches the disk yet — a `Path` is just a string wrapper
until you actually read from it.

**Line 2 (the function)** does three things at once: `iterdir()` lists the folder, the `if` keeps
only JPEGs, `sorted()` makes the selection reproducible, and `[:N_PER_CLASS]` takes the first N.
`.suffix.lower()` guards against a file named `.JPG`.

⚠️ **`sorted()` is alphabetical, not numeric.** You get `1.jpg, 10.jpg, 100.jpg, 1000.jpg…`, not
1, 2, 3. So "the first 200" is a scattered sample, not images 1–200. Harmless here — it's still an
arbitrary sample — but surprising if you don't expect it.

**Line 3** concatenates cats then dogs. **Order is load-bearing**: line 4 assumes the first
`N_PER_CLASS` rows are cats.

**Line 4** builds the answer key: 200 zeros followed by 200 ones. `np.full(n, v)` makes an array of
`n` copies of `v`; `np.concatenate` joins them.

**Line 5** prints both lengths. If they ever differ, every row of your data is misaligned with its
label and *nothing downstream will error* — you'll just get garbage results. Cheap check, huge
payoff.

**Output:**

```text
400 images, 400 labels
```

---

## Cell 04: Turn one image into a tensor

```python
def image_to_tensor(path):
    img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    return preprocess_input(img_to_array(img))

print(image_to_tensor(paths[0]).shape)
```

**Line 2 — `load_img(..., target_size=...)`.** Opens the JPEG and resizes it in one step. The
result is a PIL image, not yet numbers.

**Line 3 — `img_to_array` then `preprocess_input`.** The first gives you a `(224, 224, 3)` float
array with values 0–255. The second converts that into what ResNet50 actually expects: **BGR channel
order with the ImageNet channel means subtracted.**

⚠️ **This is the classic silent bug.** Writing `img_to_array(img) / 255.0` instead **does not
crash**. It runs, produces vectors, and every single one is subtly wrong — your accuracy just comes
out mysteriously mediocre. Always use the `preprocess_input` that ships with your specific backbone.

**Line 4** confirms the shape before you loop 400 times. Height, width, 3 colour channels.

**Output:**

```text
(224, 224, 3)
```

---

## Cell 05: Build the frozen ResNet50 feature extractor

```python
backbone = ResNet50(weights='imagenet', include_top=False, pooling='avg')
backbone.trainable = False

print(backbone.output_shape)
```

**Line 1** — three arguments, each doing real work:

- `weights='imagenet'` — load the pretrained weights. `weights=None` would give you a randomly
  initialised network, which produces meaningless vectors.
- `include_top=False` — **chop off the 1000-class classifier.** You don't want ResNet's *guess*
  ("golden retriever, 0.82"), you want the internal representation it built *before* guessing.
  That representation is far richer.
- `pooling='avg'` — without the top, the output is a `7×7×2048` feature *map*. Averaging over the
  7×7 grid flattens it to one `2048`-number vector per image.

**Line 2 — `trainable = False`.** Nothing is trained anywhere in this notebook. The backbone is a
fixed function: picture in, 2048 numbers out. This is what makes the pixels-vs-embeddings
comparison honest — you changed the *input representation*, not the algorithms.

**Line 3** shows `(None, 2048)`. The `None` is the batch dimension: "this model accepts any number
of images at once."

**Output:**

```text
(None, 2048)
```

---

## Cell 06: Embed every image

```python
batch = np.stack([image_to_tensor(p) for p in paths])
X = backbone.predict(batch, verbose=0)

print('X:', X.shape)
```

**Line 1** decodes all 400 images and stacks them into one `(400, 224, 224, 3)` array.

⚠️ **This is the line that doesn't scale.** That array is
`400 × 224 × 224 × 3 × 4 bytes ≈ 240 MB` — fine. At the real notebook's 2000 images it's **1.2 GB
allocated before a single prediction runs.** That's exactly why the production version embeds in
batches of 32 and throws each batch away after predicting. Keep this simple version for learning,
but recognise the limit.

**Line 2 — `.predict()`** runs all 400 images through the frozen network. `verbose=0` suppresses
the progress bar, which otherwise dumps a lot of junk into your saved notebook.

**Line 3** — `(400, 2048)`. You've replaced 150,528 pixel values per image with 2,048 numbers, and
those 2,048 are *about the content*: fur texture, ear shape, snout. That's the whole point.

**Output:**

```text
X: (400, 2048)
```

---

## Cell 07: L2-normalise the vectors

```python
X = X / np.linalg.norm(X, axis=1, keepdims=True)

print('norm of row 0:', round(float(np.linalg.norm(X[0])), 4))
```

**Line 1** — three parts worth understanding separately:

- `np.linalg.norm(X, axis=1)` computes the **length of each row** — 400 numbers.
- `keepdims=True` keeps the result shaped `(400, 1)` instead of collapsing it to `(400,)`. That
  shape is what lets it broadcast correctly across the division, one divisor per row. Without it you
  get a broadcasting error — try it, the error is instructive.
- The division puts every vector on the **unit sphere**. Only *direction* survives; magnitude is
  discarded.

**Why bother?** On the unit sphere, Euclidean distance becomes a direct function of cosine
similarity — the notion of "similar" these embeddings are built for. Two photos of the same cat at
different exposures point the same direction but had different magnitudes; normalising makes them
neighbours.

**And why no `StandardScaler`?** Standardising per-dimension afterwards would destroy the unit norm
you just created, and would amplify low-variance noise dimensions. Normalise, then stop.

**Output:**

```text
norm of row 0: 1.0
```

---

## Cell 08: K-means (the unsupervised split)

```python
kmeans = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE).fit(X)
km_labels = kmeans.labels_

print('K-means cluster sizes:', np.bincount(km_labels))
```

**Line 1:**

- `n_clusters=2` — you're asking for two groups. K-means does not discover the number for you.
- `n_init=10` — restart from 10 different random seedings and keep the best result. K-means can
  converge to a bad local optimum from an unlucky start; this is cheap insurance.
- `random_state` — reproducibility.

**Line 2 — `labels_`.** The trailing underscore is scikit-learn's convention for "this attribute
was computed during `fit`". You get one cluster ID (0 or 1) per image.

**The key point: `y` appears nowhere in this cell.** K-means is genuinely unsupervised — it groups
by geometry alone. The labels only show up later, to *evaluate*.

**Line 3 — `np.bincount`** counts how many times each integer appears. Two roughly equal numbers is
a good sign; a 390/10 split would mean K-means found one dense blob and a handful of outliers
rather than the structure you wanted.

**Output:**

```text
K-means cluster sizes: [198 202]
```

---

## Cell 09: KNN (the supervised split)

```python
knn_labels = cross_val_predict(KNeighborsClassifier(n_neighbors=K_NEIGHBORS), X, y, cv=5)
knn_acc = (knn_labels == y).mean()

print(f'KNN accuracy vs true labels: {knn_acc:.4f}')
```

**Line 1 — why `cross_val_predict` and not `fit` + `predict`.** This is the most important
methodological line in the notebook.

KNN doesn't really learn — it memorises every training point. If you fit on `X` and then predict
on `X`, **every image finds itself as its own nearest neighbour**, at distance zero, with a
guaranteed-correct label. You'd score ~100% and it would mean absolutely nothing.

`cross_val_predict` splits the data into 5 folds and predicts each fold using a model trained on
the *other four*. The image being predicted is never in the model's memory, so it can't match
itself. Every one of the 400 labels comes back honest.

**Line 2** — `(knn_labels == y)` gives 400 booleans; `.mean()` treats `True` as 1 and `False` as 0,
so the mean *is* the accuracy. A neat NumPy idiom worth remembering.

**Line 3** — this number is the licence to interpret M at all. Above 0.95, KNN is a reliable
stand-in for the truth. Near 0.50, it's a coin flip and M becomes meaningless (see Cell 11).

**Output:**

```text
KNN accuracy vs true labels: 0.9825
```

---

## Cell 10: Align the clusters, then compute M

```python
asis = (km_labels != knn_labels).sum()
flip = ((1 - km_labels) != knn_labels).sum()

if flip < asis:
    km_aligned, disagreements = 1 - km_labels, flip
else:
    km_aligned, disagreements = km_labels, asis

M = disagreements / len(y)

print(f'disagreements: {disagreements} of {len(y)}')
print(f'M            : {M:.4f}')
print(f'agreement    : {1 - M:.4f}')
```

**Why this cell exists.** K-means cluster IDs are **arbitrary**. Nothing forces its "cluster 0" to
be the cats — a different random seed could number them the other way round. If you counted
disagreements without checking, you could report M = 0.97 for a *perfect* clustering, purely
because the two methods used opposite names for the same split.

**Lines 1–2** count disagreements both ways. `1 - km_labels` flips a 0/1 array (`1-0=1`, `1-1=0`) —
no loop, no `if`, no `np.where`.

**Lines 3–6** keep whichever mapping agrees more. Note it keeps *both* the aligned labels and the
count, so you don't recompute.

**Line 7 — M itself.** The formula is

$$M = \frac{1}{N}\sum_{i=1}^{N}\big(\text{label}^{\text{KMeans}}_i - \text{label}^{\text{KNN}}_i\big)^2$$

but since both labels are 0 or 1, each squared difference is **1 when they disagree and 0 when they
agree**. So the sum is literally just the count of disagreements — which is why counting with `!=`
gives the identical answer with no squaring in sight.

**M is the disagreement rate; agreement is 1 − M.** M = 0 means the two partitions are identical.

**Output:**

```text
disagreements: 12 of 400
M            : 0.0300
agreement    : 0.9700
```

---

## Cell 11: Sanity checks

```python
from sklearn.metrics import adjusted_rand_score

km_acc = max((km_labels == y).mean(), ((1 - km_labels) == y).mean())
ari = adjusted_rand_score(km_labels, knn_labels)

print(f'K-means accuracy vs true : {km_acc:.4f}')
print(f'ARI (K-means vs KNN)     : {ari:.4f}')
print(f'KNN accuracy vs true     : {knn_acc:.4f}')
```

**Line 2 — K-means vs the truth.** Same flip trick, now against `y`. This is the headline result:
a method that **never saw a single label** sorted cats from dogs at ~96%. On raw pixels this is
impossible — there, K-means groups by background and brightness instead, because pixel distance
asks "do these photos have light and dark in the same places?", a question about *photography*, not
about *animals*.

**Line 3 — ARI.** The Adjusted Rand Index compares two partitions **without caring what the groups
are called**, so your alignment step in Cell 10 cannot flatter it. It's an independent confirmation
that M isn't an artefact of how you aligned. ARI near 0 means "no better than random agreement";
near 1 means "the same partition".

**⚠️ Line 5's placement is the real lesson: never report M without KNN accuracy next to it.**

M measures how much K-means and KNN *disagree*. You'd like to read that as "how well did K-means
recover the true classes on its own" — **but that reading is only valid if KNN actually tracks the
truth.** KNN is a stand-in for the answer key, and a stand-in is only as good as its accuracy.

In the raw-pixel version of this experiment, KNN scored **0.537** — barely above a coin flip. M was
0.47 there, and that number was *computed perfectly correctly* while meaning nothing at all: it
measured the disagreement between **two near-arbitrary partitions**. Here KNN is at 0.98, so the
interpretation holds.

**A metric can be correct and still be uninterpretable. Always check the assumption it rests on.**

**Output:**

```text
K-means accuracy vs true : 0.9625
ARI (K-means vs KNN)     : 0.8834
KNN accuracy vs true     : 0.9825
```

---

## Cell 12: Look at it

```python
X_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)

for ax, labels, title in zip(axes,
                             [y, km_aligned, knn_labels],
                             ['True label', 'K-means (aligned)', 'KNN']):
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='coolwarm', s=10, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('PC1')

axes[0].set_ylabel('PC2')
plt.tight_layout()
plt.show()
```

**Line 1 — PCA.** Finds the 2 directions along which the 400 points vary most, and projects onto
them. `fit_transform` does both steps in one call.

**Line 2 — `subplots(1, 3)`** returns the figure plus an array of 3 axes. `sharex`/`sharey` force
identical scales across the panels, which matters — without it, the three plots would be
individually auto-scaled and you couldn't compare them by eye.

**Lines 4–9 — the loop.** `zip` pairs each axis with its label array and title, so `ax.scatter` is
written once instead of three times. `c=labels` colours each point by its group; `X_2d[:, 0]` is
every row's first column (PC1) and `X_2d[:, 1]` the second (PC2).

**⚠️ PCA is for the eyes only.** This 2-D projection keeps only around 15% of the variance — the
other 85% of what distinguishes these images is invisible in the picture. **Every calculation in
Cells 08–11 ran on all 2,048 dimensions.** Never cluster or classify on the 2-D projection, and
don't be alarmed that the blobs overlap more than the 96% accuracy suggests — that overlap is a
limitation of the drawing, not of the model.

**What to look for:** three panels that look nearly the same. That visual sameness *is* what a low
M means, made concrete.

**Output:** three side-by-side scatter plots, each showing two overlapping blobs split
left-to-right, all three colourings looking near-identical.

---
---

## Where to go next

Once this runs end to end, the interesting experiments are:

1. **Swap the representation, change nothing else.** Replace Cells 04–07 with grayscale 64×64 raw
   pixels (`Image.open(p).convert('L').resize((64,64))`, flattened, `/255`). Same K-means, same KNN,
   same M. Watch M jump from ~0.03 to ~0.47 and KNN accuracy collapse to ~0.54. That single
   controlled change is the entire thesis of the project.
2. **Raise `N_PER_CLASS`** and find out where your memory gives out — then implement the batched
   version to fix it.
3. **The honest objection to your own result:** ImageNet contains ~120 dog breeds and several cat
   classes, so ResNet50's embedding space was *built* with supervised knowledge of exactly the
   distinction you're testing. The K-means step is unsupervised *given the features*, but the
   features themselves carry transferred supervision. The clean way to close that gap is a
   self-supervised backbone (DINO, SimCLR, MAE) that never saw a class label. If M stays low there,
   the claim gets much stronger.
