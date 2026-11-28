# Day 12 — TF-IDF Document Search from Scratch

A tiny search engine in ~150 lines: give it a query, it ranks a small
corpus of documents by how relevant they are — using nothing but NumPy
and the TF-IDF idea. No sklearn, so every step of the maths is visible.

## The idea

A search engine has to turn text into numbers before it can rank
anything. TF-IDF does it in two parts:

1. **Term frequency (TF)** — how often a word appears in a document,
   divided by the document's length. A word that appears twice in a
   10-word document matters more than the same word twice in a
   100-word one.
2. **Inverse document frequency (IDF)** — how rare a word is across the
   whole corpus. Words that appear in nearly every document ("data" in
   a machine-learning corpus) carry little information; words that
   appear in one document ("neural") carry a lot.

Multiply them and each document becomes a vector. A query becomes a
vector the same way, and the **cosine similarity** (the cosine of the
angle between the two vectors) is the relevance score: 1.0 means the
query and the document point the same way, 0.0 means they share no
words at all.

## The maths — in brief

For a term `t` in document `d`, with `N` documents in the corpus and
`df(t)` the number of documents containing `t`:

```
tf(t, d)  = count(t, d) / |d|
idf(t)    = log((1 + N) / (1 + df(t))) + 1     (smoothed, sklearn-style)
tf-idf    = tf × idf
```

The `+1` smoothing stops a word that appears in every document from
getting an IDF of exactly zero, and stops a word in a single document
from exploding. Each TF-IDF row is then L2-normalised, so cosine
similarity is just a dot product — and document length stops mattering.

## The data

12 short documents written for this project, on four topics
(machine learning, space, cooking, cricket), so a good ranking is
checkable by eye. Stopwords ("the", "a", "is", …) are dropped before
counting. Fixed corpus — no randomness anywhere.

## How to run

```bash
python3 tfidf.py
```

Requires NumPy and matplotlib only. Prints the IDF of a rare vs a
common word, the top-3 documents for four test queries, and the
precision@1 score; saves `search_scores.svg`.

## Measured results

Corpus stats:

| Quantity | Value |
|---|---:|
| Documents | 12 |
| Vocabulary (after stopword removal) | 137 terms |
| TF-IDF matrix shape | 12 × 137 |
| Non-zero entries | 149 of 1,644 (9.1% dense) |

Rare words really do weigh more — the IDF check from the run:

| Word | Documents it appears in | IDF |
|---|---|---:|
| `neural` | 1 | **2.872** |
| `data` | 3 | 2.179 |

Ranking results — all four test queries put the intended document
first:

| Query | Top document | Similarity | 2nd place | Similarity |
|---|---|---:|---|---:|
| `neural network training` | ml-1 | **0.4115** | ml-3 | 0.1612 |
| `black hole gravity` | space-2 | **0.5113** | — (all 0.0000) | 0.0000 |
| `chocolate cake recipe` | food-1 | **0.3333** | — (all 0.0000) | 0.0000 |
| `cricket world cup` | sport-1 | **0.4718** | — (all 0.0000) | 0.0000 |

**Precision@1: 4/4 = 1.00.** The `search_scores.svg` bar chart shows the
same story for the first query: ml-1 towers over the rest, ml-3 gets a
small score because it shares the words "training" and "data", and
everything else scores zero.

## What I learned

- IDF is the whole trick: without it, a long document full of common
  words would beat a short document with the one rare word you actually
  searched for. The measured gap (2.872 vs 2.179) is small here because
  the corpus is tiny — on a real corpus with millions of documents the
  spread is far wider.
- A query word that appears in *no* document contributes nothing:
  "recipe" is not in any document, so the food query is really scored
  on "chocolate" + "cake" only — which is why its top score (0.3333)
  is lower than the others. Real search engines handle this with
  stemming, synonyms or spell correction.
- Exact-word matching is also the ceiling: this engine would give
  zero to a query about "football" even though the corpus has cricket,
  and zero to "artificial intelligence" even though it has "neural
  networks". That gap — meaning vs spelling — is exactly what word
  embeddings and modern search are built to close.
- The matrix is 9.1% dense on 12 documents; on a real corpus it would
  be well under 1% — which is why production TF-IDF is stored as
  sparse matrices, not dense NumPy arrays.
