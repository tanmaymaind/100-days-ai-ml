"""Day 12 — TF-IDF document search from scratch (NumPy only).

A tiny search engine: given a small corpus of documents and a query, rank
the documents by cosine similarity of their TF-IDF vectors. No sklearn —
the TF-IDF maths is built by hand so you can see every step.

Corpus: 12 short documents written for this project, on four topics
(machine learning, space, cooking, cricket) so a good ranking is
checkable by eye.
"""

import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- data

DOCS = [
    ("ml-1", "Neural networks learn by adjusting weights with gradient descent "
             "on training data until the loss stops falling."),
    ("ml-2", "A decision tree splits data on the feature that reduces impurity "
             "the most, and keeps splitting until each leaf is pure."),
    ("ml-3", "Overfitting happens when a model memorises the training set "
             "instead of learning patterns that generalise to new data."),
    ("ml-4", "Gradient descent updates model weights a small step at a time "
             "in the direction that lowers the loss function."),
    ("space-1", "The James Webb telescope looks at infrared light from the "
                "first galaxies that formed after the Big Bang."),
    ("space-2", "A black hole is a region of space where gravity is so strong "
                "that not even light can escape from it."),
    ("space-3", "Mars rovers drive across the red planet collecting rock "
                "samples and searching for signs of ancient water."),
    ("space-4", "Astronauts on the space station orbit Earth sixteen times "
                "a day and run experiments in weightlessness."),
    ("food-1", "To bake a chocolate cake, whisk flour, cocoa, sugar and eggs, "
               "then bake in a hot oven until a skewer comes out clean."),
    ("food-2", "A good biryani layers fragrant basmati rice with spiced "
               "chicken and is sealed so the steam cooks it slowly."),
    ("sport-1", "The cricket world cup final went to a super over after both "
               "teams finished tied on runs after fifty overs each."),
    ("sport-2", "A fast bowler runs in and delivers the ball at over 140 "
               "kilometres per hour, aiming to hit the stumps."),
]

QUERIES = [
    ("neural network training", "ml-1"),   # expected top document
    ("black hole gravity", "space-2"),
    ("chocolate cake recipe", "food-1"),
    ("cricket world cup", "sport-1"),
]

STOPWORDS = set(
    "a an the and or of on in to is are was were it its this that with by "
    "for from as at after each until so".split()
)

# ---------------------------------------------------------------- helpers


def tokenize(text):
    """Lowercase, keep words only, drop stopwords."""
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def build_vocab(tokenised_docs):
    vocab = sorted({w for doc in tokenised_docs for w in doc})
    return {w: i for i, w in enumerate(vocab)}, vocab


def tfidf_matrix(tokenised_docs, vocab_index):
    """Return the L2-normalised TF-IDF matrix (docs x terms).

    tf  = count(term, doc) / total terms in doc
    idf = log((1 + N) / (1 + df)) + 1   (smoothed, sklearn-style)
    """
    n_docs = len(tokenised_docs)
    n_terms = len(vocab_index)
    tf = np.zeros((n_docs, n_terms))
    df = np.zeros(n_terms)
    for d, doc in enumerate(tokenised_docs):
        for w in doc:
            tf[d, vocab_index[w]] += 1
        for w in set(doc):
            df[vocab_index[w]] += 1
        tf[d] /= len(doc)
    idf = np.log((1 + n_docs) / (1 + df)) + 1
    matrix = tf * idf
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms), idf, vocab_index


def search(query, matrix, idf, vocab_index):
    """Cosine similarity of the query vector against every document."""
    q = np.zeros(len(vocab_index))
    for w in tokenize(query):
        if w in vocab_index:
            q[vocab_index[w]] += 1
    if q.sum() == 0:
        return np.zeros(matrix.shape[0])
    q = q * idf
    q = q / np.linalg.norm(q)
    return matrix @ q


# ---------------------------------------------------------------- run

def main():
    tokenised = [tokenize(text) for _, text in DOCS]
    vocab_index, vocab = build_vocab(tokenised)
    matrix, idf, vocab_index = tfidf_matrix(tokenised, vocab_index)
    print(f"Documents: {len(DOCS)}   Vocabulary: {len(vocab)} terms")
    print(f"TF-IDF matrix shape: {matrix.shape}")
    print(f"Non-zero entries: {int(np.count_nonzero(matrix))} "
          f"of {matrix.size} ({100 * np.count_nonzero(matrix) / matrix.size:.1f}% dense)")

    # A rare word should carry a bigger IDF than a common one.
    for word in ("neural", "data", "the"):
        if word in vocab_index:
            print(f"  idf('{word}') = {idf[vocab_index[word]]:.3f}")
    print(f"  idf('data') appears in {sum(1 for d in tokenised if 'data' in d)} docs; "
          f"idf('neural') appears in {sum(1 for d in tokenised if 'neural' in d)} doc")

    correct_at_1 = 0
    all_scores = {}
    for query, expected in QUERIES:
        scores = search(query, matrix, idf, vocab_index)
        all_scores[query] = scores
        order = np.argsort(-scores)
        top = DOCS[order[0]][0]
        correct_at_1 += top == expected
        print(f"\nQuery: '{query}'  (expected top: {expected})")
        for rank in range(3):
            d = order[rank]
            print(f"  #{rank + 1}  {DOCS[d][0]:8s}  similarity {scores[d]:.4f}")
        print(f"  top-1 correct: {'yes' if top == expected else 'NO'}")

    print(f"\nPrecision@1 over {len(QUERIES)} queries: "
          f"{correct_at_1}/{len(QUERIES)} = {correct_at_1 / len(QUERIES):.2f}")

    # Plot: similarity of every document to the first query.
    query = QUERIES[0][0]
    scores = all_scores[query]
    labels = [doc_id for doc_id, _ in DOCS]
    colours = ["#2e7d32" if s == scores.max() else "#90a4ae" for s in scores]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(labels, scores, color=colours)
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"TF-IDF search — query: '{query}'")
    ax.set_ylim(0, max(scores) * 1.15)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("search_scores.svg", format="svg")
    print(f"\nSaved search_scores.svg (top doc '{labels[int(np.argmax(scores))]}' "
          f"scores {scores.max():.4f})")


if __name__ == "__main__":
    main()
