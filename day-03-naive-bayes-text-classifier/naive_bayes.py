"""
Day 3 — Naive Bayes Text Classifier from Scratch
=================================================
A multinomial Naive Bayes spam/ham classifier built with NumPy only.

The dataset is 40 short messages written for this project (20 spam, 20 ham),
split 30 train / 10 test with a fixed seed so the result is reproducible.
Words are lowercased, split on non-letters, and counted as a bag of words.
Laplace (add-one) smoothing keeps unseen words from zeroing a probability.

Run:  python naive_bayes.py
"""

import re
import numpy as np

# ---------------------------------------------------------------------------
# Data: 20 spam + 20 ham messages (short, phone-SMS style).
# ---------------------------------------------------------------------------
SPAM = [
    "Congratulations you have won a free prize claim now",
    "Win cash now click this link to claim your reward",
    "You are selected for a free gift card offer limited time",
    "Urgent your account will be blocked verify now",
    "Get rich quick earn money fast from home today",
    "Free lottery ticket claim your jackpot prize today",
    "You won a brand new iphone click here to collect",
    "Limited offer buy one get one free act now",
    "Your loan is approved get instant cash now",
    "Claim your cashback reward before it expires today",
    "Hot singles near you chat now for free",
    "Double your money in one week guaranteed profit",
    "Your package is waiting pay the delivery fee now",
    "Exclusive deal only today 90 percent off sale",
    "You have been chosen for a free holiday trip",
    "Earn 5000 rupees daily working from home apply now",
    "Your bank account needs verification click the link",
    "Win a free cruise holiday text WIN to claim",
    "Special discount on watches buy now before stock ends",
    "Congratulations your number won the lucky draw prize",
]

HAM = [
    "Can we meet for coffee tomorrow morning",
    "Please send me the report by friday evening",
    "Happy birthday hope you have a great day",
    "I will pick up milk and bread on my way home",
    "The meeting has been moved to three pm today",
    "Did you watch the match last night it was amazing",
    "Call me when you are free this evening",
    "Thanks for your help with the assignment today",
    "Let us plan the trip for next weekend together",
    "I have finished reading the book you suggested",
    "Please remind me to pay the electricity bill",
    "The train arrives at platform two in ten minutes",
    "Can you share the photos from yesterday party",
    "I am cooking dinner tonight come home early",
    "Good luck for your exam tomorrow study well",
    "The doctor appointment is confirmed for monday morning",
    "Please bring your notebook to class tomorrow",
    "We are going to the park after lunch today",
    "Happy anniversary to you both enjoy your day",
    "I will reach the station by seven in the evening",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TOKEN = re.compile(r"[a-z]+")


def tokenize(text):
    return TOKEN.findall(text.lower())


def build_vocab(texts):
    vocab = sorted({w for t in texts for w in tokenize(t)})
    return {w: i for i, w in enumerate(vocab)}


def vectorize(texts, word_to_idx):
    """Bag-of-words count matrix (n_docs x vocab_size)."""
    X = np.zeros((len(texts), len(word_to_idx)), dtype=np.float64)
    for r, text in enumerate(texts):
        for w in tokenize(text):
            if w in word_to_idx:
                X[r, word_to_idx[w]] += 1
    return X


def train(X, y, alpha=1.0):
    """Return log priors and log likelihoods P(word | class)."""
    classes = np.unique(y)
    n_docs, n_vocab = X.shape
    log_prior = np.zeros(len(classes))
    log_likelihood = np.zeros((len(classes), n_vocab))
    for c in classes:
        Xc = X[y == c]
        log_prior[c] = np.log(Xc.shape[0] / n_docs)
        counts = Xc.sum(axis=0) + alpha  # Laplace smoothing
        log_likelihood[c] = np.log(counts / counts.sum())
    return classes, log_prior, log_likelihood


def predict(X, classes, log_prior, log_likelihood):
    scores = X @ log_likelihood.T + log_prior
    return classes[np.argmax(scores, axis=1)]


def top_words(word_to_idx, log_likelihood, classes, n=8):
    """Most indicative words per class by log-odds vs the other class."""
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    out = {}
    for c in classes:
        other = 1 - c
        odds = log_likelihood[c] - log_likelihood[other]
        top = np.argsort(odds)[::-1][:n]
        out[c] = [(idx_to_word[i], round(float(odds[i]), 2)) for i in top]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(42)
    texts = SPAM + HAM
    labels = np.array([1] * len(SPAM) + [0] * len(HAM))  # 1 = spam, 0 = ham
    perm = rng.permutation(len(texts))
    train_idx, test_idx = perm[:30], perm[30:]

    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    word_to_idx = build_vocab(train_texts)
    X_train = vectorize(train_texts, word_to_idx)
    X_test = vectorize(test_texts, word_to_idx)

    classes, log_prior, log_likelihood = train(X_train, y_train, alpha=1.0)
    y_pred = predict(X_test, classes, log_prior, log_likelihood)

    acc = float((y_pred == y_test).mean())
    tp = int(((y_pred == 1) & (y_test == 1)).sum())
    tn = int(((y_pred == 0) & (y_test == 0)).sum())
    fp = int(((y_pred == 1) & (y_test == 0)).sum())
    fn = int(((y_pred == 0) & (y_test == 1)).sum())

    print(f"Vocabulary size : {len(word_to_idx)} words")
    print(f"Train / test    : {len(train_texts)} / {len(test_texts)} messages")
    print(f"Test accuracy   : {acc:.2f} ({int(acc * len(y_test))}/{len(y_test)} correct)")
    print(f"Confusion matrix: TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    words = top_words(word_to_idx, log_likelihood, classes, n=8)
    print("\nMost spam-indicative words (log-odds):")
    for w, s in words[1]:
        print(f"  {w:>14}  {s:+.2f}")
    print("Most ham-indicative words (log-odds):")
    for w, s in words[0]:
        print(f"  {w:>14}  {s:+.2f}")

    # Accuracy vs smoothing alpha, for the plot.
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    accs = []
    for a in alphas:
        _, lp, ll = train(X_train, y_train, alpha=a)
        accs.append(float((predict(X_test, classes, lp, ll) == y_test).mean()))
    print("\nAccuracy vs alpha:", list(zip(alphas, accs)))


if __name__ == "__main__":
    main()
