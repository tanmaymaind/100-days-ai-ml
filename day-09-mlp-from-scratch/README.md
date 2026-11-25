# Day 9 — Multi-Layer Perceptron (MLP) from Scratch

A tiny feed-forward neural network, built with NumPy only — no sklearn,
no PyTorch. One hidden layer, trained with full-batch gradient descent
and backpropagation, on the two-moons dataset: two interleaving
half-circles that no straight line can separate.

## The idea

Day 5's logistic regression is a single sigmoid unit — it can only draw
a straight line. An MLP adds a **hidden layer** between input and
output, and that changes everything: each hidden unit learns its own
curved feature, and the output unit combines them into a curved boundary.

```
x1 ──┐
     ├──► [ tanh ] ──┐
x2 ──┘      × 8      ├──► [ sigmoid ] ──► P(class 1)
                    ─┘
```

- Hidden activation: **tanh** — squashes to (−1, 1), centred at 0,
  which keeps gradients healthier than a sigmoid hidden layer.
- Output activation: **sigmoid** — turns the score into a probability.
- Loss: **binary cross-entropy**, the same as logistic regression.

## The maths — backpropagation in one paragraph

Forward pass computes `z1 = X·W1 + b1`, `a1 = tanh(z1)`,
`z2 = a1·W2 + b2`, `ŷ = sigmoid(z2)`. The loss gradient at the output
is beautifully simple — the sigmoid and the cross-entropy cancel:

```
∂L/∂z2 = (ŷ − y) / n
```

Then the chain rule walks the error backwards: multiply by `W2ᵀ` to
reach the hidden layer, and by `tanh′(z1) = 1 − tanh²(z1)` to pass
through the activation. Every weight moves a small step (`lr = 0.5`)
down its gradient. That is all backprop is: the chain rule, applied
layer by layer, reusing the error from the layer above.

Weights start with Xavier-style init (`σ = √(1/fan-in)`) so the first
activations are neither all-zero nor saturated.

## The data

Synthetic, fixed seed (42): 400 points on two interleaving moons
(noise 0.15), 80/20 train/test split (320 / 80). The 0.15 noise means
the moons overlap slightly — even a perfect model cannot reach 100%.

## How to run

```bash
python3 mlp.py
```

Requires NumPy and matplotlib only. Prints the loss every 200 epochs,
the accuracies, and a hidden-size sweep; saves
`decision_boundary.svg` and `loss_curve.svg`.

## Measured results

2-8-1 network, 2000 epochs, lr 0.5:

| Metric | Value |
|---|---|
| Final loss (BCE) | **0.1931** (from 0.6147 at epoch 0) |
| Train accuracy | **0.9062** |
| Test accuracy | **0.9000** |

The loss falls smoothly from 0.61 to 0.19 (see `loss_curve.svg`) and
the learned boundary wraps around both moons (see
`decision_boundary.svg`) — something Day 5's logistic regression
could never do.

### Hidden-size sweep — how wide is wide enough?

| Hidden units | Train acc | Test acc |
|---|---:|---:|
| 1 | 0.8531 | 0.8625 |
| 2 | 0.8531 | 0.8625 |
| 4 | 0.9062 | **0.9000** |
| 8 | 0.9062 | **0.9000** |
| 16 | 0.9094 | 0.8875 |
| 32 | 0.9187 | 0.8875 |

1–2 hidden units are too few to bend the boundary around both moons.
4–8 units are the sweet spot. Beyond that, train accuracy keeps
creeping up while test accuracy **falls** — the extra capacity just
memorises the noise. Same overfitting lesson as Day 8's depth sweep,
now in width instead of depth.

## What I learned

- A hidden layer is what makes a network *deep* in the meaningful
  sense: it lets the model learn its own features instead of using the
  raw inputs directly.
- Backprop is not magic — it is the chain rule plus one lucky
  cancellation (`ŷ − y`) at the output. Writing it by hand once makes
  every framework's `loss.backward()` feel obvious.
- The loss function had a real bug on the first run: a `(320,)` label
  vector broadcast against a `(320, 1)` prediction, silently computing
  a 320×320 mean. Loss *rose* while accuracy improved — a nonsense
  combination that was the clue. Flattening both arrays before the
  mean fixed it, and the loss curve immediately behaved.
- Width is a regulariser too: 4 hidden units beat 32 on unseen data.
