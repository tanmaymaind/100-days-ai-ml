# Day 7 — Sobel Edge Detection from Scratch

## The idea

Before neural networks, computer vision found edges with calculus: an
edge is just a place where the brightness changes fast. The **Sobel
operator** measures that change in both directions — how fast brightness
changes left-to-right (Gx) and top-to-bottom (Gy) — by convolving the
image with two small 3×3 kernels. Where the combined gradient is large,
there is an edge.

## The maths in brief

- **2D convolution:** slide a 3×3 kernel over the image; each output
  pixel is the weighted sum of its 3×3 neighbourhood (kernel flipped —
  convolution, not cross-correlation). Zero padding keeps the output the
  same size as the input.
- **Sobel kernels:**

  ```
  Gx = [-1  0  1]      Gy = [-1 -2 -1]
       [-2  0  2]           [ 0  0  0]
       [-1  0  1]           [ 1  2  1]
  ```

  Gx responds to **vertical** edges (brightness changing horizontally);
  Gy responds to **horizontal** edges. The centre row/column gets
  weight 2, which smooths noise a little while differentiating.
- **Gradient magnitude:** `|G| = √(Gx² + Gy²)` — large wherever an edge
  runs in any direction. The strongest possible response for a 0→1 step
  is 4.0 (the kernel weights sum to 4 on one side), so dividing by 4
  normalises it to [0, 1].
- **Gradient direction:** `θ = atan2(Gy, Gx)` — the direction of fastest
  brightness change, perpendicular to the edge itself.
- **Thresholding:** pixels with normalised magnitude above 0.25 are
  marked as edges, everything else is not. That single threshold turns a
  smooth gradient map into a binary edge map.

## The data

Synthetic 128×128 grayscale image, fixed seed (42): a bright square, a
bright circle and a dark bar on a gentle left-to-right gradient, plus a
little Gaussian noise (σ = 0.02) so the edges are realistic instead of
perfectly clean. No OpenCV, no scipy — NumPy and a hand-written
convolution only.

## How to run

```bash
python3 sobel_edge_detection.py
```

NumPy + matplotlib only. Produces `original_vs_edges.svg` and
`gradient_magnitude.svg`.

## Results (measured, seed 42)

| Metric | Value |
|---|---|
| Image size | 128×128 = 16,384 pixels |
| Max \|Gx\| (vertical edges) | 3.665 |
| Max \|Gy\| (horizontal edges) | 3.614 |
| Max gradient magnitude | 3.813 (normalised 0.953) |
| Mean gradient magnitude | 0.2110 |
| Threshold (normalised magnitude) | 0.25 |
| Edge pixels | 655 / 16,384 = 4.00% |
| Edge pixels on a true shape border | 655 / 655 = 100.0% |
| Mean gradient orientation, square's left edge | 0.8° (expect ~0°) |
| Mean gradient orientation, square's top edge | 89.0° (expect ~90°) |

Every single detected edge pixel lies within 2 px of a real shape
border — the threshold cleanly separates true edges from the noise
floor (the noise-only gradient is ~14× smaller than the threshold).
And the gradient direction behaves exactly as the theory says: ~0°
(horizontal change) on a vertical edge, ~90° (vertical change) on a
horizontal edge.

## What I learned

- Edge detection is differentiation, not magic: the Sobel kernels are
  just a smoothed finite difference in each direction.
- The 2 in the middle of the kernel matters — it is a tiny bit of
  averaging that keeps single noisy pixels from looking like edges.
- A threshold is a trade-off, not a truth: 0.25 catches every true
  border here and no noise, but on a noisier real photo the same value
  would either miss faint edges or keep speckles. That trade-off is
  exactly why later methods (like Canny) use two thresholds and link
  weak edges to strong ones.
- The gradient direction is as useful as its size: it tells you which
  way the edge runs, which is what makes non-maximum suppression and
  edge linking possible.
