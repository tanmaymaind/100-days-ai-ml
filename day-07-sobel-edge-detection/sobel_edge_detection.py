"""
Day 7 — Sobel Edge Detection from Scratch (Computer Vision basics)

Build a synthetic test image (a circle, a square, and a smooth gradient),
then find its edges the way classic computer vision does: convolve the
image with the Sobel kernels to get the horizontal and vertical gradients,
combine them into a gradient magnitude, and threshold it into a binary
edge map. No OpenCV, no scipy — just NumPy and a hand-written 2D
convolution.

Run:  python3 sobel_edge_detection.py
Needs: numpy, matplotlib
Produces: original_vs_edges.svg, gradient_magnitude.svg
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42

# --- Sobel kernels -------------------------------------------------------
# Gx responds to vertical edges (intensity changing left -> right);
# Gy responds to horizontal edges (intensity changing top -> bottom).
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=float)
SOBEL_Y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=float)


def convolve2d(image, kernel):
    """Hand-written 2D convolution with zero padding (no scipy)."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)))
    out = np.zeros_like(image, dtype=float)
    # Flip the kernel: convolution, not cross-correlation.
    k = kernel[::-1, ::-1]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = np.sum(padded[i:i + kh, j:j + kw] * k)
    return out


def make_test_image(size=128, seed=SEED):
    """Synthetic 128x128 grayscale image: square + circle + gradient.

    Pixel values in [0, 1]. A little Gaussian noise (seeded) makes the
    edges realistic instead of perfectly clean.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    image = 0.15 * xx / size  # gentle left-to-right gradient background
    # Bright square, top-left.
    image[20:60, 20:60] = 0.9
    # Bright circle, bottom-right.
    image[(xx - 92) ** 2 + (yy - 92) ** 2 < 22 ** 2] = 0.9
    # Dark bar across the middle.
    image[64:72, 10:118] = 0.05
    image += rng.normal(0, 0.02, size=(size, size))
    return np.clip(image, 0, 1)


def main():
    image = make_test_image()

    gx = convolve2d(image, SOBEL_X)
    gy = convolve2d(image, SOBEL_Y)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.degrees(np.arctan2(gy, gx))

    # The strongest possible Sobel response for a 0->1 step is 4.0
    # (kernel weights sum to 4 on one side), so normalise by that.
    magnitude_norm = magnitude / 4.0

    threshold = 0.25
    edges = (magnitude_norm > threshold).astype(float)

    # --- measured results ------------------------------------------------
    edge_pixels = int(edges.sum())
    total_pixels = image.size
    print(f"Image size: {image.shape[0]}x{image.shape[1]} = {total_pixels} pixels")
    print(f"Max |Gx| (vertical edges): {np.abs(gx).max():.3f}")
    print(f"Max |Gy| (horizontal edges): {np.abs(gy).max():.3f}")
    print(f"Max gradient magnitude: {magnitude.max():.3f} "
          f"(normalised {magnitude_norm.max():.3f})")
    print(f"Mean gradient magnitude: {magnitude.mean():.4f}")
    print(f"Threshold: {threshold} (normalised magnitude)")
    print(f"Edge pixels: {edge_pixels} / {total_pixels} "
          f"= {100 * edge_pixels / total_pixels:.2f}%")
    # Edge pixels are overwhelmingly at the true shape borders: check the
    # fraction of edge pixels that lie within 2 px of a known shape border.
    yy, xx = np.mgrid[0:128, 0:128]
    near_border = (
        ((np.abs(xx - 20) <= 2) | (np.abs(xx - 59) <= 2)) & (yy >= 18) & (yy <= 61)
        | ((np.abs(yy - 20) <= 2) | (np.abs(yy - 59) <= 2)) & (xx >= 18) & (xx <= 61)
        | (np.abs(np.sqrt((xx - 92) ** 2 + (yy - 92) ** 2) - 22) <= 2)
        | ((np.abs(yy - 64) <= 2) | (np.abs(yy - 71) <= 2)) & (xx >= 8) & (xx <= 119)
    )
    on_border = int((edges.astype(bool) & near_border).sum())
    print(f"Edge pixels on a true shape border: {on_border} / {edge_pixels} "
          f"= {100 * on_border / edge_pixels:.1f}%")
    # Gradient direction is an orientation (mod 180 deg): on the square's
    # left edge (a vertical edge) it should be ~0 deg; on the bar's top
    # edge (a horizontal edge) it should be ~90 deg.
    def orientation_deg(a):
        return np.abs((a + 90) % 180 - 90)

    left_edge_dir = orientation_deg(direction[30:50, 20])
    top_edge_dir = orientation_deg(direction[20, 30:50])
    print(f"Mean gradient orientation on square's left edge: "
          f"{left_edge_dir.mean():.1f} deg (expect ~0)")
    print(f"Mean gradient orientation on square's top edge: "
          f"{top_edge_dir.mean():.1f} deg (expect ~90)")

    # --- plots -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original (synthetic)")
    axes[1].imshow(magnitude_norm, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Gradient magnitude")
    axes[2].imshow(edges, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Edges (threshold {threshold})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("original_vs_edges.svg", format="svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(magnitude_norm.ravel(), bins=60, range=(0, 1))
    ax.axvline(threshold, color="red", linestyle="--",
               label=f"threshold = {threshold}")
    ax.set_xlabel("Normalised gradient magnitude")
    ax.set_ylabel("Pixel count")
    ax.set_title("Most pixels are flat; edges are the long tail")
    ax.legend()
    fig.tight_layout()
    fig.savefig("gradient_magnitude.svg", format="svg")
    plt.close(fig)
    print("Saved original_vs_edges.svg and gradient_magnitude.svg")


if __name__ == "__main__":
    main()
