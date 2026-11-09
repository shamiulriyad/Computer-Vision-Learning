"""Generate the figures used in Day-3/README.md.

Run from the Day-3 folder:
    python make_figures.py

Needs only numpy and matplotlib (Pillow comes with matplotlib). Images are read from ../Day-2/data
and the PNGs are written to ./images.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageEnhance

HERE = Path(__file__).parent
DATA = HERE.parent / "Day-2" / "data"
OUT = HERE / "images"
OUT.mkdir(exist_ok=True)

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)   # vertical edges
SOBEL_Y = SOBEL_X.T                                                     # horizontal edges


def load_gray(name, width):
    img = Image.open(DATA / name).convert("L")
    img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    return np.asarray(img, dtype=float)


def conv(img, kernel):
    # Slide the kernel, multiply and sum (cross-correlation, which is what
    # deep-learning "convolution" really computes). Edge pixels are repeated
    # as padding so the output keeps the input size.
    k = kernel.shape[0] // 2
    padded = np.pad(img, k, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)
    return np.einsum("ijkl,kl->ij", windows, kernel)


def save(fig, name):
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", OUT / name)


# 1. Sobel edge detection ------------------------------------------------------
def sobel_figure():
    g = load_gray("p2.jpg", 640)
    gx, gy = conv(g, SOBEL_X), conv(g, SOBEL_Y)
    mag = np.hypot(gx, gy)

    panels = [
        (g, "Grayscale input"),
        (np.abs(gx), "Sobel X: vertical edges"),
        (np.abs(gy), "Sobel Y: horizontal edges"),
        (mag, "Combined: sqrt(Gx² + Gy²)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    for ax, (im, title) in zip(axes.flat, panels):
        ax.imshow(im, cmap="gray", vmax=np.percentile(im, 99.5))
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle("Sobel filter on a parking garage: pillars light up in X, ceiling lines in Y",
                 fontsize=13)
    fig.tight_layout()
    save(fig, "sobel_edges.png")


# 2. Different kernels -> different feature maps --------------------------------
def kernels_figure():
    g = load_gray("images.jpg", 480)
    kernels = {
        "Identity": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        "Box blur": np.ones((3, 3)) / 9,
        "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        "Sobel X (vertical edge)": SOBEL_X,
        "Sobel Y (horizontal edge)": SOBEL_Y,
        "Laplacian (all edges)": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
    for ax, (name, k) in zip(axes.flat, kernels.items()):
        out = conv(g, np.asarray(k, dtype=float))
        if name in ("Identity", "Box blur", "Sharpen"):
            ax.imshow(np.clip(out, 0, 255), cmap="gray", vmin=0, vmax=255)
        else:
            out = np.abs(out)
            ax.imshow(out, cmap="gray", vmax=np.percentile(out, 99.5))
        rows = "\n".join(" ".join(f"{v:>4.2g}" for v in row) for row in np.asarray(k, float))
        ax.set_title(name, fontsize=12)
        ax.text(0.02, 0.03, rows, transform=ax.transAxes, family="monospace", fontsize=8,
                color="black", bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        ax.axis("off")
    fig.suptitle("Same image, different 3×3 kernels → different feature maps", fontsize=13)
    fig.tight_layout()
    save(fig, "kernels.png")


# 3. Padding and stride ---------------------------------------------------------
def draw_grid(ax, n, origin=(0, 0), fill=None, pad=0, title=""):
    ox, oy = origin
    total = n + 2 * pad
    for i in range(total):
        for j in range(total):
            is_pad = i < pad or j < pad or i >= n + pad or j >= n + pad
            face = "#e8e8e8" if is_pad else "white"
            if fill and (i, j) in fill:
                face = fill[(i, j)]
            ax.add_patch(Rectangle((ox + j, oy + total - 1 - i), 1, 1,
                                   facecolor=face, edgecolor="#555", lw=0.8))
            if is_pad:
                ax.text(ox + j + 0.5, oy + total - 1 - i + 0.5, "0", ha="center",
                        va="center", fontsize=8, color="#888")
    ax.text(ox + total / 2, oy + total + 0.4, title, ha="center", fontsize=10)


def padding_stride_figure():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    blue, orange = "#9ecae1", "#fdae6b"

    # (a) no padding, stride 1
    ax = axes[0]
    win = {(i, j): blue for i in range(3) for j in range(3)}
    draw_grid(ax, 5, fill=win, title="Input 5×5, kernel 3×3")
    draw_grid(ax, 3, origin=(6.5, 1), fill={(0, 0): blue}, title="Output 3×3")
    ax.set_title("(a) p = 0, s = 1  →  (5 − 3)/1 + 1 = 3", fontsize=11)

    # (b) padding 1, stride 1 ("same")
    ax = axes[1]
    win = {(i, j): blue for i in range(3) for j in range(3)}
    draw_grid(ax, 5, pad=1, fill=win, title="Input 5×5 + zero padding")
    draw_grid(ax, 5, origin=(8, 1), fill={(0, 0): blue}, title="Output 5×5")
    ax.set_title("(b) p = 1, s = 1  →  (5 + 2 − 3)/1 + 1 = 5", fontsize=11)

    # (c) stride 2
    ax = axes[2]
    win = {(i, j): blue for i in range(3) for j in range(3)}
    win.update({(i, j): orange for i in range(3) for j in range(2, 5)})
    win.update({(i, 2): "#c7a0c9" for i in range(3)})  # overlap column
    draw_grid(ax, 5, fill=win, title="Kernel jumps 2 pixels")
    draw_grid(ax, 2, origin=(6.5, 1.5), fill={(0, 0): blue, (0, 1): orange},
              title="Output 2×2")
    ax.set_title("(c) p = 0, s = 2  →  ⌊(5 − 3)/2⌋ + 1 = 2", fontsize=11)

    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
        ax.autoscale_view()
        ax.set_xlim(-0.5, 13.5)
        ax.set_ylim(-0.5, 8.2)
    fig.suptitle("Padding keeps the size, stride shrinks it", fontsize=13)
    fig.tight_layout()
    save(fig, "padding_stride.png")


# 4. Conv -> ReLU -> Max pool on a real image -----------------------------------
def max_pool(x, k=2):
    h, w = (x.shape[0] // k) * k, (x.shape[1] // k) * k
    return x[:h, :w].reshape(h // k, k, w // k, k).max(axis=(1, 3))


def conv_relu_pool_figure():
    g = load_gray("p2.jpg", 256)
    c = conv(g, SOBEL_X)
    r = np.maximum(c, 0)
    p1 = max_pool(r)
    p2 = max_pool(p1)
    panels = [
        (g, "gray", f"Input\n{g.shape[1]}×{g.shape[0]}"),
        (c, "RdBu_r", f"Convolution (Sobel X)\n{c.shape[1]}×{c.shape[0]}, has negatives"),
        (r, "gray", f"ReLU = max(0, x)\nnegatives → 0"),
        (p1, "gray", f"Max pool 2×2\n{p1.shape[1]}×{p1.shape[0]}"),
        (p2, "gray", f"Max pool again\n{p2.shape[1]}×{p2.shape[0]}"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6),
                             gridspec_kw={"width_ratios": [4, 4, 4, 2, 1]})
    for ax, (im, cmap, title) in zip(axes, panels):
        if cmap == "RdBu_r":
            v = np.percentile(np.abs(im), 99)
            ax.imshow(im, cmap=cmap, vmin=-v, vmax=v)
        else:
            ax.imshow(im, cmap=cmap, vmax=np.percentile(im, 99.5))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("One CNN block by hand: Convolution → ReLU → Pooling (the map gets smaller, "
                 "the edges survive)", fontsize=12)
    fig.tight_layout()
    save(fig, "conv_relu_pool.png")


# 5. Data augmentation ----------------------------------------------------------
def augmentation_figure():
    img = Image.open(DATA / "images.jpg").convert("RGB")
    img = img.resize((360, 240), Image.LANCZOS)
    w, h = img.size
    zoom = img.crop((int(w * 0.18), int(h * 0.18), int(w * 0.82), int(h * 0.82))).resize((w, h))
    shifted = Image.new("RGB", (w, h))
    shifted.paste(img, (50, 25))
    panels = [
        (img, "Original"),
        (img.transpose(Image.FLIP_LEFT_RIGHT), "Horizontal flip"),
        (img.rotate(20, resample=Image.BILINEAR), "Rotate 20°"),
        (zoom, "Zoom / crop"),
        (ImageEnhance.Brightness(img).enhance(1.6), "Brighter"),
        (ImageEnhance.Brightness(img).enhance(0.5), "Darker"),
        (ImageEnhance.Contrast(img).enhance(1.8), "More contrast"),
        (shifted, "Shift (translate)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, (im, title) in zip(axes.flat, panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("Data augmentation: 1 photo → 8 training images, all still labelled \"car\"",
                 fontsize=13)
    fig.tight_layout()
    save(fig, "data_augmentation.png")


if __name__ == "__main__":
    sobel_figure()
    kernels_figure()
    padding_stride_figure()
    conv_relu_pool_figure()
    augmentation_figure()
