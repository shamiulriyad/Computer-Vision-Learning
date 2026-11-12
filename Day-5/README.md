# Day-5: Padding and Strides in a CNN

Two settings that control the **size of every feature map** in a CNN: **padding** (add a border so the image doesn't shrink) and **stride** (how far the filter jumps each step). The theory is below; the notebook then builds three real Keras models on MNIST to show how each choice changes the output shape and the number of parameters.

## ▶ [Open the Padding & Stride Lab](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-5/padding_stride_lab.html)

An interactive page: watch the filter slide over a padded input, compare strides, use the output-size calculator, rebuild the notebook's three models in a live network planner, and practise with checked answers. GitHub shows `.html` as source code, so use the link above (or open [padding_stride_lab.html](padding_stride_lab.html) locally in a browser).

> 📓 **Notebook:** [padding_and_strides_demo.ipynb](padding_and_strides_demo.ipynb) · 📝 **Quick notes:** [learning.txt](learning.txt) · ⬅️ **Previous day:** [Day-4 Convolution Lab](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html)

| Symbol | Meaning |
|:---:|---|
| `n` | input size (height or width) |
| `f` | filter (kernel) size |
| `p` | padding: pixels added on **each** side |
| `s` | stride: pixels the filter moves per step |

---

## 1. Why padding?

Without padding, a convolution has two problems:

1. **The image shrinks after every layer.** A 3×3 filter on a 28×28 image gives 26×26. Three layers give 28 → 26 → 24 → 22. A deep network would shrink the image to almost nothing.
2. **Border pixels are used less.** A corner pixel is covered by the filter only once; a centre pixel is covered up to 9 times (3×3 filter). Information at the edges of the image is under-used.

Padding fixes both: the output keeps its size, and edge pixels take part in as many positions as the others.

## 2. How padding works

Add `p` rows and columns of zeros on **every** side of the input, then slide the filter as usual.

```
Original 3×3          Zero padding, p = 1 → 5×5
                       0 0 0 0 0
  1 2 3                0 1 2 3 0
  4 5 6       →        0 4 5 6 0
  7 8 9                0 7 8 9 0
                       0 0 0 0 0
```

The padded size is `n + 2p` (the `2` is because padding goes on both sides). With stride 1:

```
Output = n + 2p − f + 1
```

## 3. Valid vs same padding

| Keras | Padding | Output size (stride 1) | 28×28 input, 3×3 filter |
|---|---|---|---|
| `padding='valid'` | none, `p = 0` | `n − f + 1` (shrinks) | 26×26 |
| `padding='same'` | enough to keep the size | `n` (unchanged) | 28×28 |

## 4. How much padding?

For `same` padding with an odd filter and stride 1:

```
p = (f − 1) / 2
```

| Filter `f` | Padding `p` |
|:---:|:---:|
| 3×3 | 1 |
| 5×5 | 2 |
| 7×7 | 3 |

You often hear "padding is 1". That is only because 3×3 is the most common filter. For other filter sizes the padding is larger.

## 5. What is stride?

Stride is how many pixels the filter moves each step.

- `s = 1`: move one pixel at a time. Every position is visited (the default).
- `s = 2`: jump two pixels. Every other position is skipped, so the output is about half the size in each direction.

## 6. Why use a stride?

A stride larger than 1 **downsamples** the feature map, which gives:

1. **Faster computation.** Fewer filter positions means fewer multiplications.
2. **Less training time and memory.** Smaller feature maps, and far fewer weights in the Dense layer after `Flatten` (see the experiment below).
3. **Higher-level features.** After downsampling, each pixel stands for a bigger area of the original image, so the next layers learn larger patterns (shapes, parts of objects) instead of small details like edges.

**Trade-off:** fine detail between the visited positions is skipped. A large stride very early in the network can throw away useful information.

## 7. The output size formula

The general formula covers both padding and stride:

```
Output = ⌊(n + 2p − f) / s⌋ + 1
```

- `⌊ ⌋` means **floor**: round down.
- Divide the **whole** `(n + 2p − f)` by `s`, then add 1.
- With `s = 1` it becomes `n + 2p − f + 1` (the padding formula above).
- With `p = 0` it becomes `⌊(n − f) / s⌋ + 1`.

Worked example, `n = 28`, `f = 3`, `p = 1`, `s = 2`:

```
⌊(28 + 2 − 3) / 2⌋ + 1 = ⌊27 / 2⌋ + 1 = ⌊13.5⌋ + 1 = 13 + 1 = 14
```

This is the 14×14 the notebook shows for `padding='same', strides=(2,2)`.

## 8. Padding and strides in Keras

```python
from keras.layers import Conv2D

Conv2D(32, kernel_size=(3, 3), padding='same',  activation='relu')                  # keeps the size
Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu')                  # shrinks by f − 1
Conv2D(32, kernel_size=(3, 3), padding='same',  strides=(2, 2), activation='relu')  # halves the size
```

## 9. The experiment: same vs valid vs stride 2

The [notebook](padding_and_strides_demo.ipynb) loads MNIST (60,000 images of 28×28) and builds the same model three times:
`Conv2D(32, 3×3)` ×3 → `Flatten` → `Dense(128)` → `Dense(10, softmax)`. Only padding and stride change.

| Model | Conv output sizes | Flatten size | Dense(128) params | **Total params** |
|---|---|---:|---:|---:|
| `same`, stride 1 | 28 → 28 → 28 | 25,088 | 3,211,392 | **3,231,498** |
| `valid`, stride 1 | 26 → 24 → 22 | 15,488 | 1,982,592 | **2,002,698** |
| `same`, stride 2 (first layer) | 14 → 14 → 14 | 6,272 | 802,944 | **823,050** |

What this shows:

- **Conv layer parameters never change** (320, 9,248, 9,248). A filter's weights depend on its size, the input channels and the number of filters, not on the image size:
  ```
  Conv2D params = (f × f × input_channels + 1) × filters
  first layer:  (3 × 3 × 1  + 1) × 32 = 320
  next layers:  (3 × 3 × 32 + 1) × 32 = 9,248
  ```
- **All the difference is in the Dense layer.** Flatten size = height × width × 32, and each of those numbers connects to 128 neurons: `25,088 × 128 + 128 = 3,211,392`.
- **One stride of 2 cut the parameters by about 4×** (3.23 M → 0.82 M), because halving both height and width leaves a quarter of the pixels.
- `same` padding keeps more information but costs more parameters; stride is the cheap way to shrink the map.

### Running the notebook

It needs TensorFlow (not in the root `requirements.txt`):

```bash
pip install tensorflow
```

Or open it in [Google Colab](https://colab.research.google.com/), where TensorFlow is already installed. MNIST downloads automatically the first time.

## 10. Revision

| | Formula |
|---|---|
| Padded size | `n + 2p` |
| Output (stride 1) | `n + 2p − f + 1` |
| Output (general) | `⌊(n + 2p − f) / s⌋ + 1` |
| Same padding | `p = (f − 1) / 2` |
| Conv2D parameters | `(f × f × C_in + 1) × filters` |

**Common mistakes**

- Writing `(n − f / s) + 1`: divide the whole `(n + 2p − f)` by `s`, not only `f`.
- Forgetting the `2` in `2p`: padding goes on both sides.
- Rounding up instead of down.
- Thinking padding is always 1: it is `(f − 1) / 2`, so 1 only for a 3×3 filter.
- Thinking conv parameters depend on image size: they don't. Dense parameters do.

## Sources

- [Dive into Deep Learning §7.3: Padding and Stride](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html)
- [Stanford CS231n: Convolutional Networks](https://cs231n.github.io/convolutional-networks/)
- V. Dumoulin & F. Visin, [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285) and its [animations](https://github.com/vdumoulin/conv_arithmetic)
- [Keras `Conv2D` documentation](https://keras.io/api/layers/convolution_layers/convolution2d/)
