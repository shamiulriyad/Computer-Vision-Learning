<div align="center">

# Computer Vision Learning

**Learning computer vision one day at a time: from reading images with OpenCV to the maths inside a CNN.**

<a href="https://shamiulriyad.github.io/Computer-Vision-Learning/"><img src="https://img.shields.io/badge/Live_Website-Visit-4357e6?style=for-the-badge&logo=githubpages&logoColor=white" alt="Live website"></a>
<a href="https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html"><img src="https://img.shields.io/badge/Convolution_Lab-Try_it_live-ffb534?style=for-the-badge&logo=html5&logoColor=black" alt="Convolution Lab: try it live"></a>
<a href="Day-5/README.md"><img src="https://img.shields.io/badge/New-Day_5_Padding_%26_Strides-0891b2?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Day-5 notes: padding and strides"></a>

<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.11">
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV">
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter">
<img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white" alt="Keras">
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square" alt="MIT licence"></a>

</div>

---

## 🔗 Quick links

| | Where | What you get |
|---|---|---|
| 🌐 | **[Website](https://shamiulriyad.github.io/Computer-Vision-Learning/)** | Homepage with every day of the journey |
| ▶️ | **[Convolution Lab (live)](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html)** | Interactive page: watch a filter slide, then practise by hand |
| 🆕 | **[Day-5 notes](Day-5/README.md)** | Padding and strides: theory, formulas, Keras experiment |
| 📓 | **[Day-5 notebook](Day-5/padding_and_strides_demo.ipynb)** | Same vs valid vs stride 2 on MNIST |
| 📘 | **[Day-4 notes](Day-4/README.md)** | Convolution maths: worked example, formulas, sources |
| 📗 | **[Day-3 notes](Day-3/README.md)** | CNN theory with diagrams and worked examples |
| 📓 | **[Day-2 notebooks](Day-2/)** | 8 OpenCV notebooks |

## 📚 What you will learn (48 topics)

Every topic is marked ✅ and linked to the exact place it is taught. Click a topic to start.

<details open>
<summary><b>Day 2 · OpenCV fundamentals: 10 topics</b></summary>

| | Topic | You will learn |
|:---:|---|---|
| ✅ | [**Reading images**](Day-2/reading%20image.ipynb) | cv2.imread, image shape, showing photos inline |
| ✅ | [**Reading videos**](Day-2/reading%20video.ipynb) | cv2.VideoCapture frame by frame, quit with q |
| ✅ | [**Grayscale conversion**](Day-2/basic%20function.ipynb) | cv2.cvtColor from BGR to gray |
| ✅ | [**Gaussian blur**](Day-2/basic%20function.ipynb) | smoothing noise with cv2.GaussianBlur |
| ✅ | [**Canny edge detection**](Day-2/basic%20function.ipynb) | finding edges with cv2.Canny |
| ✅ | [**Cropping**](Day-2/croping%20image.ipynb) | cutting regions with NumPy slicing |
| ✅ | [**Resizing**](Day-2/resize%20image.ipynb) | changing image size with cv2.resize |
| ✅ | [**Drawing shapes and text**](Day-2/shapes%20text.ipynb) | line, rectangle, circle, putText |
| ✅ | [**Warp perspective**](Day-2/wrap_prespective.ipynb) | a top-down view from four points |
| ✅ | [**Joining images**](Day-2/joining%20image.ipynb) | np.hstack and np.vstack |
| ⏳ | **Reading the webcam** | live video from your camera. Coming soon |

</details>

<details open>
<summary><b>Day 3 · CNN theory: 14 topics</b></summary>

| | Topic | You will learn |
|:---:|---|---|
| ✅ | [**Where CNN fits**](Day-3/README.md#1-where-cnn-fits-in-deep-learning) | ANN vs CNN vs RNN, and which data each suits |
| ✅ | [**What is a CNN**](Day-3/README.md#2-what-is-a-cnn) | networks for grid-shaped data like images |
| ✅ | [**ANN vs CNN**](Day-3/README.md#3-ann-vs-cnn-matrix-multiplication-vs-convolution) | matrix multiplication vs convolution |
| ✅ | [**History: LeNet-5**](Day-3/README.md#4-history-lenet-5-1998) | Yann LeCun's 1998 digit recogniser |
| ✅ | [**CNN architecture**](Day-3/README.md#5-cnn-architecture) | conv, pooling, flatten, fully connected, softmax |
| ✅ | [**Filters and edge detection**](Day-3/README.md#6-filters-kernels-and-edge-detection) | kernels, Sobel, hand-made vs learned |
| ✅ | [**Maths of each layer**](Day-3/README.md#7-the-maths-behind-each-layer) | output size, parameters, ReLU, max pooling |
| ✅ | [**Backpropagation in a CNN**](Day-3/README.md#8-backpropagation-in-a-cnn) | how gradients update the kernels |
| ✅ | [**Why not an ANN for images**](Day-3/README.md#9-why-not-use-an-ann-for-images) | cost, overfitting, lost spatial structure |
| ✅ | [**Overfitting**](Day-3/README.md#10-the-overfitting-problem) | the signs, and how to reduce it |
| ✅ | [**Data augmentation**](Day-3/README.md#11-data-augmentation) | flips, rotations, zooms, brightness |
| ✅ | [**Famous architectures**](Day-3/README.md#12-famous-cnn-architectures-and-keras-applications) | AlexNet, VGG, Inception, ResNet, Keras Applications |
| ✅ | [**Transfer learning**](Day-3/README.md#13-transfer-learning) | freeze an ImageNet model, then fine-tune |
| ✅ | [**Image classification project**](Day-3/README.md#14-project-image-classification) | from folders of photos to a trained model |

</details>

<details open>
<summary><b>Day 4 · Convolution maths (interactive): 14 topics</b></summary>

| | Topic | You will learn |
|:---:|---|---|
| ✅ | [**Input and kernel matrices**](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html#setup) | images as numbers, filters as weights |
| ✅ | [**The convolution equation**](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html#setup) | O[i][j] = ΣΣ I[i·S+m][j·S+n] · K[m][n] |
| ✅ | [**Step-by-step calculation**](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html#worked) | all 9 positions, nothing skipped |
| ✅ | [**Sliding-window animation**](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html#animate) | Next, Previous, Play, Pause, Reset |
| ✅ | [**Cross-correlation vs convolution**](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html#concept-a) | why CNNs don't flip the kernel |
| ✅ | [**Element-wise multiplication**](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html#concept-b) | ⊙ vs matrix multiplication |
| ✅ | [**Summation and bias**](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html#concept-c) | nine products into one number, as a dot product |
| ✅ | [**Stride**](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html#concept-d) | S = 1 vs S = 2, and uneven strides |
| ✅ | [**Padding**](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html#concept-e) | valid, same and full padding |
| ✅ | [**Output size formula**](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html#concept-f) | ⌊(N − F + 2P) / S⌋ + 1, with a calculator |
| ✅ | [**Multiple filters**](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html#concept-g) | K filters give K feature maps |
| ✅ | [**Multi-channel input**](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html#concept-g) | RGB filters and parameter counting |
| ✅ | [**Hand-calculation practice**](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html#practice) | 8 exercises with hidden solutions |
| ✅ | [**Revision and self-test**](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html#notes) | formulas, common mistakes, 5 questions |

</details>

<details open>
<summary><b>Day 5 · Padding and strides: 10 topics</b></summary>

| | Topic | You will learn |
|:---:|---|---|
| ✅ | [**Why padding**](Day-5/README.md#1-why-padding) | the image shrinks and border pixels get lost |
| ✅ | [**How padding works**](Day-5/README.md#2-how-padding-works) | zero padding, padded size n + 2p |
| ✅ | [**Valid vs same padding**](Day-5/README.md#3-valid-vs-same-padding) | Keras `padding='valid'` and `'same'` |
| ✅ | [**How much padding**](Day-5/README.md#4-how-much-padding) | p = (f − 1) / 2, why it is 1 for a 3×3 filter |
| ✅ | [**What is stride**](Day-5/README.md#5-what-is-stride) | s = 1 vs s = 2, how far the filter jumps |
| ✅ | [**Why use a stride**](Day-5/README.md#6-why-use-a-stride) | faster, fewer parameters, higher-level features |
| ✅ | [**Output size formula**](Day-5/README.md#7-the-output-size-formula) | ⌊(n + 2p − f) / s⌋ + 1 with a worked example |
| ✅ | [**Padding and strides in Keras**](Day-5/README.md#8-padding-and-strides-in-keras) | `Conv2D(..., padding, strides)` |
| ✅ | [**Same vs valid vs stride 2**](Day-5/README.md#9-the-experiment-same-vs-valid-vs-stride-2) | 3 MNIST models: 3.23 M vs 2.00 M vs 0.82 M params |
| ✅ | [**Revision and common mistakes**](Day-5/README.md#10-revision) | formula sheet and the errors to avoid |

</details>

## 🗺️ Learning journey

| Day | Topic | Status | Open |
|:---:|---|:---:|---|
| **05** | Padding and strides in a CNN | ✅ New | [Notes](Day-5/README.md) · [Notebook](Day-5/padding_and_strides_demo.ipynb) · [Quick notes](Day-5/learning.txt) |
| **04** | The convolution operation (maths first) | ✅ Done | [▶️ Live lab](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html) · [Notes](Day-4/README.md) · [Code](Day-4/) |
| **03** | Convolutional Neural Networks (theory) | ✅ Done | [Notes](Day-3/README.md) · [Notebook](Day-3/filters_demo.ipynb) |
| **02** | OpenCV fundamentals | ✅ Done | [Notebooks](Day-2/) |

---

## Day 5: Padding and strides

📗 **[Read the full notes](Day-5/README.md)** · 📓 **[padding_and_strides_demo.ipynb](Day-5/padding_and_strides_demo.ipynb)** · 📝 **[Quick notes](Day-5/learning.txt)**

How padding and stride decide the size of every feature map, and what that does to the size of the model.

- **Padding**: why (shrinking images, lost borders), zero padding, `valid` vs `same`, `p = (f − 1) / 2`
- **Stride**: what it is, and why `s > 1` means faster training and higher-level features
- **Formula**: `⌊(n + 2p − f) / s⌋ + 1`, with a worked example (28 → 14)
- **Experiment**: the same Keras CNN on MNIST three times:

| Model | Conv output | Total params |
|---|---|---:|
| `same`, stride 1 | 28 → 28 → 28 | 3,231,498 |
| `valid`, stride 1 | 26 → 24 → 22 | 2,002,698 |
| `same`, stride 2 | 14 → 14 → 14 | 823,050 |

Conv layers always have 320 + 9,248 + 9,248 parameters; the difference is all in the Dense layer after `Flatten`. One stride of 2 cuts the model to about a quarter.

## Day 4: The convolution operation

> **[▶️ Open the Convolution Lab](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html)**. It runs in the browser; nothing to install.

How a CNN filter turns an input matrix into a feature map, with every multiplication written out.

- **Full calculation**: a 5×5 input and a 3×3 kernel, all 9 positions: region ⊙ kernel → sum → output cell
- **Animated filter**: Next, Previous, Play, Pause and Reset, with switchable kernel, stride and padding
- **Concepts A–G**: cross-correlation vs convolution, stride, padding, `⌊(N − F + 2P) / S⌋ + 1`, multiple filters
- **Practice**: 8 hand-calculation exercises with hidden step-by-step solutions
- **Verified**: [`verify_convolution.py`](Day-4/verify_convolution.py) checks every answer with NumPy

## Day 3: Convolutional Neural Networks

📗 **[Read the full notes](Day-3/README.md)** · 📓 **[filters_demo.ipynb](Day-3/filters_demo.ipynb)**

<details>
<summary><b>What's covered</b></summary>

- **Where CNN fits**: ANN for tabular data, CNN for grids like images, RNN for sequences
- **History**: LeNet-5 (Yann LeCun, 1998) for handwritten digit recognition
- **Architecture**: Convolution + ReLU → Pooling → Flatten → Fully connected → Softmax
- **Filters**: kernels, Sobel edge detection, hand-made vs learned filters, shown on real photos
- **Maths**: convolution, output size, parameter count, ReLU, max pooling, softmax
- **Backpropagation**: how gradients flow back through each layer to update the kernels
- **Why not an ANN for images**: computation cost, overfitting, lost spatial structure
- **Data augmentation**: flips, rotations, zooms and brightness changes against overfitting
- **Famous models**: LeNet-5, AlexNet, VGG, Inception, ResNet-50 with Keras Applications
- **Transfer learning**: freeze an ImageNet model, train a new head, then fine-tune

The notebook compares RGB and BGR, splits colour channels, and runs a Sobel filter written from scratch, checked against `cv2.filter2D`. Figures are generated by [`make_figures.py`](Day-3/make_figures.py).

</details>

## Day 2: OpenCV fundamentals

| Topic | Notebook | What it shows |
|---|---|---|
| Reading images | [📓 reading image](Day-2/reading%20image.ipynb) | `cv2.imread`, image shape, fitting large photos on screen |
| Reading videos | [📓 reading video](Day-2/reading%20video.ipynb) | `cv2.VideoCapture` frame by frame, press `q` to quit |
| Basic functions | [📓 basic function](Day-2/basic%20function.ipynb) | Grayscale, `GaussianBlur`, `Canny` edges |
| Cropping | [📓 croping image](Day-2/croping%20image.ipynb) | NumPy slicing `image[y1:y2, x1:x2]` |
| Resizing | [📓 resize image](Day-2/resize%20image.ipynb) | `cv2.resize` |
| Shapes and text | [📓 shapes text](Day-2/shapes%20text.ipynb) | `line`, `rectangle`, `circle`, `putText` |
| Warp perspective | [📓 wrap_prespective](Day-2/wrap_prespective.ipynb) | `getPerspectiveTransform` + `warpPerspective` |
| Joining images | [📓 joining image](Day-2/joining%20image.ipynb) | `np.hstack` / `np.vstack` |
| Reading webcam | ⏳ To do | |

---

## 🚀 Getting started

```bash
git clone https://github.com/shamiulriyad/Computer-Vision-Learning.git
cd Computer-Vision-Learning
pip install -r requirements.txt     # opencv-python, matplotlib
pip install tensorflow              # only for the Day-5 notebook
```

Open the notebooks with Jupyter or VS Code (Jupyter extension), **from inside their day folder** (for example `Day-2/`) so the `data/...` image paths resolve. The Convolution Lab needs nothing: open the [live link](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html), or open `Day-4/convolution_lab.html` locally.

## 📁 Project structure

```
Computer-Vision-Learning/
├── index.html          website homepage (GitHub Pages)
├── Day-2/              OpenCV notebooks + data/ sample images
├── Day-3/              CNN notes, figures, filters notebook
├── Day-4/              Convolution Lab (HTML/JS), notes, NumPy verifier
└── Day-5/              Padding & strides notes + Keras notebook
```

<details>
<summary><b>💡 Tips and known issues</b></summary>

- **Video files aren't in the repo.** `.mp4` files are git-ignored (GitHub's 100 MB limit). Put your own video at `Day-2/data/video.mp4`.
- **`cv2.imshow` in VS Code** opens a pop-up window that can hide behind the editor. Press a key in it to continue, or display images inline with matplotlib.
- **Colour order.** OpenCV uses BGR, matplotlib expects RGB: `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.
- **Joining images.** `np.hstack` needs equal heights, `np.vstack` equal widths, and both need the same number of channels.

</details>

## 📄 License

[MIT](LICENSE) © shamiulriyad
