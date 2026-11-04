# Day-3: Convolutional Neural Networks (CNN)

Notes on what a CNN is, where it came from, how it is built, and why we use it for images instead of a plain ANN. These are expanded from the keywords in [learn.txt](learn.txt).

## Contents

1. [Where CNN fits in deep learning](#1-where-cnn-fits-in-deep-learning)
2. [What is a CNN?](#2-what-is-a-cnn)
3. [ANN vs CNN: matrix multiplication vs convolution](#3-ann-vs-cnn-matrix-multiplication-vs-convolution)
4. [History: LeNet-5 (1998)](#4-history-lenet-5-1998)
5. [CNN architecture](#5-cnn-architecture)
6. [The maths behind each layer](#6-the-maths-behind-each-layer)
7. [Why not use an ANN for images?](#7-why-not-use-an-ann-for-images)
8. [The overfitting problem](#8-the-overfitting-problem)
9. [Famous CNN architectures and Keras Applications](#9-famous-cnn-architectures-and-keras-applications)
10. [Key terms](#10-key-terms)
11. [References](#11-references)

---

## 1. Where CNN fits in deep learning

Deep learning has three main network families. Each one suits a different shape of data.

```mermaid
flowchart TD
    DL[Deep Learning] --> ANN["ANN<br/>Artificial Neural Network"]
    DL --> CNN["CNN<br/>Convolutional Neural Network"]
    DL --> RNN["RNN<br/>Recurrent Neural Network"]
    ANN --> A1["Tabular data<br/>(rows and columns)"]
    CNN --> C1["Grid data<br/>(images, video frames)"]
    RNN --> R1["Sequence data<br/>(text, audio, time series)"]
```

| Network | Best for | Core operation |
|---------|----------|----------------|
| ANN | Tabular data, simple features | Matrix multiplication (every input connects to every neuron) |
| CNN | Images, video, anything laid out on a grid | Convolution (a small filter slides over the input) |
| RNN | Sequences where order matters | Recurrence (output of the last step feeds the next step) |

The main job of a CNN is **feature extraction**: it learns by itself which edges, textures, and shapes matter, so we don't have to hand-design them.

---

## 2. What is a CNN?

A **Convolutional Neural Network** is a neural network designed for data with a **grid-like topology**.

An image is a grid of pixels:

- A grayscale image is a 2D grid: `height × width`.
- A colour image is a 3D grid: `height × width × 3` (one layer each for B, G, R in OpenCV, or R, G, B elsewhere).

```
Grayscale 5×5 image (each number = pixel brightness 0–255)

  ┌─────┬─────┬─────┬─────┬─────┐
  │  12 │  40 │  45 │  38 │  10 │
  ├─────┼─────┼─────┼─────┼─────┤
  │  35 │ 200 │ 210 │ 190 │  30 │
  ├─────┼─────┼─────┼─────┼─────┤
  │  40 │ 220 │ 255 │ 205 │  42 │
  ├─────┼─────┼─────┼─────┼─────┤
  │  33 │ 195 │ 215 │ 188 │  29 │
  ├─────┼─────┼─────┼─────┼─────┤
  │  11 │  38 │  44 │  36 │   9 │
  └─────┴─────┴─────┴─────┴─────┘
        a bright blob in the middle
```

In a grid, a pixel's **neighbours** carry meaning. A pixel on its own says almost nothing; a group of neighbouring pixels forms an edge, a corner, an eye, a wheel. A CNN is built to look at small neighbourhoods and keep that spatial structure.

CNNs learn features in a hierarchy:

```mermaid
flowchart LR
    P[Raw pixels] --> L1["Layer 1<br/>edges, lines"]
    L1 --> L2["Layer 2<br/>corners, textures"]
    L2 --> L3["Layer 3<br/>parts: eye, ear, wheel"]
    L3 --> L4["Deep layers<br/>whole objects: face, car, zebra"]
```

---

## 3. ANN vs CNN: matrix multiplication vs convolution

**ANN → matrix multiplication.** The image is flattened into one long vector, and every pixel connects to every neuron in the next layer (a *fully connected* or *dense* layer):

```
output = activation(W · x + b)
```

**CNN → convolution.** A small matrix of weights called a **kernel** (or **filter**), for example 3×3, slides across the image. At each position it multiplies the overlapping pixels by the kernel values and adds them up. The same kernel is reused at every position.

```mermaid
flowchart LR
    subgraph ANN["ANN (fully connected)"]
        direction LR
        x1((x1)) --> h1((h1))
        x1 --> h2((h2))
        x2((x2)) --> h1
        x2 --> h2
        x3((x3)) --> h1
        x3 --> h2
    end
    subgraph CNN["CNN (local + shared weights)"]
        direction LR
        p1((p1)) --> o1((o1))
        p2((p2)) --> o1
        p2 --> o2((o2))
        p3((p3)) --> o1
        p3 --> o2
        p4((p4)) --> o2
    end
```

| | ANN | CNN |
|---|-----|-----|
| Operation | Matrix multiplication | Convolution |
| Connections | Every input to every neuron | Each output looks at a small local patch |
| Weights | A separate weight for every connection | The same kernel is shared across the whole image |
| Input shape | Must be flattened to 1D | Keeps the 2D/3D grid |
| Spatial info | Lost when flattening | Kept |
| Parameters for images | Huge | Small |

Two ideas make CNNs efficient:

- **Local connectivity**: each neuron only sees a small patch (its *receptive field*).
- **Weight sharing**: one 3×3 kernel has 9 weights, whether the image is 28×28 or 4000×3000. An edge detector that works in the top-left corner also works in the bottom-right.

---

## 4. History: LeNet-5 (1998)

- **Who**: **Yann LeCun**, with Léon Bottou, Yoshua Bengio, and Patrick Haffner.
- **Where**: **AT&T Bell Labs**.
- **When**: published in **1998** in the paper *"Gradient-Based Learning Applied to Document Recognition"*. LeCun's earlier convolutional networks date back to 1989.
- **What for**: **document recognition**: reading handwritten **digits** (ZIP codes, bank cheques) and other characters such as **letters**.
- **Network**: **LeNet-5**, one of the first CNNs used in real products. It was used to read cheques in US banks. The "5" is usually explained as its 5 main learnable layers (C1, C3, C5, F6, output); counting the pooling layers it has 7.

### LeNet-5 architecture

```mermaid
flowchart LR
    I["Input<br/>32×32×1"] --> C1["C1: Conv 5×5<br/>6 maps 28×28"]
    C1 --> S2["S2: Pool 2×2<br/>6 maps 14×14"]
    S2 --> C3["C3: Conv 5×5<br/>16 maps 10×10"]
    C3 --> S4["S4: Pool 2×2<br/>16 maps 5×5"]
    S4 --> C5["C5: Conv 5×5<br/>120 units"]
    C5 --> F6["F6: Fully connected<br/>84 units"]
    F6 --> O["Output<br/>10 classes (0–9)"]
```

| Layer | Type | Output size |
|-------|------|-------------|
| Input | Grayscale image | 32 × 32 × 1 |
| C1 | Convolution, 6 filters of 5×5 | 28 × 28 × 6 |
| S2 | Subsampling (average pooling) 2×2 | 14 × 14 × 6 |
| C3 | Convolution, 16 filters of 5×5 | 10 × 10 × 16 |
| S4 | Subsampling 2×2 | 5 × 5 × 16 |
| C5 | Convolution, 120 filters of 5×5 | 1 × 1 × 120 |
| F6 | Fully connected | 84 |
| Output | 10 classes | 10 |

LeNet-5 has about **60,000 parameters**. Modern networks follow the same basic pattern of convolution → pooling → fully connected, just much deeper.

---

## 5. CNN architecture

![CNN architecture: input image, convolution + ReLU and pooling blocks for feature extraction, then a flatten layer, fully connected layers, and a softmax output](Screenshot%202026-11-05%20011034.png)

*From the lecture: an image of a zebra goes through Convolution + ReLU and Pooling blocks (feature extraction), then a Flatten layer and Fully Connected layers (classification), and finally Softmax gives probabilities for Horse, Zebra, and Dog.*

A CNN has three building blocks:

1. **Convolution layer** (+ ReLU)
2. **Pooling layer**
3. **Fully connected (FC) layer**

```mermaid
flowchart LR
    IN["Input image<br/>e.g. 224×224×3"] --> CV1["Conv + ReLU"]
    CV1 --> PL1[Pooling]
    PL1 --> CV2["Conv + ReLU"]
    CV2 --> PL2[Pooling]
    PL2 --> CV3["Conv + ReLU"]
    CV3 --> PL3[Pooling]
    PL3 --> FL[Flatten]
    FL --> FC["Fully connected<br/>layers"]
    FC --> SM[Softmax]
    SM --> OUT["Horse 0.2<br/>Zebra 0.7<br/>Dog 0.1"]

    subgraph FE["Feature extraction"]
        CV1
        PL1
        CV2
        PL2
        CV3
        PL3
    end
    subgraph CL["Classification"]
        FL
        FC
    end
    subgraph PD["Probabilistic distribution"]
        SM
    end
```

| Stage | Layers | What it does |
|-------|--------|--------------|
| **Feature extraction** | Convolution + ReLU, Pooling (repeated) | Finds edges, textures, shapes, and object parts. Produces **feature maps**. |
| **Classification** | Flatten, Fully connected | Turns the feature maps into a single vector and decides which class the features match. |
| **Probabilistic distribution** | Softmax | Converts raw scores into probabilities that add up to 1. |

### 5.1 Convolution layer

- A **kernel** slides over the image and produces a **feature map**.
- Each kernel learns to detect one pattern (a vertical edge, a curve, a colour blob).
- A layer usually has many kernels (32, 64, 128…), so it outputs many feature maps stacked together.
- **ReLU** is applied after each convolution to add non-linearity.

### 5.2 Pooling layer

- Shrinks each feature map, usually by half.
- **Max pooling** keeps the largest value in each window; **average pooling** keeps the mean.
- Fewer numbers means less computation and fewer parameters later on.
- Gives some **translation invariance**: if the object moves by a pixel or two, the pooled output barely changes.
- Has **no learnable weights**.

### 5.3 Flatten layer

- Unrolls the final stack of feature maps into one long 1D vector so it can feed a dense layer.
- Example: `7 × 7 × 64` → vector of `3136` numbers.

### 5.4 Fully connected (FC) layer

- A normal ANN layer: every input connects to every neuron.
- Combines the extracted features to make the final decision.

### 5.5 Softmax output

- The last layer has one neuron per class.
- Softmax turns its raw scores into probabilities, for example `Horse 0.2, Zebra 0.7, Dog 0.1`, and the class with the highest probability is the prediction.

---

## 6. The maths behind each layer

### 6.1 Convolution (worked example)

A 5×5 input and a 3×3 **vertical edge** kernel:

```
Input (5×5)              Kernel (3×3)
┌───┬───┬───┬───┬───┐    ┌────┬───┬────┐
│ 1 │ 2 │ 0 │ 1 │ 3 │    │  1 │ 0 │ -1 │
├───┼───┼───┼───┼───┤    ├────┼───┼────┤
│ 0 │ 1 │ 3 │ 1 │ 0 │    │  1 │ 0 │ -1 │
├───┼───┼───┼───┼───┤    ├────┼───┼────┤
│ 2 │ 1 │ 0 │ 2 │ 1 │    │  1 │ 0 │ -1 │
├───┼───┼───┼───┼───┤    └────┴───┴────┘
│ 1 │ 0 │ 1 │ 3 │ 2 │
├───┼───┼───┼───┼───┤
│ 0 │ 2 │ 1 │ 0 │ 1 │
└───┴───┴───┴───┴───┘
```

Place the kernel on the top-left 3×3 patch, multiply element by element, and add:

```
(1×1) + (2×0) + (0×-1)
+ (0×1) + (1×0) + (3×-1)
+ (2×1) + (1×0) + (0×-1)
= 1 + 0 + 0 + 0 + 0 - 3 + 2 + 0 + 0 = 0
```

Slide one step right and repeat, then move down a row. The full output (feature map) is:

```
Output (3×3)
┌────┬────┬────┐
│  0 │  0 │ -1 │
├────┼────┼────┤
│ -1 │ -4 │  1 │
├────┼────┼────┤
│  1 │ -2 │ -2 │
└────┴────┴────┘
```

Strictly speaking, deep learning libraries compute *cross-correlation* (they don't flip the kernel), but everyone calls it convolution.

Try it yourself with NumPy:

```python
import numpy as np

image = np.array([[1, 2, 0, 1, 3],
                  [0, 1, 3, 1, 0],
                  [2, 1, 0, 2, 1],
                  [1, 0, 1, 3, 2],
                  [0, 2, 1, 0, 1]])
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

out = np.zeros((3, 3), dtype=int)
for i in range(3):
    for j in range(3):
        out[i, j] = np.sum(image[i:i+3, j:j+3] * kernel)
print(out)
```

Or on a real image with OpenCV (this connects back to Day-2):

```python
import cv2
import numpy as np

img = cv2.imread("../Day-2/data/image1.jpg", cv2.IMREAD_GRAYSCALE)
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]], dtype=np.float32)
edges = cv2.filter2D(img, -1, kernel)
```

In a CNN, the kernel values are not hand-picked like this. They start random and are **learned** during training.

### 6.2 Output size formula

```
Output size = ⌊(n + 2p − f) / s⌋ + 1
```

| Symbol | Meaning |
|--------|---------|
| `n` | Input size (height or width) |
| `f` | Filter (kernel) size |
| `p` | Padding: extra border of zeros added around the input |
| `s` | Stride: how many pixels the kernel jumps each step |

Examples:

- `n = 5, f = 3, p = 0, s = 1` → `(5 + 0 − 3)/1 + 1 = 3` → 3×3 (the example above)
- `n = 32, f = 5, p = 0, s = 1` → `28` (LeNet-5 layer C1)
- `n = 224, f = 3, p = 1, s = 1` → `224` ("same" padding keeps the size)
- `n = 224, f = 3, p = 1, s = 2` → `112` (stride 2 halves it)

```
Padding p = 1 (zeros around a 3×3 input)   Stride s = 2

  0 0 0 0 0                                  [k k k] . .      . . [k k k]
  0 a b c 0                                  [k k k] . .  →   . . [k k k]
  0 d e f 0                                  [k k k] . .      . . [k k k]
  0 g h i 0                                  kernel jumps 2 pixels each step
  0 0 0 0 0
```

### 6.3 Number of parameters in a convolution layer

```
Parameters = (f × f × C_in + 1) × C_out
```

`C_in` is the number of input channels, `C_out` is the number of filters, and `+1` is the bias for each filter.

Example: 32 filters of 3×3 on an RGB image → `(3 × 3 × 3 + 1) × 32 = 896` parameters. This does not depend on the image size at all.

### 6.4 ReLU activation

```
ReLU(x) = max(0, x)
```

Negative values become 0, positive values stay the same. Applied to the feature map above:

```
[ 0  0 -1]          [0 0 0]
[-1 -4  1]  ReLU →  [0 0 1]
[ 1 -2 -2]          [1 0 0]
```

Without a non-linear function like ReLU, stacking layers would collapse into one big linear function and the network couldn't learn complex shapes.

### 6.5 Max pooling (2×2, stride 2)

```
Input (4×4)                     Output (2×2)
┌───┬───┬───┬───┐
│ 1 │ 3 │ 2 │ 1 │               ┌───┬───┐
├───┼───┼───┼───┤               │ 6 │ 5 │
│ 4 │ 6 │ 5 │ 0 │   max pool    ├───┼───┤
├───┼───┼───┼───┤   ───────→    │ 3 │ 7 │
│ 3 │ 1 │ 1 │ 2 │               └───┴───┘
├───┼───┼───┼───┤
│ 0 │ 2 │ 7 │ 4 │
└───┴───┴───┴───┘
```

Each 2×2 block becomes its largest value: `max(1,3,4,6) = 6`, `max(2,1,5,0) = 5`, `max(3,1,0,2) = 3`, `max(1,2,7,4) = 7`.

### 6.6 Softmax

```
softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)
```

Example: raw scores `[2.0, 1.0, 0.1]` for `[Zebra, Horse, Dog]` → probabilities `[0.66, 0.24, 0.10]`. They add up to 1, and Zebra wins.

---

## 7. Why not use an ANN for images?

```mermaid
flowchart TD
    Q["Feed an image straight into an ANN?"] --> P1["1. High computation cost"]
    Q --> P2["2. Overfitting"]
    Q --> P3["3. Loss of important spatial features"]
    P1 --> S["Solution: CNN<br/>local connections + shared weights + pooling"]
    P2 --> S
    P3 --> S
```

### 7.1 High computation cost

An ANN needs one weight for every pixel-to-neuron connection.

| Image | Input values | Weights for one dense layer of 1,000 neurons |
|-------|--------------|----------------------------------------------|
| 28 × 28 grayscale (MNIST) | 784 | 784,000 |
| 224 × 224 × 3 colour | 150,528 | **150,528,000** (about 150 million) |
| 1920 × 1080 × 3 (Full HD) | 6,220,800 | about 6.2 billion |

Compare: a CNN layer with 64 filters of 3×3 on an RGB image needs only `(3×3×3 + 1) × 64 = 1,792` parameters, whatever the image size.

More weights means more memory, slower training, and slower predictions.

### 7.2 Overfitting

With millions of weights and a limited number of training images, an ANN can simply **memorise** the training images (including their noise and backgrounds) instead of learning general patterns. It then does well on training data and badly on new images. See [section 8](#8-the-overfitting-problem).

### 7.3 Loss of important features (spatial information)

To feed an image into an ANN you must **flatten** it into a 1D vector:

```
2D image                       Flattened vector
┌───┬───┬───┐
│ a │ b │ c │
├───┼───┼───┤        →        [ a  b  c  d  e  f  g  h  i ]
│ d │ e │ f │
├───┼───┼───┤        "b" and "e" were neighbours (one above the other),
│ g │ h │ i │        now they are 3 positions apart. The network no
└───┴───┴───┘        longer knows which pixels were next to each other.
```

- Pixels that were next to each other vertically end up far apart.
- The network has no built-in idea of "neighbour", "edge", or "shape".
- If the object shifts by a few pixels, completely different input neurons light up, so the ANN has to learn the same object at every position separately.

A CNN keeps the grid and looks at neighbourhoods, so it keeps these features.

---

## 8. The overfitting problem

**Overfitting** happens when a model learns the training data too well, including noise and small details that don't generalise. It scores high on training data but low on new, unseen data.

```
Accuracy
  ▲
  │            ________________  training accuracy (keeps rising)
  │          /
  │        /   _____
  │      /   /      \_______    validation accuracy (rises, then falls)
  │    /   /                \___
  │  /   /
  │/   /        ↑
  │  /     overfitting starts here
  └──────────────────────────────────▶ Epochs
```

| | Underfitting | Good fit | Overfitting |
|---|--------------|----------|-------------|
| Training accuracy | Low | High | Very high |
| Test accuracy | Low | High | Low |
| Cause | Model too simple | Right balance | Model too complex / too little data |

### Signs

- Training loss keeps going down but validation loss starts going up.
- Big gap between training accuracy and test accuracy.

### How to reduce overfitting

| Technique | Idea |
|-----------|------|
| **Use a CNN instead of an ANN** | Weight sharing means far fewer parameters to memorise with. |
| **More data** | More examples make memorising harder. |
| **Data augmentation** | Create new training images by flipping, rotating, cropping, zooming, and changing brightness. |
| **Dropout** | Randomly switch off some neurons during training so the network can't rely on any single one. |
| **Pooling** | Reduces the size of feature maps and the number of parameters. |
| **L1 / L2 regularisation** | Penalise large weights. |
| **Batch normalisation** | Stabilises training and has a slight regularising effect. |
| **Early stopping** | Stop training when validation loss stops improving. |
| **Transfer learning** | Start from a network already trained on millions of images (see section 9). |

---

## 9. Famous CNN architectures and Keras Applications

```mermaid
flowchart LR
    A["1998<br/>LeNet-5<br/>handwritten digits"] --> B["2012<br/>AlexNet<br/>won ImageNet"]
    B --> C["2014<br/>VGG16 / VGG19<br/>stacks of 3×3 convs"]
    B --> D["2014<br/>Inception (GoogLeNet)<br/>parallel filter sizes"]
    C --> E["2015<br/>ResNet-50<br/>skip connections"]
    D --> E
```

| Model | Year | Depth | Parameters (approx.) | Key idea |
|-------|------|-------|----------------------|----------|
| **LeNet-5** | 1998 | 5 learnable layers | 60 K | First practical CNN, digit recognition |
| **AlexNet** | 2012 | 8 layers | 60 M | ReLU, dropout, trained on GPUs; won ImageNet 2012 |
| **VGG16** | 2014 | 16 layers | 138 M | Only 3×3 convolutions, stacked deep |
| **VGG19** | 2014 | 19 layers | 144 M | Same as VGG16, 3 more conv layers |
| **Inception v1 (GoogLeNet)** | 2014 | 22 layers | 6.8 M | Inception module: 1×1, 3×3, 5×5 filters in parallel |
| **ResNet-50** | 2015 | 50 layers | 25.6 M | Residual (skip) connections: `output = F(x) + x` |

ResNet's skip connection, the idea that made 50+ layer networks trainable:

```mermaid
flowchart LR
    X[x] --> L1[Conv + ReLU]
    L1 --> L2[Conv]
    L2 --> ADD(("+"))
    X -- "skip connection" --> ADD
    ADD --> R[ReLU] --> Y["F(x) + x"]
```

### Keras Applications

[Keras Applications](https://keras.io/api/applications/) gives you these models ready to use, with weights already trained on ImageNet (1.4 million images, 1,000 classes). This is called **transfer learning**.

```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

model = ResNet50(weights="imagenet")

img = image.load_img("zebra.jpg", target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print(decode_predictions(preds, top=3)[0])
# e.g. [('n02391049', 'zebra', 0.98), ...]
```

Other models available the same way: `VGG16`, `VGG19`, `InceptionV3`, `MobileNetV2`, `EfficientNetB0`, and more. AlexNet and LeNet-5 are not in Keras Applications, but they are easy to build by hand.

A small LeNet-5-style CNN in Keras:

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(32, 32, 1)),
    layers.Conv2D(6, kernel_size=5, activation="relu"),   # 28×28×6
    layers.AveragePooling2D(pool_size=2),                 # 14×14×6
    layers.Conv2D(16, kernel_size=5, activation="relu"),  # 10×10×16
    layers.AveragePooling2D(pool_size=2),                 # 5×5×16
    layers.Flatten(),                                     # 400
    layers.Dense(120, activation="relu"),
    layers.Dense(84, activation="relu"),
    layers.Dense(10, activation="softmax"),               # digits 0–9
])
model.summary()
```

---

## 10. Key terms

| Term | Meaning |
|------|---------|
| **Kernel / Filter** | Small matrix of learnable weights that slides over the image |
| **Feature map** | The output of one kernel applied across the whole image |
| **Receptive field** | The patch of the input that one output value "sees" |
| **Stride** | How many pixels the kernel moves each step |
| **Padding** | Zeros added around the border so the output keeps its size |
| **Channel** | A depth layer of the input, such as R, G, B, or one feature map |
| **Weight sharing** | The same kernel weights are used at every position |
| **Pooling** | Downsampling a feature map (max or average) |
| **Flatten** | Turning a 3D stack of feature maps into a 1D vector |
| **Fully connected (dense)** | Every input connected to every neuron |
| **ReLU** | `max(0, x)` activation |
| **Softmax** | Turns scores into class probabilities that sum to 1 |
| **Overfitting** | Memorising training data instead of learning general patterns |
| **Transfer learning** | Reusing a model pre-trained on a big dataset for a new task |

---

## 11. References

**Papers**

- LeCun, Bottou, Bengio, Haffner (1998). [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf). The LeNet-5 paper; Figure 2 is the original LeNet-5 diagram.
- Krizhevsky, Sutskever, Hinton (2012). [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) (AlexNet).
- Simonyan, Zisserman (2014). [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) (VGG16/19).
- Szegedy et al. (2014). [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842) (Inception / GoogLeNet).
- He, Zhang, Ren, Sun (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (ResNet).

**Diagrams and visual explanations**

- [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/). Has an animated demo of a kernel sliding over a 3-channel input.
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/). Interactive, in-browser visualisation of every layer of a small CNN.
- [Convolution arithmetic animations (vdumoulin/conv_arithmetic)](https://github.com/vdumoulin/conv_arithmetic). GIFs of padding and stride.
- [Wikipedia: Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network). Includes the LeNet vs AlexNet comparison diagram.
- [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/). The handwritten digit dataset LeNet was tested on.

**Docs**

- [Keras Applications](https://keras.io/api/applications/). Pre-trained models and their sizes and accuracies.
- [Keras Conv2D layer](https://keras.io/api/layers/convolution_layers/convolution2d/).
- [OpenCV `filter2D`](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html). Apply your own kernel to an image.
