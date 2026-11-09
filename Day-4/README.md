# Day-4: The Convolution Operation in a CNN

A hands-on learning module for one idea: **how a convolution filter turns an input matrix into a feature map**. It is built around the mathematics: every multiplication and every sum is written out, then animated, then practised by hand.

> Understand the mathematics first, visualise the operation second, practise on your own third.

## ▶ [Open the live lab](https://shamiulriyad.github.io/Computer-Vision-Learning/Day-4/convolution_lab.html)

GitHub shows `.html` files as source code. The link above opens the working page, served by GitHub Pages.

## How to use it

Online: use the live link above. Offline: open **[convolution_lab.html](convolution_lab.html)** in a browser (double-click it, or right-click → *Open with* → your browser). You don't need a server or an internet connection; only the external diagrams in section 5 need internet.

| Section | What you do there |
|---|---|
| 1. The problem | See the 5×5 input, the 3×3 kernel, and predict the output size with the formula |
| 2. Full calculation | All 9 filter positions: region ⊙ kernel = products → sum → store → slide |
| 3. Filter movement | Animator with **Next / Previous / Play / Pause / Reset**. Change the input, kernel (or type your own), stride and padding |
| 4. Concepts A–G | Cross-correlation vs convolution, element-wise multiplication, summation, stride, padding, output size, multiple filters |
| 5. Diagrams & sources | Stanford CS230 and conv_arithmetic diagrams, each checked against the formula, with links |
| 6. Practice | 8 hand-calculation exercises (answers hidden until you reveal them) + output-size drills |
| 7. Learning notes | Summary, formulas, common mistakes, terminology, revision sheet, 5 self-test questions |

Check the maths from the command line (needs only NumPy):

```bash
cd Day-4
python verify_convolution.py           # 22 checks: worked example, concepts, all exercise answers
python verify_convolution.py --steps   # also prints all 9 positions of the worked example
```

## Files

```
Day-4/
├── convolution_lab.html      the learning page (content and layout)
├── verify_convolution.py     recomputes every answer with NumPy (two independent ways)
├── learn.txt                 quick revision notes
└── assets/
    ├── conv-math.js          the maths only: pad, flip, ⊙, sum, cross-correlate, output size
    ├── components.js         reusable UI: MatrixGrid, step card, ConvolutionStepper, Exercise, quizzes
    ├── lab-data.js           every matrix + expected answers (JSON, shared with the Python check)
    ├── app.js                connects data and components to the page sections
    └── styles.css            layout, light/dark colours, responsive rules
```

The arithmetic lives in one place, `conv-math.js`. The page never hard-codes a result: it computes everything and logs a console error if a result disagrees with the expected answer in `lab-data.js`. `verify_convolution.py` checks the same expected answers with NumPy.

## The worked example (in short)

Input `I` (5×5) and kernel `K` (3×3):

```
I = 1 2 0 1 3        K =  1  0 -1
    0 1 3 2 1             1  0 -1
    2 0 1 4 0             1  0 -1
    1 3 2 0 1
    0 1 1 2 2
```

Output size: `⌊(5 − 3 + 2·0) / 1⌋ + 1 = 3`, so the feature map is 3×3.

Position 1, output `O[0][0]` (window rows 0–2, columns 0–2):

```
region     ⊙  kernel      =  products
1 2 0         1 0 -1          1 0  0
0 1 3         1 0 -1          0 0 -3
2 0 1         1 0 -1          2 0 -1

sum = 1 + 0 + 0 + 0 + 0 + (−3) + 2 + 0 + (−1) = −1   →  O[0][0] = −1
```

Repeating this for all 9 positions (slide 1 column right; at the end of a row, move 1 row down and back to column 0) gives:

```
O = −1  −4   0
    −3  −2   4
    −1  −2   1
```

Shortcut check: this kernel computes *(left-column sum) − (right-column sum)* of each window, which is why it detects **vertical edges**.

## Key formulas

| | Formula |
|---|---|
| One output value | `O[i][j] = Σₘ Σₙ I[i·S + m][j·S + n] · K[m][n] + b` |
| Output size | `⌊(N − F + 2P) / S⌋ + 1` (apply to height and width separately) |
| "Same" padding | `P = (F − 1) / 2` (odd F, S = 1) |
| Parameters of a layer | `K × (F·F·C + 1)` |
| Output volume | `H_out × W_out × K` (one feature map per filter) |

CNN libraries (PyTorch, TensorFlow/Keras, `cv2.filter2D`) compute **cross-correlation**: they do not flip the kernel, even though the layer is called "convolution". The textbook convolution rotates the kernel by 180° first. For the Day-4 kernel this flips the sign of every output.

## Sources

- Afshine & Shervine Amidi, [Stanford CS230 CNN Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks) (MIT licence). Diagrams used for the convolution layer, stride, padding and filter dimensions.
- V. Dumoulin & F. Visin, [conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic) (MIT licence) and [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285). Animated GIFs for stride and padding.
- [Stanford CS231n: Convolutional Networks](https://cs231n.github.io/convolutional-networks/)
- [Dive into Deep Learning §7.2](https://d2l.ai/chapter_convolutional-neural-networks/conv-layer.html) and [§7.3](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html)
- [Setosa: Image Kernels explained visually](https://setosa.io/ev/image-kernels/)
- [PyTorch `nn.Conv2d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

The external images are loaded from their original sites, not copied into this repository.
