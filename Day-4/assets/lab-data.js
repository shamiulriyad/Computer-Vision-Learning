/*
 * lab-data.js — every matrix used on the page, plus the expected answers.
 *
 * The block between JSON-START and JSON-END is strict JSON so that
 * verify_convolution.py can read the very same numbers and check them with
 * NumPy. The page itself recomputes every answer with conv-math.js and warns
 * in the browser console if anything disagrees with "expected".
 *
 * Exercise fields: input, kernel, stride, padding, expected (the feature map).
 */
var LAB_DATA = /*JSON-START*/ {
  "worked": {
    "input": [
      [1, 2, 0, 1, 3],
      [0, 1, 3, 2, 1],
      [2, 0, 1, 4, 0],
      [1, 3, 2, 0, 1],
      [0, 1, 1, 2, 2]
    ],
    "kernel": [
      [1, 0, -1],
      [1, 0, -1],
      [1, 0, -1]
    ],
    "expected": [
      [-1, -4, 0],
      [-3, -2, 4],
      [-1, -2, 1]
    ]
  },

  "concepts": {
    "trueConvolutionExpected": [[1, 4, 0], [3, 2, -4], [1, 2, -1]],
    "stride2Expected": [[-1, 0], [-1, 1]],
    "padding1Expected": [
      [-3, -2, 0, -1, 3],
      [-3, -1, -4, 0, 7],
      [-4, -3, -2, 4, 6],
      [-4, -1, -2, 1, 6],
      [-4, -2, 2, 0, 2]
    ],
    "padding1Stride2Expected": [[-3, 0, 3], [-4, -2, 6], [-4, 2, 2]],
    "horizontalKernel": [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
    "horizontalExpected": [[0, -2, -1], [-2, 1, 3], [1, 1, 0]],
    "multiChannel": {
      "channels": [
        [[1, 2, 0], [0, 1, 3], [2, 1, 1]],
        [[0, 1, 1], [2, 0, 1], [1, 1, 0]]
      ],
      "kernels": [
        [[1, 0], [0, -1]],
        [[0, 1], [1, 0]]
      ],
      "bias": 1,
      "perChannelExpected": [[[0, -1], [-1, 0]], [[3, 1], [1, 2]]],
      "expected": [[4, 1], [1, 3]]
    }
  },

  "inputs": {
    "day4": {
      "label": "Day-4 example (5×5)",
      "matrix": [
        [1, 2, 0, 1, 3],
        [0, 1, 3, 2, 1],
        [2, 0, 1, 4, 0],
        [1, 3, 2, 0, 1],
        [0, 1, 1, 2, 2]
      ]
    },
    "vEdge": {
      "label": "Vertical edge: bright | dark (6×6)",
      "matrix": [
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0],
        [10, 10, 10, 0, 0, 0]
      ]
    },
    "hEdge": {
      "label": "Horizontal edge: bright over dark (6×6)",
      "matrix": [
        [10, 10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10, 10],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
      ]
    }
  },

  "kernels": {
    "vertical":   { "label": "Vertical edge (Day-4 kernel)", "matrix": [[1, 0, -1], [1, 0, -1], [1, 0, -1]] },
    "horizontal": { "label": "Horizontal edge",              "matrix": [[1, 1, 1], [0, 0, 0], [-1, -1, -1]] },
    "sobelX":     { "label": "Sobel X",                      "matrix": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] },
    "sharpen":    { "label": "Sharpen",                      "matrix": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
    "laplacian":  { "label": "Laplacian (all edges)",        "matrix": [[0, 1, 0], [1, -4, 1], [0, 1, 0]] },
    "identity":   { "label": "Identity (copies the centre)", "matrix": [[0, 0, 0], [0, 1, 0], [0, 0, 0]] },
    "boxSum":     { "label": "Box sum (all ones)",           "matrix": [[1, 1, 1], [1, 1, 1], [1, 1, 1]] }
  },

  "exercises": [
    {
      "id": "ex1",
      "level": "Beginner",
      "title": "One window, one number",
      "tags": ["3×3 input", "3×3 kernel", "S = 1", "P = 0"],
      "prompt": "The kernel is exactly as big as the input, so it fits in only one position. Multiply the nine pairs and add them up.",
      "hint": "Only the corners and the centre of the kernel are 1. So the answer is the sum of the input's four corners plus its centre.",
      "input": [[3, 1, 2], [0, 4, 1], [2, 1, 0]],
      "kernel": [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
      "stride": 1, "padding": 0,
      "expected": [[11]]
    },
    {
      "id": "ex2",
      "level": "Beginner",
      "title": "A 2×2 kernel with a negative weight",
      "tags": ["3×3 input", "2×2 kernel", "S = 1", "P = 0", "negative weight"],
      "prompt": "Slide the 2×2 kernel over every valid position. There are four. Watch the sign of the −1 weight.",
      "hint": "Each output = 2·(top-left) − (top-right) + 0·(bottom-left) + 1·(bottom-right) of the 2×2 window.",
      "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
      "kernel": [[2, -1], [0, 1]],
      "stride": 1, "padding": 0,
      "expected": [[5, 7], [11, 13]]
    },
    {
      "id": "ex3",
      "level": "Beginner+",
      "title": "Sobel X: positive and negative weights",
      "tags": ["4×4 input", "3×3 kernel", "S = 1", "P = 0", "negative weights"],
      "prompt": "Apply the Sobel X kernel. The middle column is all zeros, so you can skip those three products, but write them down the first time.",
      "hint": "Output = (right column of window, weighted 1, 2, 1) − (left column of window, weighted 1, 2, 1).",
      "input": [[2, 3, 7, 4], [6, 6, 9, 8], [3, 4, 8, 3], [4, 1, 0, 2]],
      "kernel": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
      "stride": 1, "padding": 0,
      "expected": [[16, 4], [9, 1]]
    },
    {
      "id": "ex4",
      "level": "Intermediate",
      "title": "Sharpen kernel",
      "tags": ["4×4 input", "3×3 kernel", "S = 1", "P = 0", "mixed signs"],
      "prompt": "The sharpen kernel boosts the centre pixel (×5) and subtracts its four direct neighbours.",
      "hint": "Output = 5·centre − (up + down + left + right). The corners of the window are multiplied by 0.",
      "input": [[1, 2, 1, 0], [2, 4, 3, 1], [1, 3, 5, 2], [0, 1, 2, 1]],
      "kernel": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
      "stride": 1, "padding": 0,
      "expected": [[10, 4], [4, 15]]
    },
    {
      "id": "ex5",
      "level": "Intermediate",
      "title": "Stride 2",
      "tags": ["5×5 input", "3×3 kernel", "S = 2", "P = 0"],
      "prompt": "First use the formula to find the output size. Then the window jumps 2 columns at a time, and 2 rows when it reaches the end of a row.",
      "hint": "Output size = ⌊(5 − 3 + 0)/2⌋ + 1 = 2. The window's top-left corners are (0,0), (0,2), (2,0), (2,2).",
      "input": [[1, 0, 2, 1, 0], [3, 1, 0, 2, 1], [0, 2, 1, 1, 3], [1, 1, 0, 2, 0], [2, 0, 1, 1, 1]],
      "kernel": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
      "stride": 2, "padding": 0,
      "expected": [[1, -5], [-1, -6]]
    },
    {
      "id": "ex6",
      "level": "Intermediate",
      "title": "Padding 1 keeps the size",
      "tags": ["3×3 input", "3×3 kernel", "S = 1", "P = 1"],
      "prompt": "Pad the input with one ring of zeros (it becomes 5×5), then slide the kernel. The output should be the same size as the original input.",
      "hint": "Write out the 5×5 padded matrix first. The kernel is a plus sign: output = centre + up + down + left + right, where anything outside the original image is 0.",
      "input": [[1, 2, 0], [3, 1, 1], [0, 2, 4]],
      "kernel": [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
      "stride": 1, "padding": 1,
      "expected": [[6, 4, 3], [5, 9, 6], [5, 7, 7]]
    },
    {
      "id": "ex7",
      "level": "Advanced",
      "title": "Padding 1 and stride 2 together",
      "tags": ["4×4 input", "3×3 kernel", "S = 2", "P = 1", "negative weights"],
      "prompt": "Pad first (4×4 → 6×6), then move the window 2 cells at a time. Use the formula to get the output size before you start.",
      "hint": "Output size = ⌊(4 − 3 + 2)/2⌋ + 1 = 2. In the padded matrix the windows start at (0,0), (0,2), (2,0), (2,2).",
      "input": [[2, 1, 0, 3], [1, 4, 2, 0], [0, 2, 3, 1], [3, 0, 1, 2]],
      "kernel": [[1, 0, -1], [0, 1, 0], [-1, 0, 1]],
      "stride": 2, "padding": 1,
      "expected": [[6, -4], [-4, 9]]
    },
    {
      "id": "ex8",
      "level": "Challenge",
      "title": "Complete feature map: find the edge",
      "tags": ["6×6 input", "3×3 kernel", "S = 1", "P = 0", "16 outputs"],
      "prompt": "Use the Day-4 vertical-edge kernel on this 6×6 image. Compute all 16 values. Then look at the result: which columns of the feature map light up, and why?",
      "hint": "Shortcut for this kernel: output = (sum of the window's left column) − (sum of its right column). The image is bright on the left and dark on the right.",
      "input": [
        [4, 4, 4, 1, 1, 1],
        [4, 5, 4, 1, 0, 1],
        [3, 4, 4, 2, 1, 0],
        [4, 4, 5, 1, 1, 1],
        [4, 3, 4, 0, 1, 2],
        [5, 4, 4, 1, 1, 1]
      ],
      "kernel": [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
      "stride": 1, "padding": 0,
      "expected": [[-1, 9, 10, 2], [-2, 9, 11, 2], [-2, 8, 10, 0], [0, 9, 10, -2]]
    }
  ],

  "sizeQuiz": [
    { "N": 32,  "F": 5, "P": 0, "S": 1, "answer": 28 },
    { "N": 32,  "F": 3, "P": 1, "S": 1, "answer": 32 },
    { "N": 28,  "F": 5, "P": 2, "S": 2, "answer": 14 },
    { "N": 7,   "F": 3, "P": 0, "S": 2, "answer": 3 },
    { "N": 6,   "F": 3, "P": 0, "S": 2, "answer": 2 },
    { "N": 224, "F": 7, "P": 3, "S": 2, "answer": 112 }
  ],

  "filterQuiz": [
    {
      "question": "Input 32×32×3 (RGB). The layer has 10 filters of size 5×5, P = 0, S = 1. How many feature maps does it output?",
      "answer": 10,
      "solution": "One filter → one feature map. 10 filters → 10 feature maps. Output volume = 28×28×10, since ⌊(32 − 5 + 0)/1⌋ + 1 = 28."
    },
    {
      "question": "Same layer. How many learnable parameters (weights + biases)?",
      "answer": 760,
      "solution": "Each filter is 5×5×3 = 75 weights + 1 bias = 76. With 10 filters: 10 × 76 = 760."
    }
  ]
} /*JSON-END*/;

if (typeof module === 'object' && module.exports) module.exports = LAB_DATA;
