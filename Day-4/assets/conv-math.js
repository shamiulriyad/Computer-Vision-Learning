/*
 * conv-math.js — the mathematics of the convolution layer, with no DOM code.
 *
 * Everything the page shows (worked steps, the animation, the exercise
 * answers) is computed by the functions in this file, so there is exactly one
 * place where the arithmetic lives. It runs in the browser (window.ConvMath)
 * and in Node (require('./conv-math.js')), which is how it is tested.
 *
 * Conventions
 *   - A matrix is an array of rows: m[row][col].
 *   - Indices are 0-based, like NumPy: the top-left element is m[0][0].
 *   - "Convolution" in CNNs is really cross-correlation (no kernel flip).
 *     crossCorrelate() is what PyTorch / TensorFlow / cv2.filter2D compute.
 *     convolve() is the textbook definition (flip the kernel first).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.ConvMath = factory();
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  function dims(m) {
    return { rows: m.length, cols: m[0].length };
  }

  function zeros(rows, cols) {
    return Array.from({ length: rows }, function () { return new Array(cols).fill(0); });
  }

  function clone(m) {
    return m.map(function (row) { return row.slice(); });
  }

  /*
   * Zero padding: surround the matrix with P rings of zeros.
   * An N×M input becomes (N + 2P)×(M + 2P). Original value m[r][c]
   * moves to out[r + P][c + P].
   */
  function pad(m, p) {
    if (!p) return clone(m);
    var d = dims(m);
    var out = zeros(d.rows + 2 * p, d.cols + 2 * p);
    for (var r = 0; r < d.rows; r++) {
      for (var c = 0; c < d.cols; c++) out[r + p][c + p] = m[r][c];
    }
    return out;
  }

  /*
   * Output Size = floor((N − F + 2P) / S) + 1
   *   N = input size, F = filter size, P = padding, S = stride.
   * Applied separately to height and width.
   */
  function outputSize(N, F, P, S) {
    return Math.floor((N - F + 2 * P) / S) + 1;
  }

  /*
   * Same formula, but returns every intermediate number so the page can
   * print the substitution: (N − F + 2P) → ÷ S → floor → + 1.
   */
  function outputSizeSteps(N, F, P, S) {
    var numerator = N - F + 2 * P;
    var quotient = numerator / S;
    var floored = Math.floor(quotient);
    return {
      N: N, F: F, P: P, S: S,
      twoP: 2 * P,
      numerator: numerator,
      quotient: quotient,
      floored: floored,
      result: floored + 1,
      exact: numerator % S === 0,        // false → the last rows/cols are never visited
      valid: numerator >= 0 && S > 0
    };
  }

  /* Rotate the kernel by 180°: reverse the row order and each row. */
  function flip(k) {
    return k.slice().reverse().map(function (row) { return row.slice().reverse(); });
  }

  /* Take an h×w block whose top-left corner is (r0, c0). */
  function slice(m, r0, c0, h, w) {
    var out = [];
    for (var r = 0; r < h; r++) out.push(m[r0 + r].slice(c0, c0 + w));
    return out;
  }

  /*
   * Element-wise (Hadamard) product A ⊙ B: multiply elements that sit in the
   * same position. Both matrices must have the same shape. This is NOT
   * matrix multiplication.
   */
  function hadamard(a, b) {
    return a.map(function (row, r) {
      return row.map(function (v, c) { return v * b[r][c]; });
    });
  }

  /* Add every element of a matrix. */
  function sum(m) {
    var s = 0;
    for (var r = 0; r < m.length; r++) for (var c = 0; c < m[r].length; c++) s += m[r][c];
    return s;
  }

  /* Ordinary matrix multiplication (only used to show the common mistake). */
  function matmul(a, b) {
    var n = a.length, k = b.length, m = b[0].length;
    var out = zeros(n, m);
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < m; j++) {
        var s = 0;
        for (var t = 0; t < k; t++) s += a[i][t] * b[t][j];
        out[i][j] = s;
      }
    }
    return out;
  }

  function flatten(m) {
    return m.reduce(function (acc, row) { return acc.concat(row); }, []);
  }

  /*
   * Cross-correlation, the operation a CNN layer performs:
   *
   *   O[i][j] = Σ_m Σ_n  Ipad[i·S + m][j·S + n] · K[m][n]
   *
   * For every output position (i, j):
   *   1. the window's top-left corner in the padded input is (i·S, j·S)
   *   2. extract the F×F region under the window
   *   3. multiply it element-wise with the kernel
   *   4. add the products → one number
   *   5. store it in O[i][j], then slide S columns right (or S rows down
   *      and back to the left edge at the end of a row)
   *
   * Returns the output and every intermediate step.
   */
  function crossCorrelate(input, kernel, opts) {
    opts = opts || {};
    var stride = opts.stride == null ? 1 : opts.stride;
    var padding = opts.padding == null ? 0 : opts.padding;
    var bias = opts.bias == null ? 0 : opts.bias;

    var inD = dims(input), kD = dims(kernel);
    var outH = outputSize(inD.rows, kD.rows, padding, stride);
    var outW = outputSize(inD.cols, kD.cols, padding, stride);
    if (stride < 1 || outH < 1 || outW < 1) {
      throw new Error('Kernel does not fit: output would be ' + outH + '×' + outW);
    }

    var padded = pad(input, padding);
    var output = zeros(outH, outW);
    var steps = [];

    for (var i = 0; i < outH; i++) {
      for (var j = 0; j < outW; j++) {
        var row = i * stride;                       // window top-left row (padded coords)
        var col = j * stride;                       // window top-left col (padded coords)
        var region = slice(padded, row, col, kD.rows, kD.cols);
        var products = hadamard(region, kernel);    // element-wise multiplication
        var total = sum(products);                  // summation of all products
        var value = total + bias;
        output[i][j] = value;                       // one number per window position
        steps.push({
          index: steps.length,
          outRow: i, outCol: j,
          row: row, col: col,
          h: kD.rows, w: kD.cols,
          region: region,
          products: products,
          sum: total,
          value: value
        });
      }
    }

    return {
      input: input, kernel: kernel,
      stride: stride, padding: padding, bias: bias,
      padded: padded,
      output: output, outH: outH, outW: outW,
      steps: steps
    };
  }

  /* Textbook (mathematical) convolution: flip the kernel 180°, then cross-correlate. */
  function convolve(input, kernel, opts) {
    var res = crossCorrelate(input, flip(kernel), opts);
    res.originalKernel = kernel;
    return res;
  }

  /*
   * Multi-channel input (e.g. RGB): the filter has one F×F slice per channel.
   * Each channel is cross-correlated with its own slice, the channel results
   * are added together, and one bias is added → ONE feature map per filter.
   */
  function crossCorrelateMulti(channels, kernels, opts) {
    opts = opts || {};
    var bias = opts.bias == null ? 0 : opts.bias;
    var perChannel = channels.map(function (ch, c) {
      return crossCorrelate(ch, kernels[c], { stride: opts.stride, padding: opts.padding });
    });
    var outH = perChannel[0].outH, outW = perChannel[0].outW;
    var output = zeros(outH, outW);
    for (var i = 0; i < outH; i++) {
      for (var j = 0; j < outW; j++) {
        var s = bias;
        perChannel.forEach(function (res) { s += res.output[i][j]; });
        output[i][j] = s;
      }
    }
    return { perChannel: perChannel, bias: bias, output: output, outH: outH, outW: outW };
  }

  /*
   * Which window starts where. Returns a matrix the size of the padded input:
   * cells that are a window's top-left corner hold the step number (1-based),
   * other cells hold 0. Also reports cells no window ever covers (this
   * happens when (N − F + 2P) is not divisible by S).
   */
  function visitMap(rows, cols, F, P, S) {
    var Hp = rows + 2 * P, Wp = cols + 2 * P;
    var outH = outputSize(rows, F, P, S), outW = outputSize(cols, F, P, S);
    var starts = zeros(Hp, Wp);
    var covered = zeros(Hp, Wp);
    var n = 0;
    for (var i = 0; i < outH; i++) {
      for (var j = 0; j < outW; j++) {
        n += 1;
        starts[i * S][j * S] = n;
        for (var a = 0; a < F; a++) for (var b = 0; b < F; b++) covered[i * S + a][j * S + b] = 1;
      }
    }
    return { starts: starts, covered: covered, outH: outH, outW: outW };
  }

  /* Number formatting for display: real minus sign, no "-0", ≤ 2 decimals. */
  function fmt(n) {
    if (Object.is(n, -0) || n === 0) return '0';
    var s = Number.isInteger(n) ? String(Math.abs(n)) : String(+Math.abs(n).toFixed(2));
    return (n < 0 ? '−' : '') + s;
  }

  /* Wrap negatives in parentheses so "3 + (−2)" reads correctly. */
  function fmtTerm(n) {
    return n < 0 ? '(' + fmt(n) + ')' : fmt(n);
  }

  /* "1 + 0 + (−3) + …" for the summation line. */
  function sumExpression(products) {
    return flatten(products).map(fmtTerm).join(' + ');
  }

  return {
    dims: dims, zeros: zeros, clone: clone,
    pad: pad, flip: flip, slice: slice,
    hadamard: hadamard, sum: sum, matmul: matmul, flatten: flatten,
    outputSize: outputSize, outputSizeSteps: outputSizeSteps,
    crossCorrelate: crossCorrelate, convolve: convolve,
    crossCorrelateMulti: crossCorrelateMulti,
    visitMap: visitMap,
    fmt: fmt, fmtTerm: fmtTerm, sumExpression: sumExpression
  };
}));
