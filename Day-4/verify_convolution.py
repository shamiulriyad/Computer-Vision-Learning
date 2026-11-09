"""Day-4: check every number in the convolution lab, and print the worked example.

Run from the Day-4 folder:
    python verify_convolution.py            # verify everything
    python verify_convolution.py --steps    # also print all 9 positions of the worked example

It reads the matrices straight from assets/lab-data.js (the JSON block between
the JSON-START / JSON-END markers), so the page and this script always use the
same numbers. Each result is computed twice:

  1. conv2d_loops: the plain-Python definition with explicit loops,
     the way you would do it with a pencil;
  2. conv2d_numpy: a vectorised NumPy version (sliding windows + einsum);

and both must equal the "expected" answers stored in the data file.
Only NumPy is needed.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HERE = Path(__file__).parent


def load_lab_data():
    text = (HERE / "assets" / "lab-data.js").read_text(encoding="utf-8")
    block = re.search(r"/\*JSON-START\*/(.*)/\*JSON-END\*/", text, re.S).group(1)
    return json.loads(block)


def output_size(n, f, p, s):
    # Output Size = floor((N - F + 2P) / S) + 1
    return (n - f + 2 * p) // s + 1


def conv2d_loops(image, kernel, stride=1, padding=0):
    """Cross-correlation (what CNNs call convolution), written out step by step."""
    image = np.pad(np.asarray(image), padding)          # zero padding on all four sides
    kernel = np.asarray(kernel)
    fh, fw = kernel.shape
    out_h = output_size(image.shape[0] - 2 * padding, fh, padding, stride)
    out_w = output_size(image.shape[1] - 2 * padding, fw, padding, stride)
    out = np.zeros((out_h, out_w), dtype=image.dtype)
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride                  # window top-left corner
            region = image[r:r + fh, c:c + fw]             # 1. extract region
            products = region * kernel                     # 2. element-wise multiply
            out[i, j] = products.sum()                     # 3. sum, 4. store
    return out


def conv2d_numpy(image, kernel, stride=1, padding=0):
    """Same operation, vectorised: every window at once, then multiply-and-sum."""
    image = np.pad(np.asarray(image), padding)
    kernel = np.asarray(kernel)
    windows = sliding_window_view(image, kernel.shape)[::stride, ::stride]
    return np.einsum("ijkl,kl->ij", windows, kernel)


def check(name, image, kernel, expected, stride=1, padding=0):
    a = conv2d_loops(image, kernel, stride, padding)
    b = conv2d_numpy(image, kernel, stride, padding)
    exp = np.asarray(expected)
    ok = np.array_equal(a, b) and np.array_equal(a, exp)
    print(f"  {'OK ' if ok else 'FAIL'}  {name:<38} -> {a.tolist()}")
    if not ok:
        print(f"        loops={a.tolist()} numpy={b.tolist()} expected={exp.tolist()}")
    return ok


def print_steps(image, kernel, stride=1, padding=0):
    """Print every filter position like the page does."""
    image = np.pad(np.asarray(image), padding)
    kernel = np.asarray(kernel)
    f = kernel.shape[0]
    n = output_size(image.shape[0] - 2 * padding, f, padding, stride)
    for i in range(n):
        for j in range(n):
            r, c = i * stride, j * stride
            region = image[r:r + f, c:c + f]
            products = region * kernel
            terms = " + ".join(f"({v})" if v < 0 else str(v) for v in products.ravel())
            print(f"\nPosition {i * n + j + 1}: O[{i}][{j}]  window rows {r}-{r + f - 1}, cols {c}-{c + f - 1}")
            for reg_row, k_row, p_row in zip(region, kernel, products):
                print(f"   {str(reg_row.tolist()):<14} ⊙ {str(k_row.tolist()):<14} = {p_row.tolist()}")
            print(f"   sum = {terms} = {products.sum()}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")         # ⊙ and − on Windows consoles
    data = load_lab_data()
    w, c = data["worked"], data["concepts"]
    results = []

    print("Worked example and concepts")
    results.append(check("Day-4 example (S=1, P=0)", w["input"], w["kernel"], w["expected"]))
    # True convolution = cross-correlation with the kernel rotated 180 degrees.
    results.append(check("True convolution (flipped kernel)", w["input"], np.flip(w["kernel"]), c["trueConvolutionExpected"]))
    results.append(check("Stride 2", w["input"], w["kernel"], c["stride2Expected"], stride=2))
    results.append(check("Padding 1", w["input"], w["kernel"], c["padding1Expected"], padding=1))
    results.append(check("Padding 1 + stride 2", w["input"], w["kernel"], c["padding1Stride2Expected"], 2, 1))
    results.append(check("Horizontal-edge filter", w["input"], c["horizontalKernel"], c["horizontalExpected"]))

    mc = c["multiChannel"]
    per = [conv2d_loops(ch, k) for ch, k in zip(mc["channels"], mc["kernels"])]
    total = sum(per) + mc["bias"]
    ok = all(np.array_equal(p, e) for p, e in zip(per, mc["perChannelExpected"])) and np.array_equal(total, mc["expected"])
    print(f"  {'OK ' if ok else 'FAIL'}  {'Multi-channel (2 channels + bias)':<38} -> {total.tolist()}")
    results.append(ok)

    print("\nPractice exercises")
    for ex in data["exercises"]:
        results.append(check(f"{ex['id']}: {ex['title']}", ex["input"], ex["kernel"], ex["expected"], ex["stride"], ex["padding"]))

    print("\nOutput-size quiz")
    for q in data["sizeQuiz"]:
        got = output_size(q["N"], q["F"], q["P"], q["S"])
        ok = got == q["answer"]
        results.append(ok)
        print(f"  {'OK ' if ok else 'FAIL'}  N={q['N']:<4} F={q['F']} P={q['P']} S={q['S']}  -> {got}")

    # Parameter count for the filter quiz: K * (F*F*C + 1)
    params = 10 * (5 * 5 * 3 + 1)
    ok = params == data["filterQuiz"][1]["answer"]
    results.append(ok)
    print(f"  {'OK ' if ok else 'FAIL'}  10 filters of 5x5x3 -> {params} parameters")

    if "--steps" in sys.argv:
        print("\nAll 9 positions of the Day-4 example")
        print_steps(w["input"], w["kernel"])

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
