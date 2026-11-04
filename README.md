# Computer-Vision-Learning

Hands-on OpenCV practice in Python, built while following a computer vision learning playlist. Each topic is a small Jupyter notebook covering one idea: reading images and video, basic filters, cropping and resizing, drawing, perspective transforms, and joining images.

## Setup

Written with Python 3.11.

```bash
pip install -r requirements.txt
```

This installs `opencv-python` and `matplotlib`. You'll also need Jupyter, or VS Code with the Jupyter extension, to open the notebooks.

Open the notebooks from inside their day folder (for example `Day-2/`) so the relative `data/...` image paths resolve.

## Progress

| # | Topic | Status | Notebook |
|---|-------|--------|----------|
| 1 | Reading images | ✅ Done | [reading image.ipynb](Day-2/reading%20image.ipynb) |
| 2 | Reading videos | ✅ Done | [reading video.ipynb](Day-2/reading%20video.ipynb) |
| 3 | Reading webcam | ⏳ To do | |
| 4 | Basic functions | ✅ Done | [basic function.ipynb](Day-2/basic%20function.ipynb) |
| 5 | Cropping and resizing | ✅ Done | [croping image.ipynb](Day-2/croping%20image.ipynb), [resize image.ipynb](Day-2/resize%20image.ipynb) |
| 6 | Shapes and text | ✅ Done | [shapes text.ipynb](Day-2/shapes%20text.ipynb) |
| 7 | Warp perspective | ✅ Done | [wrap_prespective.ipynb](Day-2/wrap_prespective.ipynb) |
| 8 | Joining images | ✅ Done | [joining image.ipynb](Day-2/joining%20image.ipynb) |

## Day-2 notebooks

- **Reading images**: load an image with `cv2.imread`, check its shape, scale a very large photo down to fit the screen, and show it in an OpenCV window or inline in the notebook.
- **Reading videos**: read a video frame by frame with `cv2.VideoCapture`, stop cleanly at the end, resize 4K frames for display, and press `q` to quit.
- **Basic functions**: convert to grayscale (`cv2.cvtColor`), blur (`cv2.GaussianBlur`), and detect edges (`cv2.Canny`), with before/after results shown side by side.
- **Cropping**: cut out a region with NumPy slicing (`image[y1:y2, x1:x2]`).
- **Resizing**: change an image's size with `cv2.resize`.
- **Shapes and text**: draw on a blank canvas with `cv2.line`, `cv2.rectangle`, `cv2.circle`, and `cv2.putText`.
- **Warp perspective**: map four points to a flat, top-down view with `cv2.getPerspectiveTransform` and `cv2.warpPerspective`.
- **Joining images**: put images side by side with `np.hstack` and stack them with `np.vstack`, then display them inline with matplotlib.

## Notes

- **Video files aren't in the repo.** `.mp4` and other video files are git-ignored because they exceed GitHub's 100 MB file limit. To run the video notebook, put your own video at `Day-2/data/video.mp4`.
- **`cv2.imshow` windows in VS Code.** `cv2.imshow` opens a separate pop-up window, which can open behind the editor. The cell keeps running until you press a key in that window. If nothing seems to happen, look for the window in the taskbar, or display the image inline with matplotlib or `IPython.display` instead.
- **Colour order.** OpenCV stores colours as BGR, matplotlib expects RGB. Convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` before calling `plt.imshow`, or the reds and blues will be swapped.
- **Joining images.** `np.hstack` needs images with the same height and `np.vstack` needs the same width. Both also need the same number of channels, so convert a grayscale image to 3 channels first.

## License

[MIT](LICENSE)
