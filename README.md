# Computer-Vision-Learning
A collection of hands-on OpenCV projects and practice exercises using Python, covering image processing, video processing, image transformations, camera operations, and computer vision fundamentals while following a learning playlist.

## Setup

```bash
pip install -r requirements.txt
```

Run the notebooks from inside their day folder (e.g. `Day-2/`) so the relative `data/` paths resolve.

## Progress

| # | Topic | Status | Code |
|---|-------|--------|------|
| 1 | Reading images | ✅ Done | [Day-2/reading image.ipynb](Day-2/reading%20image.ipynb) |
| 2 | Reading videos | ✅ Done | [Day-2/reading video.ipynb](Day-2/reading%20video.ipynb) |
| 3 | Reading webcam | ⏳ | |
| 4 | Basic functions | ⏳ | |
| 5 | Image cropping and resizing | ⏳ | |
| 6 | Shapes and text | ⏳ | |
| 7 | Warp perspective | ⏳ | |
| 8 | Joining images | ⏳ | |

## Day-2

- **Reading images** — load an image with `cv2.imread`, check its shape, resize a large image to fit the screen, and display it either in an OpenCV window or inline in the notebook.
- **Reading videos** — read a video frame by frame with `cv2.VideoCapture`, stop cleanly at the end of the video, resize 4K frames for display, and press `q` to quit.

> **Note:** Video files (`*.mp4`) are not included in the repo because they exceed GitHub's 100 MB file limit. Put your own video at `Day-2/data/video.mp4` to run the video notebook.
