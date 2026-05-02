# Architectural Feature Detection using FAST

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

## 📋 Executive Summary
This repository provides a high-performance solution for detecting architectural features in images using the **FAST (Features from Accelerated Segment Test)** algorithm. Designed for speed and efficiency, this tool is ideal for applications requiring real-time analysis, such as automated property surveys, architectural style classification, and 3D reconstruction.

By identifying "corners"—points where image intensity changes significantly in multiple directions—this system can pinpoint critical structural elements (windows, balconies, rooflines) in complex architectural photography.

---

## ✨ Key Features
- **High-Speed Detection**: Leverages the FAST algorithm, optimized for real-time performance.
- **Architectural Optimization**: Pre-tested on architectural imagery (villas and bungalows).
- **Non-Maximum Suppression**: Built-in support to eliminate redundant overlapping detections.
- **Customizable Sensitivity**: Adjustable thresholds to tune detection for different lighting and textures.

---

## 🛠 Technical Deep Dive

### What is FAST?
**FAST (Features from Accelerated Segment Test)** is a corner detection method originally developed by Edward Rosten and Tom Drummond. Unlike other detectors (like Harris or SIFT) that may be computationally expensive, FAST is designed specifically for high-speed applications.

### How it Works
1. **Segment Test**: For every pixel $P$, the algorithm examines a circle of 16 pixels surrounding it.
2. **Thresholding**: $P$ is identified as a "corner" if a set of $n$ contiguous pixels in the circle are all brighter than $I_p + t$ or all darker than $I_p - t$ (where $I_p$ is the intensity of $P$ and $t$ is a threshold).
3. **High-Speed Test**: To exclude non-corners quickly, it first checks pixels at the 1, 5, 9, and 13 o'clock positions. At least three of these must satisfy the threshold condition for the pixel to be considered further.
4. **Non-Maximum Suppression**: Since many pixels in a localized area might satisfy the corner criteria, this step ensures only the strongest "peak" is kept, preventing clusters of dots on a single feature.

### Code Walkthrough (`main.py`)
The implementation uses `OpenCV` to handle image processing:
```python
# Initialize the FAST detector
fast = cv.FastFeatureDetector_create()

# Detect keypoints in grayscale
kp = fast.detect(img, None)

# Draw detected points back onto the image
img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))
```

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- OpenCV (`opencv-python`)

### Setup
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. **Install dependencies**:
   ```bash
   pip install opencv-python
   ```

### Running the Analysis
To run the detection on the default images:
```bash
python main.py
```
The script will process the sample architectural images and save the results as `result_villa.jpg` and `result_bungalow.jpg`.

---

## ⚙️ Configuration
The algorithm's behavior can be tuned by modifying the `fast` object parameters:

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `Threshold` | Minimum intensity difference between the central pixel and its neighbors. Lower values = more corners detected. | 10 |
| `NonmaxSuppression` | If `True`, removes multiple detections in the same vicinity. | `True` |
| `Type` | Neighborhood size (TYPE_9_16, TYPE_7_12, etc.) | `TYPE_9_16` |

---

## 💼 Business Use Cases

### 1. Automated Property Surveys
Real estate platforms can automatically extract structural features (e.g., number of windows, presence of balconies) to categorize listings without manual review.

### 2. Heritage Conservation
Architectural historians can use feature maps to track the "complexity" of building facades across different eras or styles (e.g., comparing "Funkis" vs. "Functionalist" styles).

### 3. Augmented Reality (AR)
The high speed of FAST makes it perfect for AR applications where digital information must be pinned to architectural corners in real-time through a mobile camera.

---

## 📸 Sample Data
The repository includes two example architectural images:
- `funktionelle-villa.jpg`: A classic functionalist villa.
- `funkisbungalowen.jpg`: A traditional "Funkis" style bungalow.

*Source: [Historiske Huse Stilguide](https://historiskehuse.dk/stilguide/#stilguide-enfamiliehuse)*

---
*Developed for high-precision architectural computer vision.*
