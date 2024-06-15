# Simple and Canny Edge Detectors

This repository revisits the implementation of simple edge detection and the Canny edge detection algorithm from scratch

<p align="center">
  <img src="img.png" alt="khoa16122004" />
</p>

## Simple Edge Detector

**`simple_edge_detector.py`**: Implements basic edge detection using Sobel, Perwitt operators and gradient thresholding. Detects edges but may result in thicker edges.

## Canny Edge Detector

**`canny_edge_detector.py`**: Implements the Canny edge detection algorithm.
  - Applies Gaussian smoothing to reduce noise.

  - Uses Sobel operators for gradient calculation and considers gradient orientation to verify edge presence.

