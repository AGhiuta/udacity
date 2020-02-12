# Problem Set 1: Edges and Lines

### 1. Edge image
a0) Input image - a1) Edge image  
<img src="input/ps1-input0.png" height="180">
<img src="output/ps1-1-a-1.png" height="180">

### 2. Linear Hough Transform
a) Hough Accumulator - b) Hough Peaks - c) Hough lines  
<img src="output/ps1-2-a-1.png" height="180" width="60">
<img src="output/ps1-2-b-1.png" height="180" width="60">
<img src="output/ps1-2-c-1.png" height="180">

### 3. Linear Hough Transform on Noisy Image
a0) Input image with noise - a1) Smoothed image - b1) Edge image from original - b2) Edge image from smoothed  
<img src="input/ps1-input0-noise.png" height="180">
<img src="output/ps1-3-a-1.png" height="180">
<img src="output/ps1-3-b-1.png" height="180">
<img src="output/ps1-3-b-2.png" height="180">  
c1) Hough Peaks - c2) Original image with Hough Lines  
<img src="output/ps1-3-c-1.png" height="180" width="60">
<img src="output/ps1-3-c-2.png" height="180">

### 4. Linear Hough Transform on Complex Image
a0) Input image - a1) Smoothed image - b1) Edge image  
<img src="input/ps1-input1.png" height="180">
<img src="output/ps1-4-a-1.png" height="180">
<img src="output/ps1-4-b-1.png" height="180">  
c1) Hough Peaks - c2) Original image with Hough Lines
<img src="output/ps1-4-c-1.png" height="180" width="60">
<img src="output/ps1-4-c-2.png" height="180">

### 5. Circular Hough Transform
a0) Input image - a1) Smoothed image - a2) Edge image  
<img src="input/ps1-input1.png" height="180">
<img src="output/ps1-5-a-1.png" height="180">
<img src="output/ps1-5-a-2.png" height="180">  
a3) Original Image with Hough Circles of known radius (r=20 px) - b1) Original Image with Hough Circles of unknown radius (r=[20, 50] px)
<img src="output/ps1-5-a-3.png" height="180">
<img src="output/ps1-5-b-1.png" height="180">  

### 6. Linear Hough Transform on Cluttered Image
a0) Input image - a1) Smoothed image with Hough lines - c1) Smoothed image with constrained Hough Lines
<img src="input/ps1-input2.png" height="180">
<img src="output/ps1-6-a-1.png" height="180">
<img src="output/ps1-6-c-1.png" height="180">  
c) constraint: parallel lines, in a distance smaller than a given threshold (rho_max)

### 7. Circular Hough Transform on Cluttered Image
a0) Input image - a1) Smoothed image with Hough Circles of unknown radius (r=[20, 40] px)
<img src="input/ps1-input2.png" height="180">
<img src="output/ps1-7-a-1.png" height="180">

### 8. Linear Hough Transform on Distorted Image
a0) Input image - a1) Smoothed image with Hough Lines
<img src="input/ps1-input3.png" height="180">
<img src="output/ps1-8-a-1.png" height="180">  
b) An Elliptical Hough Transform must be applied in order
to detect circles (ellipses) on distorted images.

