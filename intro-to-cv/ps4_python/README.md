# Problem Set 4: Harris Corners, SIFT & RANSAC

### 1. Harris Corners

##### a) X and Y gradients 
transA gradient-pair image:  
<img src="input/transA.jpg" width="200">
<img src="output/ps4-1-a-1.png" width="400">  
simA  gradient-pair image:  
<img src="input/simA.jpg" width="200">
<img src="output/ps4-1-a-2.png" width="400">  

##### b) Harris Values
Input images simA, simB, transA, transB:  
<img src="input/transA.jpg" width="200">
<img src="input/transB.jpg" width="200">
<img src="input/simA.jpg" width="200">
<img src="input/simB.jpg" width="200">
Corresponding Harris Value images:  
<img src="output/ps4-1-b-1.png" width="200">
<img src="output/ps4-1-b-2.png" width="200">
<img src="output/ps4-1-b-3.png" width="200">
<img src="output/ps4-1-b-4.png" width="200">

##### c) Harris Corners
Images with Harris Corners marked:  
<img src="output/ps4-1-c-1.png" width="200">
<img src="output/ps4-1-c-2.png" width="200">
<img src="output/ps4-1-c-3.png" width="200">
<img src="output/ps4-1-c-4.png" width="200">


### 2. SIFT Features

##### a) Interest Points on trans and sim image pairs
<img src="output/ps4-2-a-1.png" width="400">
<img src="output/ps4-2-a-2.png" width="400">

##### b) Putative pair images for the trans and sim image pairs
<img src="output/ps4-2-b-1.png" width="400">
<img src="output/ps4-2-b-2.png" width="400">


###  3. RANSAC
##### a) Largest consensus set drawn on the trans image pair
<img src="output/ps4-3-a-1.png" width="400">
```
Translation vector: [-127.  -74.]
Percentage of matches in the biggest consensus set: 4.48
```

##### b) Largest consensus set drawn on the sim image pair using similarity transform comparison
<img src="output/ps4-3-b-1.png" width="400">
```
Transform matrix: [[  0.95539488  -0.27397108  41.49677419]
                   [  0.27397108   0.95539488 -50.81290323]]
Percentage of matches in the biggest consensus set: 43.90
```

##### c) Largest consensus set drawn on the sim image pair using Affine transform comparison
<img src="output/ps4-3-c-1.png" width="400">
```
Transform matrix: [[  0.97523671  -0.20320466  22.5826657 ]
                   [  0.29133285   0.97887837 -62.79606701]]
Percentage of matches in the biggest consensus set: 35.77
```

##### d) Backwards warping and blending of the 2nd image to the 1st based on the similarity consensus set
Warped simB image:  
<img src="output/ps4-3-d-1.png" width="400">  
The simA - warped_simB overlay image:
<img src="output/ps4-3-d-2.png" width="400">

##### e) Backwards warping and blending of the 2nd image to the 1st based on the Affine consensus set
Warped simB image:  
<img src="output/ps4-3-e-1.png" width="400">
The simA - warped_simB overlay image:  
<img src="output/ps4-3-e-2.png" width="400">

