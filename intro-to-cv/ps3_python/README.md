# Problem Set 3: Geometry

### 1. Calibration
a) Projection matrix:  
```
M = [[ 0.76785834 -0.49384797 -0.02339781  0.00674445]
     [-0.0852134  -0.09146818 -0.90652332 -0.08775678]
     [ 0.18265016  0.29882917 -0.07419242  1.        ]]
3D point [1.2323 1.4421 0.4506] projected to [ 0.14190586 -0.45183985]
Residual: 0.0016
```  

b) Projection matrix estimation using 8, 12 and 16 [3d, 2d] pairs:  
```
Best M: [[-2.04785625e+00  1.18569189e+00  4.12295509e-01  2.43947598e+02]
         [-4.53919622e-01 -3.02538918e-01  2.14961233e+00  1.65096749e+02]
         [-2.24168262e-03 -1.10021965e-03  5.71417125e-04  1.00000000e+00]]
Residual val for best M: 1.4230
```
 
c) Location of the camera in 3D world coordinates:  
```
C: [303.09777053 307.19723727  30.42713161]
```


### 2. Fundamental Matrix Estimation
a) Fundamental Matrix Estimation using the least squares method on 20 2d point pair:  

b) Fundamental Matrix Rank Reduction from 3 to 2:  

c) Epipolar lines estimation using F and the 2D point pairs in each image  

d) F Improvement using the normalization matrices Ta and Tb:  

e) Improved Fundamental Matrix and Epipolar Lines using normalization:  
