# SYCL-Pathtracer

A path tracer with next-event estimation in a single C++ file using SYCL for acceleration. 

This project was inspired by [Reinhold Preiner's single-source GLSL path tracer in 111 lines of code](https://github.com/rpreiner/pocketpt), which itself is based on [Kevin Beason's smallpt](http://kevinbeason.com/smallpt). This project does not quite hit the sub 111 LOC target, but should remain quite readable at roughly 450 LOC.

<img src="512x512@2Kspp.png" width="512">