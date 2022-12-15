# SYCL-Pathtracer

A path tracer with next-event estimation in a single C++ file using SYCL for acceleration. 

This project was inspired by [Reinhold Preiner's single-source GLSL path tracer in 111 lines of code](https://github.com/rpreiner/pocketpt), which itself is based on [Kevin Beason's smallpt](http://kevinbeason.com/smallpt). This project does not quite hit the sub 111 LOC target, but should remain quite readable at roughly 450 LOC.

<img src="512x512@8Kspp.png" width="512"/>

# SYCL Business Card Raytracer

Business Card Raytracer in roughly 300 LOC. 
A port of Fabien Sanglard's code to SYCL and made text definition more explicit through binary literals and bit operations per character.
    
- [Fabien Sanglard on the business-card raytracer 1/2 (dissecting the code)](https://fabiensanglard.net/rayTracing_back_of_business_card/)
    
-   [Fabien Sanglard on the business-card raytracer 2/2 (Fast CPU & CUDA implementation)](https://fabiensanglard.net/revisiting_the_businesscard_raytracer/index.html)

To run, first type `make.bat` to build the executable, followed by `sycl-business-card-raytracer.exe > sycl-business-card-raytracer.ppm` to render the image. 

<img src="sycl-business-card-raytracer.png" width="512"/>

## SYCL Mandelbrot Set
Furthermore, this repository contains two SYCL samples that both calculate the Mandelbrot - once using `Unified Shared Memory (USM)`, and once using explicit device buffers **[1]**. 

# References

* **[1]** https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2B/DenseLinearAlgebra/vector-add/src
