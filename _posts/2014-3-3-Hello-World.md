---
layout: post
title: 'CUDA: A first program'
---

**Reference Book:** CUDA by Example

## Introduction
- The development of high compute power machines has always been aimed at solving real-world problems effectively.
- With the exponential decrease in transistor size, more powerful processors with multiple cores were developed. However, increasing clock speeds resulted in higher power consumption and heat generation, making it infeasible to continually make processors more powerful by this method.
- Supercomputers have been using parallel computing for decades to meet high computational demands. This trend extended to PCs with the introduction of multicore processors.

## Rise of GPUs
- In the late 1980s, the rise of graphics interfaces and Windows applications led to the addition of graphics acceleration hardware in PCs to enhance user experience.
- Silicon Graphics started producing 3D hardware and introduced OpenGL as a universal API to interact with the hardware.
- The popularity of 3D graphic games further increased the demand for GPUs, which was accelerated by the availability of more affordable GPUs.

## Challenges with Early GPUs
- Researchers wanted to utilize the parallel computing power of GPUs for non-graphics applications. However, GPUs were originally designed for real-time rendering and graphics pipelines.
- Initially, computation was done using vertex shaders, which required knowledge of OpenGL, DirectX, and shading languages like GLSL and HLSL. This was challenging due to the lack of debugging tools and the inability to directly allocate memory.

## Introduction of CUDA
- CUDA (Compute Unified Device Architecture) changed this by providing a unified shader pipeline, allowing every ALU on the chip to be used for general-purpose calculations.
- CUDA's instruction set supports floating-point numbers and general computation, not just graphics tasks.
- The GPU execution units can arbitrarily read and write memory and access a software-managed cache known as shared memory, making GPUs excel at both computation and traditional graphics tasks.

## CUDA Programming
- CUDA is a modified version of C. Similar to how MSVC is used for C/C++, CUDA uses `nvcc`, which is part of the CUDA toolkit provided by NVIDIA.
- In CUDA programming, the code is divided into host and device code. The host (CPU) processes tasks in a serial manner, while the device (GPU) processes tasks in parallel.
- CUDA is supported by only certain GPUs.

### Example of a Simple CUDA Program
```c
#include "../common/book.h"

int main(void) {
    printf("Hello, World!\n");
    return 0;
}

```

- This is a perfectly valid CUDA program. When given to nvcc, it is compiled into an object file.

## Heterogeneous Parallel Programming (HPP)
CUDA-style programming is also called HPP (Heterogeneous Parallel Programming) because we program both the CPU and the GPU.
The CPU code is just like normal C programming, while the GPU code is qualified by __global__ and __device__. When calling the function, we use the syntax <<<block, thread>>>.
Example of a Kernel Function
```cuda
#include <iostream>

__global__ void kernel(void) {
}

int main(void) {
    kernel<<<1,1>>>();
    printf("Hello, World!\n");
    return 0;
}
```

A function that runs on the GPU is called a kernel.
