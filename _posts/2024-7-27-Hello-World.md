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

We can pass pass parameters to kernel just like any other c program and we can additinally pass threads and block to it we will touch on it in further chapters. In order to run program in GPU we need to first allocate memory over there and then fill that empty memory with our data which is currently presnet in CPU and after that perform computation. Once this is done we have to also recieve the result back and once one free that allocated memory. All of the handling of memory is done in CPU part of the code ie host. Like C has malloc cuda has cudaMalloc(). 

- You *can* pass pointer allocated with cudaMalloc() to function that execute on the device.
- You *can* pass pointer allocated with cudaMalloc() to function that execute on the host.
- You *can* pass pointers allocated with cudaMalloc() to read or write memory from the code that executes on the host
- You *cant* pass pointers allocated with cudaMalloc() to read or write memory from the code that executes on the device

For copying we have cudaMemCpy() we need to mention the size and the directions as args. 

```c
#include <iostream>
#include "book.h"
__global__ void add( int a, int b, int *c ) {
	*c = a + b;
}
int main( void ) {
  int c;
  int *dev_c;
  HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );
  add<<<1,1>>>( 2, 7, dev_c );
  HANDLE_ERROR( cudaMemcpy( &c,
    dev_c,
    sizeof(int),
    cudaMemcpyDeviceToHost ) );
  printf( "2 + 7 = %d\n", c );
  cudaFree( dev_c );
  return 0;
}
```

## Querying Device

Many machines have more than one device so we need to select which one we want to use for these we have cudaGetDeviceCount() function. It gives us the number of devices present in our machine. And we can get the properties of this device from cudaGetDeviceProperties() function. Just like in Vulkan and win32 api here there is a big struct which shall be send as a refrence to this function and it will get populated with correct values. 

```c
#include "common/book.h"

int main( void ) 
{
    cudaDeviceProp prop;
    int count;
    HANDLE_ERROR( cudaGetDeviceCount( &count ) );
    for (int i=0; i< count; i++)
    {
        HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) );
        printf( " --- General Information for device %d ---\n", i );
        printf( "Name: %s\n", prop.name );
        printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
        printf( "Clock rate: %d\n", prop.clockRate );
        printf( "Device copy overlap: " );
        if (prop.deviceOverlap)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );

        printf( "Kernel execition timeout : " );

        if (prop.kernelExecTimeoutEnabled)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );

        printf( " --- Memory Information for device %d ---\n", i );
        printf( "Total global mem: %ld\n", prop.totalGlobalMem );
        printf( "Total constant Mem: %ld\n", prop.totalConstMem );
        printf( "Max mem pitch: %ld\n", prop.memPitch );
        printf( "Texture Alignment: %ld\n", prop.textureAlignment );
        printf( " --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count: %d\n",
        prop.multiProcessorCount );
        printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
        printf( "Registers per mp: %d\n", prop.regsPerBlock );
        printf( "Threads in warp: %d\n", prop.warpSize );
        printf( "Max threads per block: %d\n",
        prop.maxThreadsPerBlock );
        printf( "Max thread dimensions: (%d, %d, %d)\n",
        prop.maxThreadsDim[0], prop.maxThreadsDim[1],
        prop.maxThreadsDim[2] );
        printf( "Max grid dimensions: (%d, %d, %d)\n",
        prop.maxGridSize[0], prop.maxGridSize[1],
        prop.maxGridSize[2] );
        printf( "\n" );
    }
}
```

My output was: [output](https://i.ibb.co/ZT8Tzw7/Screenshot-2024-07-27-233749.png)

## Using device properties

Now that we can query them its quite annoying to loop through each property and find the relevent device which suites our needs, so cuda runtime has a feature. We have to fill the member variable with the property we desire and cuda get device will return the device id which is best as per the given struct. 

```c
#include "../common/book.h"
int main( void ) 
{
  cudaDeviceProp prop;
  int dev;
  HANDLE_ERROR( cudaGetDevice( &dev ) );
  printf( "ID of current CUDA device: %d\n", dev );
  memset( &prop, 0, sizeof( cudaDeviceProp ) );
  prop.major = 1;
  prop.minor = 3;
  HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );
  printf( "ID of CUDA device closest to revision 1.3: %d\n", dev );
  HANDLE_ERROR( cudaSetDevice( dev ) );
  return(0);
}
```

You can find the code [here](https://github.com/yashcgosavi/cuda)
