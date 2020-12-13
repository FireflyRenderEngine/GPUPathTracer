# CUDA-OpenGL Interop 

## What is Interop?
Interop stands for InterOPerability. In this project we display our graphics using OpenGL and write our kernels using CUDA.
Interop is the capability of a product or system to interact and function with others.

## Is it required?
Not at all! Interop between CUDA and OpenGL is not required but in fact necessary to avoid inessential resource transfers.
With that said, could you create an OpenGL application with CUDA without using its interop capabilities? Yes, you can. And that's how we started with this project.

## What does it do?
A general interoperability scenario includes transferring data between the CUDA device(s) and OpenGL on a regular basis (often per frame) and in both directions.  For example:

- Physics simulation in CUDA, producing the vertex data to be rendered with OpenGL
- Generating a frame with OpenGL, with further image post-processing applied in a CUDA kernel (e.g. HDR tone-mapping)
- Procedural (noise) generation in CUDA, followed by using the results as OpenGL texture in the rendering pipeline
- Physically based rendering in CUDA kernel and rendering the frame in OpenGL


Since both these APIs work for the most part on the device (GPU), we might want to use that to our advantage.
Memory transfers are the biggest bottleneck in an optimized kernel call architecture.
The amount of data being transferred from device-to-host and again from host-to-device every frame causes a huge slowdown.

For example, in our path tracer, when we started out and just wanted to get a proof-of-concept done, we had the CUDA kernel transfer its array of pixel colors back to the host and then again upload this array to the device through OpenGL to be rendered.
This is absolutely viable but not what we were looking for in terms of interactive graphics view of our physically based lighting solution.

Interop solves this by avoiding the extra transfer that was needed from CUDA's device to host and OpenGL's host to device by:

Creating a common buffer that both CUDA kernel and the OpenGL APIs can access.

And the difference as expected was noticable.
We were avoiding numerous unnecessary transfers between host and device.

## How do we do interop?
There are several CUDA-GL interop examples scattered over the web with niche and specific applications in mind. The most applicable we found for us was: [Allan MacKinnon's](https://gist.github.com/allanmac): [A tiny example of CUDA + OpenGL interop with write-only surfaces and CUDA kernels.](https://gist.github.com/allanmac/4ff11985c3562830989f)
This code is pretty self-explanatory once you know the basics and a few glossary terms. 

### Let's go over some of these:
- GL Buffers
- cudaArray
- Surface Memory
- cudaGraphicsResource
- CUDA streams
- GL Renderbuffer
- GL Framebuffer
- Double buffering or Swapchains
- cudaGraphicsMapResources/cudaGraphicsUnmapResources
- Framebuffer blit
- Pixel format

### What do we need to make interop happen?


