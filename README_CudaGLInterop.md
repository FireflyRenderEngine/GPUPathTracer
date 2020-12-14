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
- GL Buffers:   
	OpenGL stores its data in its buffer object names as GLuint pointers.
	Various types of buffers can be used for its various purposes, for example, we use vertex buffers to store vertex data information to send to the device, we use pixel buffers to render pixels to the buffer which can then be used to display these pixels on the screen. There are several key differences between these types of buffers and its intended use. For example, glBindBuffer uses the following types of buffers to be bound for access.
	
	| Buffer Binding Target 	| Purpose |
	| ------------------------------ | -------- |
	| GL_ARRAY_BUFFER |	Vertex attributes |
	| GL_ATOMIC_COUNTER_BUFFER |	Atomic counter storage |
	| GL_COPY_READ_BUFFER |	Buffer copy source 
	| GL_COPY_WRITE_BUFFER |	Buffer copy destination |
	| GL_DISPATCH_INDIRECT_BUFFER |	Indirect compute dispatch commands |
	| GL_DRAW_INDIRECT_BUFFER |	Indirect command arguments |
	| GL_ELEMENT_ARRAY_BUFFER |	Vertex array indices |
	| GL_PIXEL_PACK_BUFFER |	Pixel read target |
	| GL_PIXEL_UNPACK_BUFFER |	Texture data source |
	| GL_QUERY_BUFFER |	Query result buffer |
	| GL_SHADER_STORAGE_BUFFER |	Read-write storage for shaders |
	| GL_TEXTURE_BUFFER |	Texture data buffer |
	| GL_TRANSFORM_FEEDBACK_BUFFER |	Transform feedback buffer |
	| GL_UNIFORM_BUFFER |	Uniform block storage |

	- GL Renderbuffer: Renderbuffer Objects are OpenGL Objects that contain images. They are created and used specifically with Framebuffer Objects. They are optimized for use as render targets. Which means that we will use render buffers as the buffer objects to write our cuda kernel operation to and display the renderbuffer as is. Renderbuffer objects natively accomodate multisampling (MSAA).
	- GL Framebuffer: Framebuffer objects are a collection of attachments. A renderbuffer is an object that contains a single image. Renderbuffers cannot be accessed by Shaders in any way. The only way to work with a renderbuffer, besides creating it, is to put it into an FBO. And that's what we'll be doing here.
	Read the khronos FBO wiki for more in-depth information [here](https://www.khronos.org/opengl/wiki/Framebuffer_Object).
		
- cudaArray: CUDA arrays are similar like an array in CUDA but specially they are memory areas dedicate to textures (and surfaces). They are read-only for the GPU (and are only accessible through texture/surface fetches), and can be written to by the CPU using cudaMemcpyToArray(). cudaArray is an opaque block of memory that is optimized for binding to textures. Textures can use memory stored in a space filling curve, which allows for a better texture cache hit rate due to better 2D spatial locality. This is where we will write our pixel information from CUDA kernels. This CUDA array will then be transferred to a GL buffer as a renderbuffer to be displayed onto the framebuffer.
- Surface Memory: 
	> CUDA supports a subset of the texturing hardware that the GPU uses for graphics to access texture and surface memory. The texture and surface memory spaces reside in device memory and are cached in texture cache, so a texture fetch or surface read costs one memory read from device memory only on a cache miss, otherwise it just costs one read from texture cache. The texture cache is optimized for 2D spatial locality, so threads of the same warp that read texture or surface addresses that are close together in 2D will achieve best performance. Also, it is designed for streaming fetches with a constant latency; a cache hit reduces DRAM bandwidth demand but not fetch latency.

	-- [CUDA C Programming Guide](https://www.cs.colby.edu/courses/S18/cs336/online_materials/CUDA_C_Programming_Guide.pdf)
	
- cudaGraphicsResource: A resource must be registered to CUDA before it can be mapped using the functions mentioned in OpenGL Interoperability like cudaGraphicsMapResources among others. cudaGraphicsResource is a struct that holds information about the mapped resources.

- CUDA streams: A stream is a sequence of commands (possibly issued by different host threads) that execute in order. Different streams, on the other hand, may execute their commands out of order with respect to one another or concurrently.
Lei Mao's [blog](https://leimao.github.io/blog/CUDA-Stream/) explains this well with the following figure:
[![](https://leimao.github.io/images/blog/2020-02-02-CUDA-Stream/cuda-stream.png)](https://leimao.github.io/blog/CUDA-Stream/)
For more information, read Nvidia's blog ["GPU Pro Tip: CUDA 7 Streams Simplify Concurrency"](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/) here.

- Double buffering or Swapchains: This is a technique commonly used to increase overall throughput of the device and thus preventing bottlenecks. The implementation is even simpler than the concept. Think of letting the device write to one buffer while the other buffer is being drawn so that both the drawing buffer and the working (pathtracing) buffer are busy. So at every frame, we choose to draw one of the 2 renderbuffers to the framebuffer while the other one is being filled by the next iteration. The cost of this would be the memory that needs to be allocated for 2 buffers instead of one. 

- cudaGraphicsMapResources/cudaGraphicsUnmapResources: Before our kernel call at each frame, we map the cuda arrays and its resources for our device to read/write from. After kernel execution returns, we unmap these resources for the GL API to pick up the array and map it to its render/frame buffers. This is not supposed to be expensive at all even though NSight Compute profiler pointed us to these (cudaGraphicsMapResources/cudaGraphicsUnmapResources) calls. The issue was that we had too many unnecessary matrix multiplications at every kernel launch, but more on that later.
 
- Framebuffer blit: This is frequently used with a double buffering system with at least 2 framebuffers. glBlitFrameBuffer copies a block of pixels from one framebuffer object to another. 

### What do we need to make interop happen? And how do we tie all these aforementioned concepts together?

We create 2 framebuffers, 2 renderbuffers, 2 `cudaArray`s and 2 `cudaGraphicsResource`s to begin with. The size of these 2d buffers match the width and height of our window to be displayed:

```cpp
// allocate arrays
interop->fb = (GLuint*)calloc(fbo_count, sizeof(*(interop->fb)));
interop->rb = (GLuint*)calloc(fbo_count, sizeof(*(interop->rb)));
interop->cgr = (cudaGraphicsResource_t*)calloc(fbo_count, sizeof(*(interop->cgr)));
interop->ca = (cudaArray_t*)calloc(fbo_count, sizeof(*(interop->ca)));

// render buffer object w/a color buffer
glCreateRenderbuffers(fbo_count, interop->rb);

// frame buffer object
glCreateFramebuffers(fbo_count, interop->fb);

// attach rbo to fbo
for (int index = 0; index < fbo_count; index++)
{
	glNamedFramebufferRenderbuffer(interop->fb[index],
		GL_COLOR_ATTACHMENT0,
		GL_RENDERBUFFER,
		interop->rb[index]);
}
```
where interop object stores all the required arrays and resources. `fb` is the framebuffer object, `rb` is the renderbuffer object, `cgr` is the cudaGraphicsResource and `ca` is the cudaArray pointer. `fbo_count` here is set to 2 since we are using double buffering. 
The above code was taken from [Allan MacKinnon's](https://gist.github.com/allanmac): [A tiny example of CUDA + OpenGL interop with write-only surfaces and CUDA kernels.](https://gist.github.com/allanmac/4ff11985c3562830989f).

