# Fire Fly Engine   

## Road to developement
1. _OpenGL, Cuda & C++ architecture_
2. _Stripping it all down!_
3. _Getting CUDA to play nice with C++_
4. _Onward and Upward!_

## OpenGL, Cuda & C++ architecture

Welcome everyone to our jorney to writing a GPU based path tracer. This is not our first time writing a path tracer. We have, in our masters degree written, debugged and implemented couple of path / ray tracers on the CPU and also on the GPU. However, it was always based on some previous code provided to us by the professors that had already made many arhitectural decisions and had only left a couple of ways to implent new functionality. While this approach worked with helping us learn the basics of estimating the light transport equation it wa still limiting as we were only filling in the blanks.

Fire Fly engine was our first attempt to make our own path tracer from scratch! so naturally we jumped at the opportunity and were pretty high on spirits and confidence going in with delusions of grandeure. We both agreed that we needed a modular architecture just in case we need to swap out one module with a newer / different implementation later on (lol). We created multiple dll's, one that handled scene loading and processing, another that handled the openGL viewer and another that handled the path tracer and CUDA stuff. We were proud of our architecture and started working with cuda to load the kernel with data little id we know what was waiting for us. According to Murphey's law, everything that could go wrong will, and just like that there were run time errors everywhere. The kernels did not run as expected, data was not being sent correctly and even the basic kernels did not work. After a lot of researching we slowly realized that our great C++ classes complete with inheritance, smart pointer and the like were completely useless and not suported by CUDA. 

The thing is we were so inclined on creating the best modular architecture that we forgot to check if the data structures and the way we were storing them were even compatible with CUDA. And so our house of card came crashing down.

## Stripping it all down!

With our architecture and classes usless, and our illusions of grandiour fading fast we did what many programmers have done in their moment of desperation, when they just can't figure out what's wrong and there are waaay to many working parts. We stripped everything down to the basics

We removed the dll's and all the seperate projects and incorporated everything inside one main project. We converted from smart pointers and inherited classes to structs and raw pointers as containers. And we got rid of all complicated kernels for just one kernel to test if we could send one buffer of pixels to the kernel, color the pixels with a random color, return the buffer and use that as  a texture to paint a quad in our openGL viewer.

It took some figuring out but finally we got it to work and we were pretty proud of ourselves.

## Getting CUDA to play nice with C++

Now we neither claim or know everything about CUDA or C++. But we know that even though they have their difference they do cliam to be freinds. So we wanted to test the claim. Forewarning it may get a little technical now but if you are reading this we assume you have a little programming knowledge. Now, we had already moved to structs from classes and raw pointers from smart pointers so if we assign a memory block on the device (GPU) memory the size of the struct we will be able to populate it with data for geometry and meshes for our future rays to intersect, right?!

Well... yeah.. sort of. So its right that the memory block will be able to contain the data stored inside the struct locally but it will not be able to store the data pointed to by pointers inside the struct. It seems obvious now but, what we needed was to assign a memory block on the device (GPU) memory for the pointed objects inside the base struct.

Once we did that, we had multiple geometry meshes being sent to the CUDA kernels.

## Onward and Upward!

Now that we have the geometries being sent to the GPU. The next step was to generate rays for each pixel, find intersection with the meshes and shade these pixels. 

A path tracer basically estimates the light coming into the camera from all the light sources after bouncing on all the objects in the scene. We need to estimate this phenemenon using the Light Transport Equation. This equation is an integral over each point of intersection for each ray being spawnned from each pixel. To estimate for the integration we bounce the rays from each pixel multiple times recursively over the objects in the scene and add the contribution from each bounce. 

As a way of implementing this we though of two ways:

1. Wavefront method
2. Mega Kernel method.

In the first method what we would loop through multple bounces. And for each bounce we will generate the rays send them to the kernel to find intersection. Then get back the intersections and use that to generate new rays from those itersect points and contine till we reach the end of bounces and use the accumulated color to shade the meshes.

In the second method which we have decided to go with for now, we have one mega kernel that is run for the same amount of bounces and for each boucne we generate rays for each pixel, find intersection, accumulate the color and shade.

## Bloopers : Not all roads lead to Rome

### I can see right through you!

