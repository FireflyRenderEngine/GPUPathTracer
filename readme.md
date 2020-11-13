# Fire Fly Engine   

## Road to developement
1. _OpenGL, Cuda & C++ architecture_
2. _Stripping it all down!_
3. _Getting Cuda play nice with C++_
4. _On the road to victory!_

## OpenGL, Cuda & C++ architecture

Welcome everyone to our jorney to writing a GPU based path tracer. This is not our first time writing a path tracer. We have, in our masters degree written, debugged and implemented couple of path / ray tracers on the CPU and also on the GPU. However, it was always based on some previous code provided to us by the professors that had already made many arhitectural decisions and had only left a couple of ways to implent new functionality. While this approach worked with helping us learn the basics of estimating the light transport equation it wa still limiting as we were only filling in the blanks.


Fire Fly engine was our first attempt to make our own path tracer from scratch! so naturally we jumped at the opportunity and were pretty high on spirits and confidence going in with delusions of grandeure. We both agreed that we needed a modular architecture just in case we need to swap out one module with a newer / different implementation later on (lol). We created multiple dll's, one that handled scene loading and processing, another that handled the openGL viewer and another that handled the path tracer and CUDA stuff. We were proud of our architecture and started working with cuda to load the kernel with data little id we know what was waiting for us. According to Murphey's law, everything that could go wrong will, and just like that there were run time errors everywhere. The kernels did not run as expected, data was not being sent correctly and even the basic kernels did not work. After a lot of researching we slowly realized that our great C++ classes complete with inheritance, smart pointer and the like were completely useless and not suported by CUDA. 

The thing is we were so inclined on creating the best modular architecture that we forgot to check if the data structures and the way we were storing them were even compatible with CUDA. And so our house of card came crashing down.

## Stripping it all down!

With our architecture and classes usless, and our illusions of grandiour fading fast we did what many programmers have done in their moment of desperation, when they just can't figure out what's wrong and there are waaay to many working parts. We stripped everything down to the basics

We removed the dll's and all the seperate projects and incorporated everything inside one main project. We converted from smart pointers and inherited classes to structs and raw pointers as containers. And we got rid of all complicated kernels for just one kernel to test if we could send one buffer of pixels to the kernel, color the pixels with a random color, return the buffer and use that as  a texture to paint a quad in our openGL viewer.

It took some figring out but finally we got it to work and we were pretty proud of ourselves.

