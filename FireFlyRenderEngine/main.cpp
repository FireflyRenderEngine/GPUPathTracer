#pragma once
#include <iostream>

#include "../GLFWViewer/GLFWViewer.h"

int main(int argc, char* argv[])
{
	// sets up scene
	// Calls renderer
	// gets film
	// outputs film
	Viewer* viewer = new GLFWViewer(1024,768);
	viewer->Init();
	viewer->setupViewer();
	viewer->render();
}