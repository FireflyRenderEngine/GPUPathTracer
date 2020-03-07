#pragma once
#include <iostream>

#include "../GLFWViewer/GLFWViewer.h"

int main(int argc, char* argv[])
{
	// sets up scene
	// Calls renderer
	// gets film
	// outputs film
	GLFWViewer* viewer = new GLFWViewer();
	viewer->Init();
	viewer->setupViewer();
	viewer->render();
}