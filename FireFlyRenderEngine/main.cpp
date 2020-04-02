#pragma once
#include <iostream>
#include "Scene/Scene.h"
#include "GLFWViewer/GLFWViewer.h"
#include "vec3.hpp"
#include <Scene/Geometry.h>

int main(int argc, char* argv[])
{
	float screenWidth = 1024;
	float screenHeight = 768;

	// sets up scene
	std::shared_ptr<Scene> scene = std::make_shared<Scene>();
	scene->SetScreenWidthAndHeight(screenWidth, screenHeight);
	scene->SetRasterCamera(glm::vec3(0.0f, 5.0f, 10.0f));

	// Calls renderer
	// gets film
	// outputs film
	Viewer* viewer = new GLFWViewer(scene);
	viewer->Init();
	viewer->setupViewer();
	viewer->render();
}