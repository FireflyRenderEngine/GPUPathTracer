#pragma once
#include <iostream>
#include "Scene/Scene.h"
#include "GLFWViewer/GLFWViewer.h"

int main(int argc, char* argv[])
{
	// sets up scene
	std::shared_ptr<Scene> scene = std::make_shared<Scene>();
	std::string filePath = "../SceneResources/cube.obj";
	scene->LoadScene(filePath);

	// Calls renderer
	// gets film
	// outputs film
	Viewer* viewer = new GLFWViewer(1024,768,scene);
	viewer->Init();
	viewer->setupViewer();
	viewer->render();
}