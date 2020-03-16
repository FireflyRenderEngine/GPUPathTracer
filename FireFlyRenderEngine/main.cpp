#pragma once
#include <iostream>
#include "Scene/Scene.h"
#include "GLFWViewer/GLFWViewer.h"

int main(int argc, char* argv[])
{
	float screenWidth = 1024;
	float screenHeight = 768;

	// sets up scene
	std::shared_ptr<Scene> scene = std::make_shared<Scene>();
	std::string filePath = R"(C:\Users\rudra\Documents\Projects\FireflyRenderEngine\GPUPathTracer\SceneResources\cube.obj)";
	scene->SetScreenWidthAndHeight(screenWidth, screenHeight);
	scene->LoadScene(filePath);

	// Calls renderer
	// gets film
	// outputs film
	Viewer* viewer = new GLFWViewer(scene);
	viewer->Init();
	viewer->setupViewer();
	viewer->render();
}