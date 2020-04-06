#pragma once

#include "JSONLoader/JSONLoader.h"
#include "GLFWViewer/GLFWViewer.h"
#include "Scene/Geometry.h"

#include <iostream>

int main(int argc, char* argv[])
{

	// sets up scene
	auto sceneLoader = std::make_unique<JSONLoader>();
	sceneLoader->LoadSceneFromFile("../scenes/testscene.json");
	std::shared_ptr<Scene> scene = sceneLoader->getScene();

	// Calls renderer
	// gets film
	// outputs film
	auto viewer = std::make_unique<GLFWViewer>(scene);
	viewer->Init();
	viewer->setupViewer();
	viewer->render();
}