#pragma once

#include "JSONLoader/JSONLoader.h"
#include "GLFWViewer/GLFWViewer.h"
#include "Film/Film.h"
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
	std::shared_ptr<Film> film = std::make_shared<Film>(scene->GetScreenWidth(), scene->GetScreenHeight());
	// outputs film
	auto viewer = std::make_unique<GLFWViewer>(scene, film);
	viewer->Init();
	viewer->setupViewer();
	viewer->render();
}