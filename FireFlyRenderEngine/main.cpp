#pragma once

#include "JSONLoader/JSONLoader.h"
#include "GLFWViewer/GLFWViewer.h"
#include "Film/Film.h"
#include "Scene/Geometry.h"

#include <iostream>

// Program Parameters
/*
*	-cl		: This tells the program to execute in Comand Line mode. By default the program runs in GUI mode
*	-sfp	: This parameter specifies the scene file path
*	-outrfp	: This parameter specifies the file path to save the rendered file to
*/

int main(int argc, char* argv[])
{
	// Check program arguments
	bool cl = false;
	std::string sceneFilePath = "";
	std::string outRenderFilePath = "";
		
	for(int paramIndex = 1; paramIndex < argc; paramIndex++) 
	{
		if(argv[paramIndex] == "cl") 
		{
			cl = true;
		}

		if(argv[paramIndex] == "sfp") 
		{
			sceneFilePath = argv[paramIndex + 1];
		}

		if(argv[paramIndex] == "outrfp")
		{
			outRenderFilePath = argv[paramIndex + 1];
		}
	}

	// GUI Mode
	if (cl == false) 
	{
		// sets up scene
		auto sceneLoader = std::make_unique<JSONLoader>();

		std::string projectPath = SOLUTION_DIR;
		std::string sceneFile = projectPath + R"(scenes\testscene.json)";
		sceneLoader->LoadSceneFromFile(sceneFile);
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
	// CL Mode
	else 
	{
		// TODO
	}
}