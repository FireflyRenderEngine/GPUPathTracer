#pragma once
#include <iostream>
#include "Scene/Scene.h"
#include "GLFWViewer/GLFWViewer.h"
#include "../glm-0.9.9.7/vec3.hpp"
#include <Scene/Geometry.h>

int main(int argc, char* argv[])
{
	float screenWidth = 1024;
	float screenHeight = 768;

	// sets up scene
	std::shared_ptr<Scene> scene = std::make_shared<Scene>();
	scene->SetScreenWidthAndHeight(screenWidth, screenHeight);
	
	// TEST
	// LOAD TRIANGLE MESH
	glm::vec3 position = glm::vec3(-3.0f, 5.0f, 0.0f);
	glm::vec3 rotation = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 scale = glm::vec3(1.0f, 1.0f, 1.0f);
	std::string filePathTriangleMesh = R"(..\SceneResources\cubeTriangleMesh.obj)";
	if (scene->LoadOBJ(GeometryType::TRIANGLEMESH, position, rotation, scale, filePathTriangleMesh)) {
		std::cout << "ERROR: Could not load Geometry. Please check the geometry type or if the file exists at the path provided." << std::endl;
	}
	// LOAD CUBE
	glm::vec3 cubePosition = glm::vec3(-2.0f, 5.0f, 0.0f);
	glm::vec3 cubeRotation = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 cubeScale = glm::vec3(1.0f, 1.0f, 1.0f);
	if (scene->LoadOBJ(GeometryType::CUBE, cubePosition, cubeRotation, cubeScale)) {
		std::cout << "ERROR: Could not load Geometry. Please check the geometry type or if the file exists at the path provided." << std::endl;
	}
	// LOAD PLANE
	glm::vec3 planePosition = glm::vec3(-1.0f, 5.0f, 0.0f);
	glm::vec3 planeRotation = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 planeScale = glm::vec3(1.0f, 1.0f, 1.0f);
	if (scene->LoadOBJ(GeometryType::PLANE, planePosition, planeRotation, planeScale)) {
		std::cout << "ERROR: Could not load Geometry. Please check the geometry type or if the file exists at the path provided." << std::endl;
	}
	// LOAD SPHERE
	glm::vec3 spherePosition = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 sphereRotation = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 sphereScale = glm::vec3(1.0f, 1.0f, 1.0f);
	if (scene->LoadOBJ(GeometryType::SPHERE, spherePosition, sphereRotation, sphereScale)) {
		std::cout << "ERROR: Could not load Geometry. Please check the geometry type or if the file exists at the path provided." << std::endl;
	}
	// END TEST
	
	
	scene->SetRasterCamera(glm::vec3(0.0f, 5.0f, 10.0f));



	// Calls renderer
	// gets film
	// outputs film
	Viewer* viewer = new GLFWViewer(scene);
	viewer->Init();
	viewer->setupViewer();
	viewer->render();
}