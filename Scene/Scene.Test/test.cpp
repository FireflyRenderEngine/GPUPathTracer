#include "pch.h"
#include "../Scene.h"
#include "../glm-0.9.9.7/vec3.hpp"
#include "../Geometry.h"

TEST(TestingScene, Constructor) 
{
    std::shared_ptr<Scene> scene = std::make_shared<Scene>();
    ASSERT_TRUE(scene);
}

TEST(OBJLoadingTest, GeometryTest) {
	std::shared_ptr<Scene> scene = std::make_shared<Scene>();
	// LOAD TRIANGLE MESH
	glm::vec3 position = glm::vec3(-5.0f, 5.0f, 0.0f);
	glm::vec3 rotation = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 scale = glm::vec3(1.0f, 1.0f, 1.0f);
	std::string projectPath = SOLUTION_DIR;
	std::string filePathTriangleMesh = projectPath + R"(SceneResources\cubeTriangleMesh.obj)";
	ASSERT_TRUE(scene->LoadOBJ(GeometryType::TRIANGLEMESH, position, rotation, scale, filePathTriangleMesh));
	// LOAD CUBE
	glm::vec3 cubePosition = glm::vec3(-2.0f, 5.0f, 0.0f);
	glm::vec3 cubeRotation = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 cubeScale = glm::vec3(1.0f, 1.0f, 1.0f);
	ASSERT_TRUE(scene->LoadOBJ(GeometryType::CUBE, cubePosition, cubeRotation, cubeScale));
	// LOAD PLANE
	glm::vec3 planePosition = glm::vec3(0.0f, 5.0f, 0.0f);
	glm::vec3 planeRotation = glm::vec3(45.0f, 0.0f, 0.0f);
	glm::vec3 planeScale = glm::vec3(1.0f, 1.0f, 1.0f);
	ASSERT_TRUE(scene->LoadOBJ(GeometryType::PLANE, planePosition, planeRotation, planeScale));
	// LOAD SPHERE
	glm::vec3 spherePosition = glm::vec3(2.0f, 5.0f, 0.0f);
	glm::vec3 sphereRotation = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 sphereScale = glm::vec3(1.0f, 1.0f, 1.0f);
	ASSERT_TRUE(scene->LoadOBJ(GeometryType::SPHERE, spherePosition, sphereRotation, sphereScale));
}