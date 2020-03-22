#pragma once
#ifdef _USRDLL
#	ifdef SCENE_EXPORTS
#		define SCENE_API __declspec(dllexport)
#	else
#		define SCENE_API __declspec(dllimport)
#	endif
#else
#	define SCENE_API
#endif

#include <vector>
#include <memory>
#include <iostream>
#include "AccelerationStructure.h"
#include "vec3.hpp"

class Geometry;
class Material;
class RasterCamera;
class Camera;

class SCENE_API Scene
{
public:
	Scene();
	Scene(std::vector<std::shared_ptr<Geometry>> geometries, std::vector<int> emitterGeometryIndices, std::vector<std::shared_ptr<Material>> materials, std::vector<std::shared_ptr<Camera>> cameras, float screenWidth, float screenHeight);
	void SetScreenWidthAndHeight(float screenWidth, float screenHeight);
	void UpdateScreenWidthAndHeight(float screenWidth, float screenHeight);
	float GetScreenWidth();
	float GetScreenHeight();

	void LoadOBJ(std::string filePath);
	void SetRasterCamera(glm::vec3 cameraPosition);

	std::vector<std::shared_ptr<Geometry>> m_geometries;
	std::vector<int> m_emmitterGeometryIndices;
	std::vector<std::shared_ptr<Material>> m_materials;
	std::vector<std::shared_ptr<Camera>> m_cameras;
	std::shared_ptr<RasterCamera> m_rasterCamera;
	std::unique_ptr<AccelerationStructure> m_accel;
	float m_screenWidth, m_screenHeight;
};