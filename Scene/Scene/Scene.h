#pragma once

#ifdef SCENE_EXPORTS
#define SCENE_API __declspec(dllexport)
#else
#define SCENE_API __declspec(dllimport)
#endif

#include <vector>
#include <memory>

#include "AccelerationStructure.h"

class Geometry;
class Material;
class Camera;


class Scene
{
public:
	SCENE_API Scene();
	SCENE_API Scene(std::vector<std::shared_ptr<Geometry>> geometries, std::vector<std::shared_ptr<Material>> materials,
		std::vector<std::shared_ptr<Camera>> cameras);
private:
	std::vector<std::shared_ptr<Geometry>> m_geometries;
	std::vector<std::shared_ptr<Material>> m_materials;
	std::vector<std::shared_ptr<Camera>> m_cameras;
	std::unique_ptr<AccelerationStructure> m_accel;
};