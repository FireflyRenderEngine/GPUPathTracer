#pragma once
#include <vector>
#include <memory>

#include "AccelerationStructure.h"

class Geometry;
class Material;
class Camera;


class Scene
{
public:
	Scene();
	Scene(std::vector<std::shared_ptr<Geometry>> geometries, std::vector<std::shared_ptr<Material>> materials,
		std::vector<std::shared_ptr<Camera>> cameras);
private:
	std::vector<std::shared_ptr<Geometry>> m_geometries;
	std::vector<std::shared_ptr<Material>> m_materials;
	std::vector<std::shared_ptr<Camera>> m_cameras;
	std::unique_ptr<AccelerationStructure> m_accel;
};