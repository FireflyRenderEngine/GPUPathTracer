#include "Scene.h"

Scene::Scene()
{
}

Scene::Scene(std::vector<std::shared_ptr<Geometry>> geometries, std::vector<std::shared_ptr<Material>> materials, std::vector<std::shared_ptr<Camera>> cameras)
	:m_geometries(geometries),
	 m_materials(materials),
	 m_cameras(cameras)
{
	m_accel = std::make_unique<AccelerationStructure>();
}