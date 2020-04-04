#pragma once

#include "../SceneLoader/SceneLoader.h"

#include "json.hpp"

#include <map>
#include <memory>
#include <vector>

using json = nlohmann::json;

class Material;
class Geometry;
class Camera;
class Transform;
class Light;

class JSONLoader : public SceneLoader
{
public:
	virtual bool LoadSceneFromFile(std::string filename) override;
private:
    bool LoadGeometry(json& geometry, std::map<std::string, std::shared_ptr<Material>> mtl_map, std::vector<std::shared_ptr<Geometry>>* Geometrys);
    bool LoadLights(json& geometry, std::map<std::string, std::shared_ptr<Material>> mtl_map, std::vector<std::shared_ptr<Geometry>>* Geometrys, std::vector<int>* lights);
    bool LoadMaterial(json& material, std::map<std::string, std::shared_ptr<Material> >* mtl_map);
    bool LoadCamera(json& jsonCamera, std::vector<std::shared_ptr<Camera>>* Cameras);
    void LoadTransform(json& transform, std::shared_ptr<Geometry>& geometry);
};