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
    bool LoadGeometry(json& geometry);
    bool LoadLights(json& geometry);
    bool LoadMaterial(json& material);
    bool LoadCamera(json& jsonCamera);
    void LoadTransform(json& transform, std::shared_ptr<Geometry>& geometry);
};