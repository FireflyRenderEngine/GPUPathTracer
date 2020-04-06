
#include "pch.h"
#include "JSONLoader.h"

#include "../Scene/ThinLensCamera.h"
#include "../Scene/Material.h"
#include "../Film/Film.h"
#include "../Scene/TriangleMesh.h"
#include "../Scene/Cube.h"
#include "../Scene/Plane.h"
#include "../Scene/Sphere.h"

#include <fstream>
#include <filesystem>

bool JSONLoader::LoadSceneFromFile(std::string filename)
{
	bool ret = false;
	std::filesystem::path filePath(filename);
	if (filePath.empty())
	{
		return ret;
	}
	if (filePath.extension().filename() != ".json")
	{
		return ret;
	}
	// read a JSON file
	std::ifstream fileStream(filePath);
	json jsonDocument;
	fileStream >> jsonDocument;

	json cameraList, geometryList, materialList, lightList;
	std::map<std::string, std::shared_ptr<Material>> mtlNameToMaterial;
	// check if object "frames" exists
	bool framesPresent = jsonDocument.contains("frames");
	m_scene = std::make_shared<Scene>();
	if (framesPresent)
	{
		json frames = jsonDocument["frames"];
		if (!frames.is_array())
		{
			return ret;
		}
		//check scene object for every frame
		for (auto& frame : frames)
		{
			if (!frame.is_object())
			{
				return ret;
			}
			json sceneObj = frame["scene"];
			//load cameras
			if (sceneObj.contains("cameras"))
			{
				cameraList = sceneObj["cameras"];
				for (auto& cameraObj : cameraList)
				{
					LoadCamera(cameraObj);
				}
			}
			//load all materials in map with mtl name as key and Material itself as value
			if (sceneObj.contains("materials"))
			{
				materialList = sceneObj["materials"];
				for(auto& materialObj : materialList) 
				{
					LoadMaterial(materialObj);
				}
			}
			//load primitives and attach materials from map
			if (sceneObj.contains("geometries"))
			{
				geometryList = sceneObj["geometries"];
				for(auto& primitiveObj : geometryList)
				{
					LoadGeometry(primitiveObj);
				}
			}
			//load lights and attach materials from map
			if (sceneObj.contains("lights"))
			{
				lightList = sceneObj["lights"];
				for(auto& lightObj : lightList) 
				{
					LoadLights(lightObj);
				}
			}
		}
	}
	
	ret = true;
	return ret;
}

bool JSONLoader::LoadGeometry(json& geometry)
{
	std::shared_ptr<Geometry> shape = std::make_shared<Geometry>();
	//First check what type of geometry we're supposed to load
	std::string type;
	std::string objFilePath = "";
	if (geometry.contains(std::string("type"))) 
	{
		type = geometry["type"].get<std::string>();
	}
	GeometryType geomType = GeometryType::NONE;
	if (type.compare(std::string("TriangleMesh")) == 0)
	{
		geomType = GeometryType::TRIANGLEMESH;
		
		if (geometry.contains(std::string("filename")))
		{
			std::string projectPath = SOLUTION_DIR;
			objFilePath = projectPath + geometry["filename"].get<std::string>();
		}
		//TODO: Load Materials per triangle
	}
	else if (type.compare(std::string("Sphere")) == 0)
	{
		geomType = GeometryType::SPHERE;
	}
	else if (type.compare(std::string("Plane")) == 0)
	{
		geomType = GeometryType::PLANE;
	}
	else if (type.compare(std::string("Cube")) == 0)
	{
		geomType = GeometryType::CUBE;
	}
	else
	{
		std::cout << "Could not parse the geometry!" << std::endl;
		return NULL;
	}
	// TODO: Load Materials
	std::map<std::string, std::shared_ptr<Material>>::iterator i;
	
	//load transform to shape
	if (geometry.contains(std::string("transform"))) {
		json transform = geometry["transform"];
		LoadTransform(transform, shape);
	}
	//if (geometry.contains(std::string("name"))) primitive->name = geometry["name"];

	m_scene->LoadOBJ(geomType, shape->m_geometryPosition, shape->m_geometryRotationAngleAlongAxis, shape->m_geometryScale, objFilePath);
	return true;
}

bool JSONLoader::LoadLights(json& geometry)
{
	// TODO: LoadLights
	return false;
}

namespace
{
	glm::vec3 ToVec3(std::array<double, 3> vector)
	{
		return { vector[0], vector[1], vector[2] };
	}
}

bool JSONLoader::LoadMaterial(json& material)
{
	std::string type;

	//First check what type of material we're supposed to load
	if (material.contains(std::string("type"))) type = material["type"].get<std::string>();

	if (type.compare( std::string("MatteMaterial")) == 0)
	{
		std::shared_ptr<Film> textureMap;
		std::shared_ptr<Film> normalMap;
		glm::vec3 Kd = ToVec3(material["Kd"]);
		float sigma = static_cast<float>(material["sigma"]);
		if (material.contains(std::string("textureMap"))) {
			std::string img_filepath = (material["textureMap"]);
			textureMap = std::make_shared<Film>(img_filepath);
		}
		if (material.contains(std::string("normalMap"))) {
			std::string img_filepath = (material["normalMap"]);
			normalMap = std::make_shared<Film>(img_filepath);
		}
		//auto result = std::make_shared</*Matte*/Material>(Kd, sigma, textureMap, normalMap);
		//mtl_map->insert(std::make_pair(material["name"], result));
	}
	else if (type.compare( std::string("MirrorMaterial")) == 0)
	{
		std::shared_ptr<Film> roughnessMap;
		std::shared_ptr<Film> textureMap;
		std::shared_ptr<Film> normalMap;
		glm::vec3 Kr = ToVec3(material["Kr"]);
		float roughness = 0.f;
		if (material.contains(std::string("roughness"))) {
			roughness = material["roughness"];
		}
		if (material.contains(std::string("roughnessMap"))) {
			std::string img_filepath = (material["roughnessMap"]);
			roughnessMap = std::make_shared<Film>(img_filepath);
		}
		if (material.contains(std::string("textureMap"))) {
			std::string img_filepath = (material["textureMap"]);
			textureMap = std::make_shared<Film>(img_filepath);
		}
		if (material.contains(std::string("normalMap"))) {
			std::string img_filepath = (material["normalMap"]);
			normalMap = std::make_shared<Film>(img_filepath);
		}
		//auto result = std::make_shared</*Mirror*/Material>(Kr, roughness, roughnessMap, textureMap, normalMap);
		//mtl_map->insert(std::make_pair(material["name"], result));
	}
	else if (type.compare(std::string("TransmissiveMaterial")) == 0)
	{
		std::shared_ptr<Film> roughnessMap;
		std::shared_ptr<Film> textureMap;
		std::shared_ptr<Film> normalMap;
		glm::vec3 Kt = ToVec3(material["Kt"]);
		float eta = material["eta"];
		float roughness = 0.f;
		if (material.contains(std::string("roughness"))) {
			roughness = material["roughness"];
		}
		if (material.contains(std::string("roughnessMap"))) {
			std::string img_filepath = (material["roughnessMap"]);
			roughnessMap = std::make_shared<Film>(img_filepath);
		}
		if (material.contains(std::string("textureMap"))) {
			std::string img_filepath = (material["textureMap"]);
			textureMap = std::make_shared<Film>(img_filepath);
		}
		if (material.contains(std::string("normalMap"))) {
			std::string img_filepath = (material["normalMap"]);
			normalMap = std::make_shared<Film>(img_filepath);
		}
		//auto result = std::make_shared</*Transmissive*/Material>(Kt, eta, roughness, roughnessMap, textureMap, normalMap);
		//mtl_map->insert(std::make_pair(material["name"], result));
	}
	else if (type.compare(std::string("GlassMaterial")) == 0)
	{
		std::shared_ptr<Film> textureMapRefl;
		std::shared_ptr<Film> textureMapTransmit;
		std::shared_ptr<Film> normalMap;
		glm::vec3 Kr = ToVec3(material["Kr"]);
		glm::vec3 Kt = ToVec3(material["Kt"]);
		float eta = material["eta"];
		if (material.contains(std::string("textureMapRefl"))) {
			std::string img_filepath = (material["textureMapRefl"]);
			textureMapRefl = std::make_shared<Film>(img_filepath);
		}
		if (material.contains(std::string("textureMapTransmit"))) {
			std::string img_filepath = (material["textureMapTransmit"]);
			textureMapTransmit = std::make_shared<Film>(img_filepath);
		}
		if (material.contains(std::string("normalMap"))) {
			std::string img_filepath = (material["normalMap"]);
			normalMap = std::make_shared<Film>(img_filepath);
		}
		//auto result = std::make_shared</*Glass*/Material>(Kr, Kt, eta, textureMapRefl, textureMapTransmit, normalMap);
		//mtl_map->insert(std::make_pair(material["name"], result));
	}
	else if (type.compare(std::string("PlasticMaterial")) == 0)
	{
		std::shared_ptr<Film> roughnessMap;
		std::shared_ptr<Film> textureMapDiffuse;
		std::shared_ptr<Film> textureMapSpecular;
		std::shared_ptr<Film> normalMap;
		glm::vec3 Kd = ToVec3(material["Kd"]);
		glm::vec3 Ks = ToVec3(material["Ks"]);
		float roughness = material["roughness"];
		if (material.contains(std::string("roughnessMap"))) {
			std::string img_filepath = (material["roughnessMap"]);
			roughnessMap = std::make_shared<Film>(img_filepath);
		}
		if (material.contains(std::string("textureMapDiffuse"))) {
			std::string img_filepath = (material["textureMapDiffuse"]);
			textureMapDiffuse = std::make_shared<Film>(img_filepath);
		}
		if (material.contains(std::string("textureMapSpecular"))) {
			std::string img_filepath = (material["textureMapSpecular"]);
			textureMapSpecular = std::make_shared<Film>(img_filepath);
		}
		if (material.contains(std::string("normalMap"))) {
			std::string img_filepath = (material["normalMap"]);
			normalMap = std::make_shared<Film>(img_filepath);
		}
		//auto result = std::make_shared</*Plastic*/Material>(Kd, Ks, roughness, roughnessMap, textureMapDiffuse, textureMapSpecular, normalMap);
		//mtl_map->insert(std::make_pair(material["name"], result));
	}
	else
	{
		std::cout << "Could not parse the material!" << std::endl;
		return false;
	}

	return true;
}

bool JSONLoader::LoadCamera(json& jsonCamera)
{
	glm::vec3 cameraPosition{}, cameraForward{}, worldUp{};
	float screenWidth{}, screenHeight{}, yaw{}, pitch{}, fov{}, nearClip{}, farClip{}, sensitivity{ 0.3f };

	if (jsonCamera.contains(("position"))) cameraPosition = ToVec3(jsonCamera["position"].get<std::array<double, 3>>());
	if (jsonCamera.contains(("cameraForward"))) cameraForward = ToVec3(jsonCamera["cameraForward"].get<std::array<double, 3>>());
	if (jsonCamera.contains(("worldUp"))) worldUp = ToVec3(jsonCamera["worldUp"].get<std::array<double, 3>>());
	if (jsonCamera.contains(("width"))) screenWidth = jsonCamera["width"].get<float>();
	if (jsonCamera.contains(("height"))) screenHeight = jsonCamera["height"].get<float>();
	if (jsonCamera.contains(("fov"))) fov = jsonCamera["fov"].get<float>();
	if (jsonCamera.contains(("nearClip"))) nearClip = jsonCamera["nearClip"].get<float>();
	if (jsonCamera.contains(("farClip"))) farClip = jsonCamera["farClip"].get<float>();
	if (jsonCamera.contains(("pitch"))) pitch = jsonCamera["pitch"].get<float>();
	if (jsonCamera.contains(("yaw"))) yaw = jsonCamera["yaw"].get<float>();
	if (jsonCamera.contains(("sensitivity"))) sensitivity = jsonCamera["sensitivity"].get<float>();
	
	// only the first camera from the sceneloader should be used as the raster camera
	if (m_scene->m_cameras.empty())
	{
		m_scene->SetScreenWidthAndHeight(screenWidth, screenHeight);

		m_scene->SetRasterCamera(cameraPosition, screenWidth, screenHeight, cameraForward, worldUp, yaw, pitch, fov, nearClip, farClip, sensitivity);
	}
	std::shared_ptr<Camera> camera = std::make_shared<ThinLensCamera>(cameraPosition, screenWidth, screenHeight, cameraForward, worldUp, yaw, pitch, fov, farClip, nearClip);
	m_scene->m_cameras.push_back(camera);
	return true;
}

void JSONLoader::LoadTransform(json& transform, std::shared_ptr<Geometry>& geometry)
{
	glm::vec3 t, r, s;
	s = glm::vec3(1, 1, 1);
	if (transform.contains(std::string("translate"))) t = ToVec3(transform["translate"].get<std::array<double,3>>());
	if (transform.contains(std::string("rotate"))) r = ToVec3(transform["rotate"].get<std::array<double, 3>>());
	if (transform.contains(std::string("scale"))) s = ToVec3(transform["scale"].get<std::array<double, 3>>());
	geometry->m_geometryPosition = t;
	geometry->m_geometryRotationAngleAlongAxis = r;
	geometry->m_geometryScale = s;
}
