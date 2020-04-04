
#include "pch.h"
#include "JSONLoader.h"

#include "../Scene/Camera.h"
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
	if (filePath.extension().filename() != "json")
	{
		return ret;
	}
	// read a JSON file
	std::ifstream fileStream(filePath.filename());
	json jsonDocument;
	fileStream >> jsonDocument;

	json cameraList, primitiveList, materialList, lightList;
	std::map<std::string, std::shared_ptr<Material>> mtlNameToMaterial;
	// check if object "frames" exists
	bool framesPresent = jsonDocument.contains("frames");

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
			//load cameras
			if (frame.contains("cameras"))
			{
				cameraList = frame["cameras"];
				for (auto& cameraObj : cameraList)
				{
					LoadCamera(cameraObj, &m_scene->m_cameras);
				}
			}
			//load all materials in map with mtl name as key and Material itself as value
			if (frame.contains("materials")) 
			{
				materialList = frame["materials"];
				for(auto& materialObj : materialList) 
				{
					LoadMaterial(materialObj, &mtlNameToMaterial);
				}
			}
			//load primitives and attach materials from map
			if (frame.contains("geometries")) 
			{
				primitiveList = frame["geometries"];
				for(auto& primitiveObj : primitiveList)
				{
					LoadGeometry(primitiveObj, mtlNameToMaterial, &m_scene->m_geometries);
				}
			}
			//load lights and attach materials from map
			if (frame.contains("lights")) 
			{
				lightList = frame["lights"];
				for(auto& lightObj : lightList) 
				{
					LoadLights(lightObj, mtlNameToMaterial, &m_scene->m_geometries, &m_scene->m_emitterGeometryIndices);
				}
			}
		}
	}

	ret = true;
	return ret;
}

bool JSONLoader::LoadGeometry(json& geometry, std::map<std::string, std::shared_ptr<Material>> mtl_map, std::vector<std::shared_ptr<Geometry>>* Geometrys)
{
	std::shared_ptr<Geometry> shape = std::make_shared<Geometry>();
	//First check what type of geometry we're supposed to load
	std::string type;
	if (geometry.contains(std::string("geometry"))) 
	{
		type = geometry["geometry"].get<std::string>();
	}

	bool isMesh = false;
	if (type.compare(std::string("Mesh")) == 0)
	{
		auto mesh = std::make_shared<TriangleMesh>();
		isMesh = true;
		glm::vec3 pos{}, rotationAlongAxis{}, scale{};
		
		if (geometry.contains(std::string("transform"))) {
			json qTransform = geometry["transform"];
			LoadTransform(qTransform, shape);
		}
		
		if (geometry.contains(std::string("filename"))) {
			std::string objFilePath = geometry["filename"];
			m_scene->LoadOBJ(GeometryType::TRIANGLEMESH, shape->m_geometryPosition, shape->m_geometryRotationAngleAlongAxis, shape->m_geometryScale, objFilePath);
		}

		//std::string meshName("Unnamed Mesh");
		//if (geometry.contains(std::string("name"))) meshName = geometry["name"].get<std::string>();
		//meshName.append(std::string("'s Triangle"));
		//for (auto triangle : mesh->faces)
		//{
		//	auto primitive = std::make_shared<Primitive>(triangle);
		//	QMap<std::string, std::shared_ptr<Material>>::iterator i;
		//	if (geometry.contains(std::string("material"))) {
		//		std::string material_name = geometry["material"];
		//		for (i = mtl_map.begin(); i != mtl_map.end(); ++i) {
		//			if (i.key() == material_name) {
		//				primitive->material = i.value();
		//			}
		//		}
		//	}
		//	primitive->name = meshName;
		//	(*primitives).append(primitive);
		//}
	}
	else if (type.compare(std::string("Sphere")) == 0)
	{
		shape = std::make_shared<Sphere>();
	}
	else if (type.compare(std::string("SquarePlane")) == 0)
	{
		shape = std::make_shared<Plane>();
	}
	else if (type.compare(std::string("Cube")) == 0)
	{
		shape = std::make_shared<Cube>();
	}
	else
	{
		std::cout << "Could not parse the geometry!" << std::endl;
		return NULL;
	}




	if (!isMesh)
	{
		// The Mesh class is handled differently
		// All Triangles are added to the Primitives list
		// but a single Drawable is created to render the Mesh
		std::map<std::string, std::shared_ptr<Material>>::iterator i;
		//if (geometry.contains(std::string("material"))) {
		//	std::string material_name = geometry["material"];
		//	for (i = mtl_map.begin(); i != mtl_map.end(); ++i) {
		//		if (i.key() == material_name) {
		//			primitive->material = i.value();
		//		}
		//	}
		//}
		//load transform to shape
		if (geometry.contains(std::string("transform"))) {
			json transform = geometry["transform"];
			LoadTransform(transform, shape);
		}
		//if (geometry.contains(std::string("name"))) primitive->name = geometry["name"];
	}
	return true;
}

bool JSONLoader::LoadLights(json& geometry, std::map<std::string, std::shared_ptr<Material>> mtl_map, std::vector<std::shared_ptr<Geometry>>* Geometrys, std::vector<int>* lights)
{
	return false;
}

namespace
{
	glm::vec3 ToVec3(std::array<double, 3> vector)
	{
		return { vector[0], vector[1], vector[2] };
	}
}

bool JSONLoader::LoadMaterial(json& material, std::map<std::string, std::shared_ptr<Material>>* mtl_map)
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

bool JSONLoader::LoadCamera(json& jsonCamera, std::vector<std::shared_ptr<Camera>>* Cameras)
{
	std::shared_ptr<Camera> camera = std::make_shared<Camera>();
	//if (jsonCamera.contains(("target"))) camera->ref = ToVec3(jsonCamera["target"]);
	//if (jsonCamera.contains(("eye"))) camera->eye = ToVec3(jsonCamera["eye"]);
	//if (jsonCamera.contains(("worldUp"))) camera->world_up = ToVec3(jsonCamera["worldUp"]);
	//if (jsonCamera.contains(("width"))) camera->width = jsonCamera["width"];
	//if (jsonCamera.contains(("height"))) camera->height = jsonCamera["height"];
	//if (jsonCamera.contains(("fov"))) camera->fovy = jsonCamera["fov"];
	//if (jsonCamera.contains(("nearClip"))) camera->near_clip = jsonCamera["nearClip"];
	//if (jsonCamera.contains(("farClip"))) camera->far_clip = jsonCamera["farClip"];
	//
	//camera->RecomputeAttributes();
	Cameras->push_back(camera);
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
