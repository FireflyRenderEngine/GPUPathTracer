#include "Scene.h"
#include "Geometry.h"
#include "TriangleMesh.h"
#include "Plane.h"
#include "Cube.h"
#include "Sphere.h"
#include "RasterCamera.h"
#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tiny_obj_loader.h"

Scene::Scene()
{
}

Scene::Scene(std::vector<std::shared_ptr<Geometry>> geometries, std::vector<int> emitterGeometryIndices, std::vector<std::shared_ptr<Material>> materials, std::vector<std::shared_ptr<Camera>> cameras, float screenWidth, float screenHeight)
    :m_geometries(geometries),
    m_emmitterGeometryIndices(emitterGeometryIndices),
    m_materials(materials),
    m_cameras(cameras)
{
    m_accel = std::make_unique<AccelerationStructure>();
    SetScreenWidthAndHeight(screenWidth, screenHeight);
}

void Scene::SetScreenWidthAndHeight(float screenWidth, float screenHeight) {
    m_screenWidth = screenWidth;
    m_screenHeight = screenHeight;
}

void Scene::UpdateScreenWidthAndHeight(float screenWidth, float screenHeight)
{
    m_screenWidth = screenWidth;
    m_screenHeight = screenHeight;
    for (int cameraIndex = 0; cameraIndex < m_cameras.size(); cameraIndex++) {
        m_rasterCamera->UpdateCameraScreenWidthAndHeight(m_screenWidth, m_screenHeight);
    }   
}

float Scene::GetScreenWidth()
{
    return m_screenWidth;
}

float Scene::GetScreenHeight()
{
    return m_screenHeight;
}

bool Scene::LoadOBJ(GeometryType geometryType, glm::vec3 position, glm::vec3 rotationAlongAxis, glm::vec3 scale, std::string filePath) {
    // Handle predefined geometry file paths
    std::string projectPath = SOLUTION_DIR;
    switch (geometryType)
    {
        case GeometryType::SPHERE :
            filePath = projectPath + R"(SceneResources\sphere.obj)";
            break;
        case GeometryType::CUBE:
            filePath = projectPath + R"(SceneResources\cube.obj)";
            break;
        case GeometryType::PLANE:
            filePath = projectPath + R"(SceneResources\plane.obj)";
            break;
        default:
            // If the geometry type is not among the supported type or is a triangle mesh and the file path is empty return false
            if (geometryType != GeometryType::TRIANGLEMESH || filePath == "") {
                return false;
            }
            break;
    }
    
    // Load & insert the geometries 
    tinyobj::attrib_t geometryAttributes;
    std::vector<tinyobj::shape_t> geometries;

    std::string warn;
    std::string err;

    if (tinyobj::LoadObj(&geometryAttributes, &geometries, nullptr, &warn, &err, filePath.c_str()) == false) {
        return false;
    }

    // NOTE: We currently only support triangle geometries
    // Loop over Geometries
    for (size_t geometryIndex = 0; geometryIndex < geometries.size(); geometryIndex++) {

        // Loop over triangles / Geometry
        size_t index_offset = 0;

        // These set of attributes will hold the parameters for each geometry
        std::vector<glm::vec3> vertices;
        std::vector<glm::vec3> normals;
        std::vector<glm::vec2> uvs;
        std::vector<int> triangleIndices;

        for (size_t triangleIndex = 0; triangleIndex < geometries[geometryIndex].mesh.num_face_vertices.size(); triangleIndex++) {
            int numberOfVerticesPerFace = geometries[geometryIndex].mesh.num_face_vertices[triangleIndex];

            // Loop over the triangle attributes.
            for (size_t attributeIndex = 0; attributeIndex < numberOfVerticesPerFace; attributeIndex++) {

                // access to vertex
                tinyobj::index_t idx = geometries[geometryIndex].mesh.indices[index_offset + attributeIndex];

                glm::vec3 vertexPosition(geometryAttributes.vertices[3 * idx.vertex_index + 0], geometryAttributes.vertices[3 * idx.vertex_index + 1], geometryAttributes.vertices[3 * idx.vertex_index + 2]);
                glm::vec3 vertexNormals(geometryAttributes.normals[3 * idx.normal_index + 0], geometryAttributes.normals[3 * idx.normal_index + 1], geometryAttributes.normals[3 * idx.normal_index + 2]);
                glm::vec2 vertexUVs(geometryAttributes.texcoords[2 * idx.texcoord_index + 0], geometryAttributes.texcoords[2 * idx.texcoord_index + 1]);

                vertices.push_back(vertexPosition);
                normals.push_back(vertexNormals);
                uvs.push_back(vertexUVs);
                triangleIndices.push_back(index_offset + attributeIndex);
            }
            
            index_offset += numberOfVerticesPerFace;
        }

        // Initialze the geometry
        std::shared_ptr<Geometry> geometry;
        switch (geometryType)
        {
        case GeometryType::SPHERE:
            geometry = std::make_shared<Sphere>(geometryType, position, rotationAlongAxis, scale, vertices, normals, uvs, triangleIndices);
            break;
        case GeometryType::CUBE:
            geometry = std::make_shared<Cube>(geometryType, position, rotationAlongAxis, scale, vertices, normals, uvs, triangleIndices);
            break;
        case GeometryType::PLANE:
            geometry = std::make_shared<Plane>(geometryType, position, rotationAlongAxis, scale, vertices, normals, uvs, triangleIndices);
            break;
        case GeometryType::TRIANGLEMESH:
            geometry = std::make_shared<TriangleMesh>(geometryType, position, rotationAlongAxis, scale, vertices, normals, uvs, triangleIndices);
            break;
        default:
            break;
        }
        m_geometries.push_back(geometry);


        // TODO: Load Material
    }

    // TODO: LOAD MATERIALS AND OTHER STUFF

    return true;
}

void Scene::SetRasterCamera(glm::vec3 cameraPosition) {
    m_rasterCamera = std::make_shared<RasterCamera>(cameraPosition, m_screenWidth, m_screenHeight);
}