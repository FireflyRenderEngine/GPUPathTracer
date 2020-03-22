#include "Scene.h"
#include "Geometry.h"
#include "Camera.h"
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
        m_cameras[cameraIndex]->UpdateCameraScreenWidthAndHeight(m_screenWidth, m_screenHeight);
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

void Scene::LoadScene(std::string fllePath) {
    // Load & insert the geometries 
    tinyobj::attrib_t geometryAttributes;
    std::vector<tinyobj::shape_t> geometries;

    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&geometryAttributes, &geometries, nullptr, &warn, &err, fllePath.c_str());

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

                glm::vec3 vertexPosition(geometryAttributes.vertices[3.0f * idx.vertex_index + 0], geometryAttributes.vertices[3.0f * idx.vertex_index + 1], geometryAttributes.vertices[3.0f * idx.vertex_index + 2]);
                glm::vec3 vertexNormals(geometryAttributes.normals[3.0f * idx.normal_index + 0], geometryAttributes.normals[3.0f * idx.normal_index + 1], geometryAttributes.normals[3.0f * idx.normal_index + 2]);
                glm::vec2 vertexUVs(geometryAttributes.texcoords[2.0f * idx.texcoord_index + 0], geometryAttributes.texcoords[2.0f * idx.texcoord_index + 1]);

                vertices.push_back(vertexPosition);
                normals.push_back(vertexNormals);
                uvs.push_back(vertexUVs);
                triangleIndices.push_back(index_offset + attributeIndex);
            }
            
            index_offset += numberOfVerticesPerFace;
        }

        // Initialze the geometry
        // Initialize geometry's transforms
        glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f);
        glm::vec3 rotation = glm::vec3(0.0f, 0.0f, 0.0f);
        glm::vec3 scale = glm::vec3(1.0f, 1.0f, 1.0f);
        std::shared_ptr<Geometry> geometry = std::make_shared<Geometry>(vertices, normals, uvs, triangleIndices, position, rotation, scale);
        m_geometries.push_back(geometry);

        // TODO: Load Material
    }

    // Initialize the camers
    glm::vec3 cameraPosition = glm::vec3(0.0f, 5.0f, 10.0f);
    std::shared_ptr<Camera> camera = std::make_shared<Camera>(cameraPosition, m_screenWidth, m_screenHeight);
    m_cameras.push_back(camera);

    // TODO: LOAD MATERIALS AND OTHER STUFF
}