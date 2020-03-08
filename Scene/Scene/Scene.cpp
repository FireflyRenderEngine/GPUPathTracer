#include "Scene.h"
#include "Geometry.h"
#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "../tinyobjloader-2.0.0/tiny_obj_loader.h"

Scene::Scene()
{
}

Scene::Scene(std::vector<std::shared_ptr<Geometry>> geometries, std::vector<int> emitterGeometryIndices, std::vector<std::shared_ptr<Material>> materials, std::vector<std::shared_ptr<Camera>> cameras)
    :m_geometries(geometries),
    m_emmitterGeometryIndices(emitterGeometryIndices),
    m_materials(materials),
    m_cameras(cameras)
{
    m_accel = std::make_unique<AccelerationStructure>();
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

        // Initialze the geometry
        std::shared_ptr<Geometry> geometry = std::make_shared<Geometry>();

        for (size_t triangleIndex = 0; triangleIndex < geometries[geometryIndex].mesh.num_face_vertices.size(); triangleIndex++) {

            std::vector<glm::vec3> vertices;
            std::vector<glm::vec3> normals;
            std::vector<glm::vec2> uvs;
            // Loop over the triangle attributes.
            for (size_t attributeIndex = 0; attributeIndex < 3; attributeIndex++) {

                // access to vertex
                tinyobj::index_t idx = geometries[geometryIndex].mesh.indices[index_offset + attributeIndex];

                vertices.push_back(glm::vec3(geometryAttributes.vertices[3.0f * idx.vertex_index + 0], geometryAttributes.vertices[3.0f * idx.vertex_index + 1], geometryAttributes.vertices[3.0f * idx.vertex_index + 2]));
                normals.push_back(glm::vec3(geometryAttributes.normals[3.0f * idx.normal_index + 0], geometryAttributes.normals[3.0f * idx.normal_index + 1], geometryAttributes.normals[3.0f * idx.normal_index + 2]));
                uvs.push_back(glm::vec2(geometryAttributes.texcoords[2.0f * idx.texcoord_index + 0], geometryAttributes.texcoords[2.0f * idx.texcoord_index + 1]));
            }
            index_offset += 3;
            std::shared_ptr<Triangle> triangle = std::make_shared<Triangle>();
            triangle->InsertTriangleVertices(vertices[0], vertices[1], vertices[2]);
            triangle->InsertTriangleNormals(normals[0], normals[1], normals[2]);
            triangle->InsertTriangleUVs(uvs[0], uvs[1], uvs[2]);
            geometry->InsertTriangle(triangle);
        }
    }


    // Initialize the materials

    // Initialize the camers
}