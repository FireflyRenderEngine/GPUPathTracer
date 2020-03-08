#pragma once
#include "vec3.hpp"
#include "vec2.hpp"
#include <vector>
#include <memory>


struct Triangle {
	std::vector<glm::vec3> m_vertices;
	std::vector<glm::vec2> m_uvs;
	std::vector<glm::vec3> m_normals;

	void InsertTriangleVertices(glm::vec3 vertex1, glm::vec3 vertex2, glm::vec3 vertex3) {
		m_vertices.push_back(vertex1);
		m_vertices.push_back(vertex2);
		m_vertices.push_back(vertex3);
	}

	void InsertTriangleUVs(glm::vec2 uv1, glm::vec2 uv2, glm::vec2 uv3) {
		m_uvs.push_back(uv1);
		m_uvs.push_back(uv2);
		m_uvs.push_back(uv3);
	}

	void InsertTriangleNormals(glm::vec3 vertex1, glm::vec3 vertex2, glm::vec3 vertex3) {
		m_normals.push_back(vertex1);
		m_normals.push_back(vertex2);
		m_normals.push_back(vertex3);
	}
};

class Geometry
{
public:
	Geometry() : m_triangles(NULL) {}

	void InsertTriangle(std::shared_ptr<Triangle> newTriangle) {
		m_triangles.push_back(newTriangle);
	}

private:
	std::vector<std::shared_ptr<Triangle>> m_triangles;
};