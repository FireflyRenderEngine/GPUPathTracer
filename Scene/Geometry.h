#pragma once
#include "vec3.hpp"
#include "vec2.hpp"
#include <vector>
#include <memory>

class Geometry
{
public:
	Geometry() {}

	Geometry(std::vector<glm::vec3> vertices, std::vector<glm::vec3> normals, std::vector<glm::vec2> uvs, std::vector<int> triangleIndices) 
		: m_vertices(vertices), m_normals(normals), m_uvs(uvs), m_triangleIndices(triangleIndices) {}

private:
	std::vector<glm::vec3> m_vertices;
	std::vector<glm::vec2> m_uvs;
	std::vector<glm::vec3> m_normals;
	std::vector<int> m_triangleIndices;
};