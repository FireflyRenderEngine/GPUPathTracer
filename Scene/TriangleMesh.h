#pragma once
#include <vector>
#include "Geometry.h"

class TriangleMesh : public Geometry{
public:

	TriangleMesh(GeometryType geometryType, glm::vec3 position, glm::vec3 rotationAlongAxis, glm::vec3 scale, std::vector<glm::vec3> vertices, std::vector<glm::vec3> normals, std::vector<glm::vec2> uvs, std::vector<int> triangleIndices)
		: Geometry(geometryType, position, rotationAlongAxis, scale, vertices, normals, uvs, triangleIndices)
	{}
};