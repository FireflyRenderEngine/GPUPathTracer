#pragma once
#include "vec3.hpp"
#include "vec2.hpp"
#include "mat4x4.hpp"
#include "gtc/matrix_transform.hpp"
#include <vector>
#include <memory>

enum GeometryType {TRIANGLEMESH, SPHERE, CUBE, RECTANGLE};

class Geometry
{
public:
	Geometry() {}

	Geometry(GeometryType geometryType, glm::vec3 position, glm::vec3 rotationAlongAxis, glm::vec3 scale)
		:  m_geometryType(geometryType), m_geometryPosition(position), m_geometryRotationAngleAlongAxis(rotationAlongAxis), m_geometryScale(scale) 
	{
		SetModelMatrix();
	}

	glm::mat4 GetMeshModelMatrix()
	{
		return m_modelMatrix;
	}

	GeometryType GetGeomtryType() 
	{
		return m_geometryType;
	}

	void SetModelMatrix() {
		// Translate
		glm::mat4 translate(1.0f);
		translate *= glm::translate(glm::mat4(1.0f), m_geometryPosition);

		// Rotate
		glm::mat4 rotate(1.0f);
		rotate *= glm::rotate(glm::mat4(1.0f), m_geometryRotationAngleAlongAxis.x, glm::vec3(1.0f, 0.0f, 0.0f));
		rotate *= glm::rotate(glm::mat4(1.0f), m_geometryRotationAngleAlongAxis.y, glm::vec3(0.0f, 1.0f, 0.0f));
		rotate *= glm::rotate(glm::mat4(1.0f), m_geometryRotationAngleAlongAxis.z, glm::vec3(0.0f, 0.0f, 1.0f));

		// Sacle
		glm::mat4 scale(1.0f);
		scale *= glm::scale(glm::mat4(1.0f), m_geometryScale);

		// Combine the transformations
		m_modelMatrix = glm::mat4(1.0f);
		m_modelMatrix *= translate * rotate * scale;
	}


	glm::mat4 m_modelMatrix;
	glm::vec3 m_geometryPosition;
	glm::vec3 m_geometryRotationAngleAlongAxis;
	glm::vec3 m_geometryScale;
	GeometryType m_geometryType;
};

