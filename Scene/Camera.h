#pragma once

#ifdef _USRDLL
#	ifdef SCENE_EXPORTS
#		define SCENE_API __declspec(dllexport)
#	else
#		define SCENE_API __declspec(dllimport)
#	endif
#else
#	define SCENE_API
#endif

#include "vec3.hpp"
#include "glm.hpp"
#include <gtc/matrix_transform.hpp>

class SCENE_API Camera
{
public:
	Camera() = default;

	Camera(glm::vec3 cameraPosition, float screenWidth, float screenHeight, glm::vec3 cameraForward = glm::vec3( 0.f, 0.f, -1.f ), glm::vec3 worldUp = glm::vec3( 0.0f, 1.0f, 0.0f ), float yaw = -90.0f, float pitch = 0.0f, float fov = 70, float nearClip = 0.1f, float farClip = 1000.0f)
	{
		m_cameraPosition = cameraPosition;
		m_cameraForward = cameraForward;
		m_screenWidth = screenWidth;
		m_screenHeight = screenHeight;
		m_screenHeight = screenHeight;
		m_worldUp = worldUp;
		m_cameraYaw = yaw;
		m_cameraPitch = pitch;
		m_fov = fov;
		m_nearClip = nearClip; 
		m_farClip = farClip;
		UpdateBasisAxis();
	}
	virtual ~Camera() {}

	virtual void UpdateCameraScreenWidthAndHeight(float screenWidth, float screenHeight) 
	{
		m_screenWidth = screenWidth;
		m_screenHeight = screenHeight;
	}

	// View Projection Matrix
	virtual glm::mat4 GetViewMatrix()
	{
		return glm::lookAtRH(m_cameraPosition, m_cameraPosition + m_cameraForward, m_cameraUp);
	}

	virtual glm::mat4 GetProjectionMatrix()
	{
		return glm::perspectiveFovRH(glm::radians(m_fov), m_screenWidth, m_screenHeight, m_nearClip, m_farClip);
	}

		// Calculates the front vector from the Camera's (updated) Euler Angles
	virtual void UpdateBasisAxis()
	{
		// Calculate the new Front vector
		glm::vec3 front;
		front.x = cos(glm::radians(m_cameraYaw)) * cos(glm::radians(m_cameraPitch));
		front.y = sin(glm::radians(m_cameraPitch));
		front.z = sin(glm::radians(m_cameraYaw)) * cos(glm::radians(m_cameraPitch));
		m_cameraForward = glm::normalize(front);
		// Also re-calculate the Right and Up vector
		m_cameraRight = glm::normalize(glm::cross(m_cameraForward, m_worldUp));  // Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
		m_cameraUp = glm::normalize(glm::cross(m_cameraRight, m_cameraForward));
	}
protected:
	glm::vec3 m_cameraPosition;
	glm::vec3 m_cameraUp;
	glm::vec3 m_cameraRight;
	glm::vec3 m_cameraForward;
	glm::vec3 m_worldUp;

	float m_cameraYaw;
	float m_cameraPitch;

	float m_screenWidth;
	float m_screenHeight;
	float m_fov;
	float m_nearClip;
	float m_farClip;
};