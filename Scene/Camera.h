#pragma once
#include <vec3.hpp>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>

class Camera
{
public:
	Camera() {}

	Camera(glm::vec3 cameraPosition, glm::vec3 cameraLookAtPosition, float screenWidth, float screenHeight, glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f), float fov = 70, float cearClip = 0.1f, float farClip = 10.0f, float sensitivity = 0.01f)
		: m_cameraPosition(cameraPosition), m_cameraLookAtPosition(cameraLookAtPosition), m_screenWidth(screenWidth), m_screenHeight(screenHeight), m_worldUp(worldUp), m_fov(fov), m_nearClip(cearClip), m_farClip(farClip), m_sensitivity(sensitivity)
	{
		m_cameraForward = glm::normalize(m_cameraLookAtPosition - m_cameraPosition);
		m_cameraRight = glm::normalize(glm::cross(m_cameraForward, m_worldUp));
		m_cameraUp = glm::normalize(glm::cross(m_cameraRight, m_cameraForward));
	}

	// Camera Movement Functions
	void SetClickPosition(float xPos, float yPos) 
	{
		m_xDelta = xPos;
		m_yDelta = yPos;
	}

	void SetClickPositionDeta(float xPos, float yPos) 
	{
		m_xDelta -= xPos;
		m_yDelta -= yPos;
	}

	void ZoomInOut(float zDelta) 
	{
		m_cameraPosition = m_cameraPosition + zDelta * m_cameraForward;
	}

	void YawPitchCamera() {
		glm::mat4 rotateAlongUp(1.0f);
		glm::mat4 rotateAlongRight(1.0f);
		rotateAlongUp *= glm::rotate(glm::mat4(1.0f), glm::radians(m_xDelta), m_cameraUp);
		rotateAlongRight *= glm::rotate(glm::mat4(1.0f), glm::radians(m_yDelta), m_cameraRight);
	
		m_cameraForward = glm::normalize(glm::vec4(m_cameraForward, 0.0f) * rotateAlongUp * rotateAlongRight);
		UpdateBasisAxis();
	}

	void UpdateBasisAxis() 
	{
		m_cameraRight = glm::normalize(glm::cross(m_cameraForward, m_worldUp));
		m_cameraUp = glm::normalize(glm::cross(m_cameraRight, m_cameraForward));
	}

	// View Projection Matrix
	glm::mat4 GetViewMatrix() 
	{
		return glm::lookAtRH(m_cameraPosition, m_cameraPosition + m_cameraForward, m_cameraUp);
	}

	glm::mat4 GetProjectionMatrix() 
	{
		return glm::perspectiveFovRH(glm::radians(m_fov), m_screenWidth, m_screenHeight, m_nearClip, m_farClip);
	}

	void UpdateCameraScreenWidthAndHeight(float screenWidth, float screenHeight) {
		m_screenWidth = screenWidth;
		m_screenHeight = screenHeight;
	}
private:
	glm::vec3 m_cameraPosition;
	glm::vec3 m_cameraLookAtPosition;
	glm::vec3 m_cameraUp;
	glm::vec3 m_worldUp;
	glm::vec3 m_cameraRight;
	glm::vec3 m_cameraForward;
	float m_screenWidth;
	float m_screenHeight;
	float m_fov;
	float m_nearClip;
	float m_farClip;
	float m_xDelta, m_yDelta;
	float m_sensitivity;
};