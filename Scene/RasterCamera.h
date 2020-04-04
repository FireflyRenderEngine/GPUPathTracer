#pragma once
#include "vec3.hpp"
#include "glm.hpp"
#include <gtc/matrix_transform.hpp>

// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
enum Camera_Movement 
{
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
	UP,
	DOWN,
	YAWLEFT,
	YAWRIGHT,
	PITCHUP,
	PITCHDOWN
};

class RasterCamera
{
public:
	RasterCamera() = default;

	RasterCamera(glm::vec3 cameraPosition, float screenWidth, float screenHeight, glm::vec3 worldUp = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = -90.0f, float pitch = 0.0f, float fov = 70, float cearClip = 0.1f, float farClip = 1000.0f, float sensitivity = 0.1f)
		: m_cameraPosition(cameraPosition), m_screenWidth(screenWidth), m_screenHeight(screenHeight), m_worldUp(worldUp), m_cameraYaw(yaw), m_cameraPitch(pitch), m_fov(fov), m_nearClip(cearClip), m_farClip(farClip), m_cameraMouseSensitivity(sensitivity)
	{
		m_cameraForward = glm::vec3(0.0f, 0.0f, -1.0f);
		m_cameraFirstMouseInput = false;
		UpdateBasisAxis();
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

	void SetFirstMouseInputState(bool state) 
	{
		m_cameraFirstMouseInput = state;
	}

	bool GetFirstMouseInputState() 
	{
		return m_cameraFirstMouseInput;
	}

	// Processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
	void ProcessKeyboard(Camera_Movement direction)
	{
		float velocity = m_cameraMouseSensitivity;
		if (direction == FORWARD)
			m_cameraPosition += m_cameraForward * velocity;
		if (direction == BACKWARD)
			m_cameraPosition -= m_cameraForward * velocity;
		if (direction == LEFT)
			m_cameraPosition -= m_cameraRight * velocity;
		if (direction == RIGHT)
			m_cameraPosition += m_cameraRight * velocity;
		if (direction == UP)
			m_cameraPosition += m_cameraUp * velocity;
		if (direction == DOWN)
			m_cameraPosition -= m_cameraUp * velocity;
		if (direction == YAWLEFT) 
		{
			m_cameraYaw -= 1.0f;
			UpdateBasisAxis();
		}
		if (direction == YAWRIGHT)
		{
			m_cameraYaw += 1.0f;
			UpdateBasisAxis();
		}
		if (direction == PITCHUP) 
		{
			m_cameraPitch += 1.0f;
			UpdateBasisAxis();
		}
		if (direction == PITCHDOWN) 
		{
			m_cameraPitch -= 1.0f;
			UpdateBasisAxis();
		}
	}

	// Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
	void ProcessMouseMovement(bool constrainPitch = true)
	{
		m_xDelta *= m_cameraMouseSensitivity;
		m_yDelta *= m_cameraMouseSensitivity;

		m_cameraYaw += m_xDelta;
		m_cameraPitch += m_yDelta;

		// Make sure that when pitch is out of bounds, screen doesn't get flipped
		if (constrainPitch)
		{
			if (m_cameraPitch > 89.0f)
				m_cameraPitch = 89.0f;
			if (m_cameraPitch < -89.0f)
				m_cameraPitch = -89.0f;
		}

		// Update Front, Right and Up Vectors using the updated Euler angles
		UpdateBasisAxis();
	}

	// Calculates the front vector from the Camera's (updated) Euler Angles
	void UpdateBasisAxis()
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
	//---------------------------------------------------------------------

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
	glm::vec3 m_cameraUp;
	glm::vec3 m_cameraRight;
	glm::vec3 m_cameraForward;
    glm::vec3 m_worldUp;

    float m_cameraYaw;
    float m_cameraPitch;

    // Camera options
    float m_cameraMovementSpeed;
    float m_cameraMouseSensitivity;
    float m_cameraZoom;
	bool m_cameraFirstMouseInput;

	float m_screenWidth;
	float m_screenHeight;
	float m_fov;
	float m_nearClip;
	float m_farClip;
	float m_xDelta, m_yDelta;
};