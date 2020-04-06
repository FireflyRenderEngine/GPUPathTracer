#pragma once
#include "Camera.h"

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

class SCENE_API RasterCamera : public Camera
{
public:
	RasterCamera() = default;

	RasterCamera(glm::vec3 cameraPosition, float screenWidth, float screenHeight, glm::vec3 cameraForward = glm::vec3( 0.f, 0.f, -1.f ), glm::vec3 worldUp = glm::vec3( 0.0f, 1.0f, 0.0f ), float yaw = -90.0f, float pitch = 0.0f, float fov = 70, float nearClip = 0.1f, float farClip = 1000.0f, float sensitivity = 0.3f)
		: Camera(cameraPosition, screenWidth, screenHeight, cameraForward, worldUp, yaw, pitch, fov, nearClip, farClip ), m_cameraMouseSensitivity(sensitivity)
	{
		m_cameraFirstMouseInput = false;
		m_cameraMovementSpeed = 1.f;
	}

	virtual ~RasterCamera() override
	{
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

	
private:
    // Camera options
	float m_cameraMovementSpeed{ 0.3f };
	float m_cameraMouseSensitivity{ 0.3f };
	bool m_cameraFirstMouseInput{ false };
	float m_xDelta{0.f}, m_yDelta{0.f};
};