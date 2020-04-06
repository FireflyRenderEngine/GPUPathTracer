#pragma once

#include "Camera.h"

class ThinLensCamera : public Camera
{
public:
	ThinLensCamera() = default;
	ThinLensCamera(glm::vec3 cameraPosition, float screenWidth, float screenHeight, glm::vec3 cameraForward = { 0.f, 0.f, -1.f }, glm::vec3 worldUp = { 0.0f, 1.0f, 0.0f }, float yaw = -90.0f, float pitch = 0.0f, float fov = 70, float nearClip = 0.1f, float farClip = 1000.0f)
		: Camera(cameraPosition, screenWidth, screenHeight, cameraForward, worldUp, yaw, pitch, fov, nearClip, farClip)
	{
	}
	virtual ~ThinLensCamera()
	{
	}
private:
};