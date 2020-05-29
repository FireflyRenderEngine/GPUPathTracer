
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_types.h"
#include <stdio.h>
#include "vec3.hpp"
#include "glm.hpp"
#include <gtc/matrix_transform.hpp>
#include <iostream>
#include <fstream>

struct Triangle
{
	Triangle() = default;
	Triangle(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2)
		:m_v0(v0), m_v1(v1), m_v2(v2)
	{
	}
	glm::vec3 m_v0;
	glm::vec3 m_v1;
	glm::vec3 m_v2;
};

struct Ray
{
	glm::vec3 m_origin;
	glm::vec3 m_direction;
	float m_t{ 0.f };
};

struct Camera
{
	glm::vec3 m_position;
	glm::vec3 m_up;
	glm::vec3 m_right;
	glm::vec3 m_forward;

	glm::vec3 m_worldUp;

	float m_yaw;
	float m_pitch;

	float m_screenWidth;
	float m_screenHeight;
	float m_fov;
	float m_nearClip;
	float m_farClip;

	void UpdateCameraScreenWidthAndHeight(float screenWidth, float screenHeight)
	{
		m_screenWidth = screenWidth;
		m_screenHeight = screenHeight;
	}

	__device__ glm::mat4 GetViewMatrix()
	{
		return glm::lookAtRH(m_position, m_position + m_forward, m_up);
	}

	__device__ glm::mat4 GetInverseViewMatrix()
	{
		return glm::inverse(GetViewMatrix());
	}

	__device__ glm::mat4 GetProjectionMatrix()
	{
		return glm::perspectiveFovRH(glm::radians(m_fov), m_screenWidth, m_screenHeight, m_nearClip, m_farClip);
	}

	__device__ glm::mat4 GetInverseProjectionMatrix()
	{
		return glm::inverse(GetProjectionMatrix());
	}

	// Calculates the front vector from the Camera's (updated) Euler Angles
	void UpdateBasisAxis()
	{
		// Calculate the new Front vector
		glm::vec3 front;
		front.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
		front.y = sin(glm::radians(m_pitch));
		front.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
		m_forward = glm::normalize(front);
		// Also re-calculate the Right and Up vector
		m_right = glm::normalize(glm::cross(m_forward, m_worldUp));  // Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
		m_up = glm::normalize(glm::cross(m_right, m_forward));
	}
};

void saveToPPM(Ray* rays, int height, int width)
{
	std::ofstream renderFile;
	renderFile.open("render.ppm");

	renderFile << "P3" << std::endl;
	renderFile << width << " " << height << std::endl;
	renderFile << 255 << std::endl;

	for (int i = 0; i < width * height; ++i)
	{
		renderFile << static_cast<int>(rays[i].m_t * 255) << " " << static_cast<int>(rays[i].m_t * 255) << " " << static_cast<int>(rays[i].m_t * 255) << std::endl;
	}
	renderFile.close();
}

__global__ void generateRays(Ray* rays, Camera* camera)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int pixelSize = camera->m_screenHeight * camera->m_screenWidth;
	int pixelIndex = y * camera->m_screenWidth + x;

	if (pixelIndex >= pixelSize)
	{
		return;
	}
	Ray& ray = rays[pixelIndex];
	ray.m_origin = camera->m_position;

	float Px = (x / camera->m_screenWidth) * 2.f - 1.f;
	float Py = 1.f - (y / camera->m_screenHeight) * 2.f;

	glm::vec3 wLookAtPoint = camera->GetInverseViewMatrix() * camera->GetInverseProjectionMatrix() * (glm::vec4(Px, Py, 1.f, 1.f) * camera->m_farClip);

	ray.m_direction = glm::normalize(wLookAtPoint - ray.m_origin);
}

__device__ bool intersectTriangle(const Triangle& triangle, Ray& ray)
{
	const float EPSILON = 0.0000001;
	glm::vec3 vertex0 = triangle.m_v0;
	glm::vec3 vertex1 = triangle.m_v1;
	glm::vec3 vertex2 = triangle.m_v2;
	glm::vec3 edge1, edge2, h, s, q;
	float a, f, u, v;
	edge1 = vertex1 - vertex0;
	edge2 = vertex2 - vertex0;
	h = glm::cross(ray.m_direction, edge2);
	a = glm::dot(edge1, h);
	if (a > -EPSILON && a < EPSILON)
	{
		return false;    // This ray is parallel to this triangle.
	}
	f = 1.0 / a;
	s = ray.m_origin - vertex0;
	u = f * glm::dot(s, h);
	if (u < 0.0 || u > 1.0)
	{
		//ray.m_t = 0.4f;
		return false;
	}
	q = glm::cross(s, edge1);
	v = f * glm::dot(ray.m_direction, q);
	if (v < 0.0 || u + v > 1.0)
		return false;
	// At this stage we can compute t to find out where the intersection point is on the line.
	float t = f * glm::dot(edge2, q);
	if (t > EPSILON) // ray intersection
	{
		glm::vec3 intersectionPoint = ray.m_origin + ray.m_direction * t;
		ray.m_t = t;
		
		return true;
	}
	else // This means that there is a line intersection but not a ray intersection.
	{
		return false;
	}
}

__global__ void intersectRays(Camera* camera, Ray* rays, Triangle* triangle)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int pixelSize = camera->m_screenWidth * camera->m_screenHeight;
	int pixelIndex = y * camera->m_screenWidth + x;

	if (pixelIndex >= pixelSize)
	{
		return;
	}

	intersectTriangle(*triangle, rays[pixelIndex]);
}

int main()
{
	Triangle* t1 = new Triangle
	(
		glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3(1.f, 1.f, 0.0f),
		glm::vec3(2.f, 0.0f, 0.0f)
	);

	int dataSize = 400 * 400;

	Triangle* d_triangle = nullptr;
	cudaMalloc((void**)&d_triangle, sizeof(Triangle));

	Ray* d_rays = nullptr;
	cudaMalloc((void**)&d_rays, dataSize * sizeof(Ray));

	dim3 blockSize(8, 8, 1);
	dim3 gridSize;
	gridSize.x = (400 / blockSize.x) + 1;
	gridSize.y = (400 / blockSize.y) + 1;

	Camera* camera = new Camera();
	camera->m_position = glm::vec3(0.f, 5.f, 15.f);
	camera->m_forward = glm::vec3(0.f, 0.f, -1.f);
	camera->m_worldUp = glm::vec3(0.f, 1.f, 0.f);
	camera->m_fov = 70.f;
	camera->m_screenHeight = 400.f;
	camera->m_screenWidth = 400.f;
	camera->m_nearClip = 0.1f;
	camera->m_farClip = 1000.f;
	camera->m_pitch = 0.f;
	camera->m_yaw = -90.f;
	camera->UpdateBasisAxis();

	Camera* d_camera = nullptr;
	cudaMalloc((void**)&d_camera, sizeof(Camera));

	cudaMemcpy(d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);

	generateRays << <gridSize, blockSize >> > (d_rays, d_camera);

	cudaMemcpy(d_triangle, t1, sizeof(Triangle), cudaMemcpyHostToDevice);
	intersectRays << <gridSize, blockSize >> > (d_camera, d_rays, d_triangle);
	
	Ray* rays = new Ray[dataSize];
	cudaMemcpy(rays, d_rays, dataSize * sizeof(Ray), cudaMemcpyDeviceToHost);

	saveToPPM(rays, camera->m_screenHeight, camera->m_screenWidth);

	return 0;
}
