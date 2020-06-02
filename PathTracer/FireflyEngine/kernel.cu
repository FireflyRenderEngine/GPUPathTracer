
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utilities.h"

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

__device__ bool intersectTriangle(const Triangle& triangle, const Ray& ray, Intersect& intersect)
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
		intersect.m_intersectionPoint = ray.m_origin + ray.m_direction * t;
		intersect.m_t = t;
		return true;
	}
	else // This means that there is a line intersection but not a ray intersection.
	{
		return false;
	}
}

__global__ void intersectRays(Camera* camera, Ray* rays, Triangle* triangles, glm::vec3* pixels)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int pixelSize = camera->m_screenWidth * camera->m_screenHeight;
	int pixelIndex = y * camera->m_screenWidth + x;

	if (pixelIndex >= pixelSize)
	{
		return;
	}

	Intersect intersect;

	for (int i = 0; i < 10; ++i) 
	{
		if (intersectTriangle(triangles[i], rays[pixelIndex], intersect)) 
		{
			pixels[pixelIndex] = glm::vec3(255, 0.0f, 0.0f);
			continue;
		}
	}
}

int main()
{
	// Load a triangle mesh
	std::vector<Triangle*> trianglesInMesh;
	for (int i = 0 ; i < 10; ++i)
	{
		Triangle* triangle = new Triangle(
			glm::vec3(i, i, i),
			glm::vec3(i + 1.f, i + 1.f, 0.0f),
			glm::vec3(i + 2.f, 0.0f, 0.0f), 
			glm::vec2(0.0f), glm::vec2(0.0f), glm::vec2(0.0f),
			glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(0.0f));
		trianglesInMesh.push_back(triangle);
	}
	Mesh* triangleMesh = new Mesh(2, glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(1.0f), trianglesInMesh);

	int windowWidth = 800;
	int windowHeight = 800;
	int dataSize = windowWidth * windowHeight;

	Triangle* d_triangle = nullptr;
	cudaMalloc((void**)&d_triangle, sizeof(Triangle) * triangleMesh->m_numberOfTriangles);

	Ray* d_rays = nullptr;
	cudaMalloc((void**)&d_rays, dataSize * sizeof(Ray));

	glm::vec3* pixels = new glm::vec3[dataSize];
	// Initialize all the pixels with a base color of white
	for (int i = 0 ; i < dataSize; ++i) 
	{
		pixels[i] = glm::vec3(255.f, 255.f, 255.f);
	}
	glm::vec3* d_pixels = nullptr;
	cudaMalloc((void**)&d_pixels, dataSize * sizeof(glm::vec3));
	cudaMemcpy(d_pixels, pixels, dataSize * sizeof(glm::vec3), cudaMemcpyHostToDevice);

	dim3 blockSize(16, 16, 1);
	dim3 gridSize;
	gridSize.x = (windowWidth / blockSize.x);// +1;
	gridSize.y = (windowHeight / blockSize.y);// +1;

	Camera* camera = new Camera();
	camera->m_position = glm::vec3(0.f, 5.f, 15.f);
	camera->m_forward = glm::vec3(0.f, 0.f, -1.f);
	camera->m_worldUp = glm::vec3(0.f, 1.f, 0.f);
	camera->m_fov = 70.f;
	camera->m_screenHeight = float(windowWidth);
	camera->m_screenWidth = float(windowHeight);
	camera->m_nearClip = 0.1f;
	camera->m_farClip = 1000.f;
	camera->m_pitch = 0.f;
	camera->m_yaw = -90.f;
	camera->UpdateBasisAxis();

	GLFWViewer* viewer = new GLFWViewer(windowWidth, windowHeight, pixels);
	viewer->Create();

	Camera* d_camera = nullptr;
	cudaMalloc((void**)&d_camera, sizeof(Camera));

	cudaMemcpy(d_triangle, triangleMesh->m_triangles, sizeof(Triangle) * triangleMesh->m_numberOfTriangles, cudaMemcpyHostToDevice);

	while (!glfwWindowShouldClose(viewer->m_window))
	{
		processInput(viewer->m_window, camera, pixels);
		cudaMemcpy(d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);
		generateRays << <gridSize, blockSize >> > (d_rays, d_camera);
		// Initialize all the pixels with a base color of white
		for (int i = 0; i < dataSize; ++i)
		{
			pixels[i] = glm::vec3(255.f, 255.f, 255.f);
		}
		cudaMemcpy(d_pixels, pixels, dataSize * sizeof(glm::vec3), cudaMemcpyHostToDevice);
		intersectRays << <gridSize, blockSize >> > (d_camera, d_rays, d_triangle, d_pixels);
		cudaMemcpy(pixels, d_pixels, sizeof(glm::vec3) * dataSize, cudaMemcpyDeviceToHost);

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		viewer->Draw();

		glfwSwapBuffers(viewer->m_window);
		glfwPollEvents();
	}
	
	//Ray* rays = new Ray[dataSize];
	//cudaMemcpy(rays, d_rays, dataSize * sizeof(Ray), cudaMemcpyDeviceToHost);

	delete pixels;
	return 0;
}
