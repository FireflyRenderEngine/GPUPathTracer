
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utilities.h"


__device__ void generateRays(Ray* rays, Camera* camera)
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

__device__ bool intersectPlane(const Geometry& plane, const Ray& ray, Intersect& intersect)
{
	// CLARIFICATION: all the rays need to be in object space; convert the ray to world space elsewhere
	float denom = glm::dot(plane.m_normal, ray.m_direction);
	if (denom > 1e-7) 
	{
		glm::vec3 p0l0 = - ray.m_origin;
		intersect.m_t = glm::dot(p0l0, plane.m_normal) / denom;
		return (intersect.m_t >= 0);
	}
	return false;
}

__device__ bool intersectTriangle(const Triangle& triangle, const Ray& ray, Intersect& intersect)
{
	// CLARIFICATION: all the rays need to be in object space; convert the ray to world space elsewhere
	const float EPSILON = 1e-7;
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

__device__ void intersectRays(Camera* camera, Ray* rays, Geometry* geometries, glm::vec3* pixels)
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

	// loop through all geometries, find the smallest "t" value for a single ray
	for (int i = 0; i < 2; ++i) 
	{
		Geometry geometry = geometries[i];

		Ray& objectSpaceRay = Ray(geometry.m_inverseModelMatrix * glm::vec4(rays[pixelIndex].m_origin, 1.f), geometry.m_inverseModelMatrix * glm::vec4(rays[pixelIndex].m_direction, 0.f));
		
		switch(geometry.m_geometryType)
		{
			case GeometryType::TRIANGLEMESH:
				for (int i = 0; i < geometry.m_numberOfTriangles; ++i)
				{
					if (intersectTriangle(geometry.m_triangles[i], objectSpaceRay, intersect))
					{
						pixels[pixelIndex] = glm::vec3(255, 0.0f, 0.0f);
						continue;
					}
				}
				break;
			case GeometryType::PLANE:
				if (intersectPlane(geometry, objectSpaceRay, intersect))
				{
					pixels[pixelIndex] = glm::vec3(0.f, 255.0f, 0.0f);
					continue;
				}
				break;
			case GeometryType::SPHERE:
				break;
			default:
				return;
		}
	}
}

__global__ void launchPathTrace(PathTracerState* state)
{
	generateRays(state->d_rays, state->d_camera);
	intersectRays(state->d_camera, state->d_rays, state->d_geometry, state->d_pixels);
}

int main()
{
	PathTracerState* state;

	cudaMallocManaged((void**)&state, sizeof(PathTracerState));

	std::vector<Triangle> trianglesInMesh;
	LoadMesh(R"(D:\PathTracers\FireflyRenderEngine\GPUPathTracer\sceneResources\wahoo.obj)", trianglesInMesh);
	Geometry* triangleMeshGeometry = new Geometry(GeometryType::TRIANGLEMESH, glm::vec3(0), glm::vec3(0.0f, 0.0f, 180.0f), glm::vec3(1.0f), trianglesInMesh);

	Geometry* planeLightGeometry = new Geometry(GeometryType::PLANE, glm::vec3(0.f, -1.f, 0.f), glm::vec3(90.f, 0.f, 0.f), glm::vec3(2.f, 2.f, 1.0f));

	std::vector<Geometry> geometries;
	geometries.push_back(*triangleMeshGeometry);
	geometries.push_back(*planeLightGeometry);

	// TODO: Load scene from file
	int windowWidth = 800;
	int windowHeight = 800;
	int dataSize = windowWidth * windowHeight;

	// First we will copy the base geometry object to device memory
	state->d_geometry = nullptr;
	cudaMalloc((void**)&(state->d_geometry), sizeof(Geometry) * geometries.size());
	cudaCheckErrors("cudaMalloc geometry fail");
	cudaMemcpy(state->d_geometry, geometries.data(), sizeof(Geometry) * geometries.size(), cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy geometry fail");

	// Now we will save the internal triangle data to device memory
	for (int i = 0; i < geometries.size(); ++i)
	{
		if (geometries[i].m_geometryType == GeometryType::TRIANGLEMESH)
		{
			// TODO: Figure out a better way to allocate and deallocate this hostTriangleData
			Triangle* hostTriangleData;
			cudaMallocManaged((void**)&hostTriangleData, sizeof(Triangle) * triangleMeshGeometry->m_numberOfTriangles);
			cudaCheckErrors("cudaMalloc host triangle data fail");
			cudaMemcpy(hostTriangleData, triangleMeshGeometry->m_triangles, sizeof(Triangle) * triangleMeshGeometry->m_numberOfTriangles, cudaMemcpyHostToDevice);
			cudaCheckErrors("cudaMemcpy host triangle data fail");
			cudaMemcpy(&(state->d_geometry[i].m_triangles), &hostTriangleData, sizeof(Triangle*), cudaMemcpyHostToDevice);
			cudaCheckErrors("cudaMemcpy device triangle data fail");
		}
	}
	

	state->d_rays = nullptr;
	cudaMalloc((void**)&(state->d_rays), dataSize * sizeof(Ray));
	cudaCheckErrors("cudaMalloc rays fail");

	glm::vec3* pixels = new glm::vec3[dataSize];
	
	state->d_pixels = nullptr;
	cudaMalloc((void**)&(state->d_pixels), dataSize * sizeof(glm::vec3));
	cudaCheckErrors("cudaMalloc pixels fail");
	cudaMemcpy(state->d_pixels, pixels, dataSize * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy pixels fail");

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

	state->d_camera = nullptr;
	cudaMalloc((void**)&(state->d_camera), sizeof(Camera));
	cudaCheckErrors("cudaMalloc camera fail");

	while (!glfwWindowShouldClose(viewer->m_window))
	{
		processInput(viewer->m_window, camera, pixels);
		cudaMemcpy(state->d_camera, camera, sizeof(Camera), cudaMemcpyHostToDevice);
		cudaCheckErrors("cudaMemcpy camera data fail");
		// Initialize all the pixels with a base color of white
		for (int i = 0; i < dataSize; ++i)
		{
			pixels[i] = glm::vec3(255.f, 255.f, 255.f);
		}
		
		cudaMemcpy(state->d_pixels, pixels, dataSize * sizeof(glm::vec3), cudaMemcpyHostToDevice);
		cudaCheckErrors("cudaMemcpy pixels to device fail");
		launchPathTrace << < gridSize, blockSize >> > (state);
		cudaDeviceSynchronize();
		cudaMemcpy(pixels, state->d_pixels, sizeof(glm::vec3) * dataSize, cudaMemcpyDeviceToHost);
		cudaCheckErrors("cudaMemcpy pixels to host fail");

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		viewer->Draw();

		glfwSwapBuffers(viewer->m_window);
		glfwPollEvents();
	}
	
	Ray* rays = new Ray[dataSize];
	cudaMemcpy(rays, state->d_rays, dataSize * sizeof(Ray), cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy rays to host fail");

	cleanCUDAMemory(state);
	delete[] pixels;
	delete[] rays;
	delete viewer;
	delete triangleMeshGeometry;
	//cudaFree(hostTriangleData);
	return 0;
}
