
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utilities.h"

#include <cmath>

__device__ bool intersectPlane(const Geometry& plane, const Ray& ray, Intersect& intersect)
{
	// CLARIFICATION: all the rays need to be in object space; convert the ray to world space elsewhere
	float denom = glm::dot(plane.m_normal, ray.m_direction);
	if (glm::abs(denom) > 1e-7)
	{
		glm::vec3 p0l0 = -ray.m_origin;
		float t = glm::dot(p0l0, plane.m_normal) / denom;
		glm::vec3 P = ray.m_origin + t * ray.m_direction;
		// check bounds of the plane centered at 0,0,0 in object space
		if (!(P.x >= -0.5f && P.x <= 0.5f && P.y >= -0.5f && P.y <= 0.5f))
		{
			return false;
		}
		intersect.m_t = t;
		intersect.m_intersectionPoint = P;
		intersect.m_normal = plane.m_normal;
	
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
		intersect.m_normal = glm::normalize(glm::cross(edge1, edge2));
		return true;
	}
	else // This means that there is a line intersection but not a ray intersection.
	{
		return false;
	}
}

__device__ void setIntersection(int& tMax, Intersect& intersect, glm::mat4 modelMatrix)
{
	if (intersect.m_t < tMax)
	{
		intersect.m_normal = modelMatrix * glm::vec4(intersect.m_normal, 0.f);
		tMax = intersect.m_t;
		intersect.m_hit = true;
	}
}

__device__ Intersect intersectRays(const Ray& ray, Geometry* geometries, unsigned int raytracableObjects)
{
	Intersect intersect;

	// loop through all geometries, find the smallest "t" value for a single ray
	for (int i = 0; i < raytracableObjects; ++i)
	{
		Geometry geometry = geometries[i];

		Ray& objectSpaceRay = Ray(geometry.m_inverseModelMatrix * glm::vec4(ray.m_origin, 1.f), geometry.m_inverseModelMatrix * glm::vec4(ray.m_direction, 0.f));

		int tMax = INFINITY;

		switch (geometry.m_geometryType)
		{
		case GeometryType::TRIANGLEMESH:
			for (int j = 0; j < geometry.m_numberOfTriangles; ++j)
			{
				if (intersectTriangle(geometry.m_triangles[j], objectSpaceRay, intersect))
				{
					setIntersection(tMax, intersect, geometry.m_modelMatrix);
					intersect.geometryIndex = i;
					intersect.triangleIndex = j;
				}
			}
			break;
		case GeometryType::PLANE:
			if (intersectPlane(geometry, objectSpaceRay, intersect))
			{
				setIntersection(tMax, intersect, geometry.m_modelMatrix);
				intersect.geometryIndex = i;
			}
			break;
		case GeometryType::SPHERE:
			break;
		default:
			printf("No such Geometry implemented yet!");
			break;
		}
	}
	return intersect;
}

__device__ glm::vec3 shade(const Ray& incomingRay, const Intersect& intersect, glm::vec3& outgoingRayDirection, Geometry* geometries)
{
	Geometry hitGeometry = geometries[intersect.geometryIndex];

	Ray& objectSpaceRay = Ray(hitGeometry.m_inverseModelMatrix * glm::vec4(incomingRay.m_origin, 1.f), hitGeometry.m_inverseModelMatrix * glm::vec4(incomingRay.m_direction, 0.f));
	return hitGeometry.m_bxdf->bsdf(-objectSpaceRay.m_direction, intersect.m_normal, outgoingRayDirection, intersect);
}

__device__ void generateRays(Camera* camera, Geometry* geometries, glm::vec3* pixels, unsigned int raytracableObjects)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int pixelSize = camera->m_screenHeight * camera->m_screenWidth;
	int pixelIndex = y * camera->m_screenWidth + x;

	if (pixelIndex >= pixelSize)
	{
		return;
	}
	Ray ray;
	ray.m_origin = camera->m_position;

	float Px = (x / camera->m_screenWidth) * 2.f - 1.f;
	float Py = 1.f - (y / camera->m_screenHeight) * 2.f;

	glm::vec3 wLookAtPoint = camera->GetInverseViewMatrix() * camera->GetInverseProjectionMatrix() * (glm::vec4(Px, Py, 1.f, 1.f) * camera->m_farClip);

	ray.m_direction = glm::normalize(wLookAtPoint - ray.m_origin);


	if (pixelIndex >= pixelSize)
	{
		return;
	}

	Intersect intersect = intersectRays(ray, geometries, raytracableObjects);

	if (intersect.m_hit)
	{
		Ray outgoingRay;
		outgoingRay.m_origin = intersect.m_intersectionPoint;
		pixels[pixelIndex] = shade(ray, intersect, outgoingRay.m_direction, geometries);
	}

}

__global__ void launchPathTrace(PathTracerState* state)
{
	generateRays(state->d_camera, state->d_geometry, state->d_pixels, state->d_raytracableObjects);
}

int main()
{
	PathTracerState* state;

	cudaMallocManaged((void**)&state, sizeof(PathTracerState));

	std::vector<Triangle> trianglesInMesh;
	LoadMesh(R"(..\..\sceneResources\rocketman.obj)", trianglesInMesh);
	Geometry* triangleMeshGeometry = new Geometry(GeometryType::TRIANGLEMESH, glm::vec3(0), glm::vec3(0.0f, 180.0f, 180.0f), glm::vec3(1.0f), trianglesInMesh);

	Geometry* planeLightGeometry = new Geometry(GeometryType::PLANE, glm::vec3(0.f, -7.f, 0.f), glm::vec3(45.f, 0.f, 0.f), glm::vec3(5.f));

	BXDF* diffusebxdfMesh = new BXDF();
	diffusebxdfMesh->m_type = BXDFTyp::DIFFUSE;
	diffusebxdfMesh->m_albedo = { 1.f, 0.f, 0.f };

	BXDF* lightbxdfPlane = new BXDF();
	lightbxdfPlane->m_type = BXDFTyp::EMITTER;
	lightbxdfPlane->m_intensity = 2.0f;
	lightbxdfPlane->m_emissiveColor = { 1.f, 1.f, 1.f };

	triangleMeshGeometry->m_bxdf = diffusebxdfMesh;
	planeLightGeometry->m_bxdf = lightbxdfPlane;

	std::vector<Geometry> geometries;
	geometries.push_back(*triangleMeshGeometry);
	geometries.push_back(*planeLightGeometry);

	// TODO: Load scene from file
	int windowWidth  = 800;
	int windowHeight = 800;
	int cameraResolution = windowWidth * windowHeight;

	int samplesPerPixel = 1;

	// First we will copy the base geometry object to device memory
	state->d_geometry = nullptr;
	cudaMalloc((void**)&(state->d_geometry), sizeof(Geometry) * geometries.size());
	cudaCheckErrors("cudaMalloc geometry fail");
	cudaMemcpy(state->d_geometry, geometries.data(), sizeof(Geometry) * geometries.size(), cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy geometry fail");
	state->d_raytracableObjects = geometries.size();

	

	// Now we will save the internal triangle data to device memory
	for (int i = 0; i < geometries.size(); ++i)
	{
		BXDF* hostBXDFData;
		cudaMallocManaged((void**)&hostBXDFData, sizeof(BXDF));
		cudaCheckErrors("cudaMalloc host bxdf data fail");
		cudaMemcpy(hostBXDFData, geometries[i].m_bxdf, sizeof(BXDF), cudaMemcpyHostToDevice);
		cudaCheckErrors("cudaMemcpy host bxdf data fail");
		cudaMemcpy(&(state->d_geometry[i].m_bxdf), &hostBXDFData, sizeof(BXDF*), cudaMemcpyHostToDevice);
		cudaCheckErrors("cudaMemcpy device bxdf data fail");

		if (geometries[i].m_geometryType == GeometryType::TRIANGLEMESH)
		{
			// TODO: Figure out a better way to allocate and deallocate this hostTriangleData
			Triangle* hostTriangleData;
			cudaMallocManaged((void**)&hostTriangleData, sizeof(Triangle) * geometries[i].m_numberOfTriangles);
			cudaCheckErrors("cudaMalloc host triangle data fail");
			cudaMemcpy(hostTriangleData, geometries[i].m_triangles, sizeof(Triangle) * geometries[i].m_numberOfTriangles, cudaMemcpyHostToDevice);
			cudaCheckErrors("cudaMemcpy host triangle data fail");
			cudaMemcpy(&(state->d_geometry[i].m_triangles), &hostTriangleData, sizeof(Triangle*), cudaMemcpyHostToDevice);
			cudaCheckErrors("cudaMemcpy device triangle data fail");
		}
	}
	

	state->d_raysToTrace = 0;
	cudaMalloc((void**)&(state->d_raysToTrace), cameraResolution * samplesPerPixel * sizeof(unsigned int));
	cudaCheckErrors("cudaMalloc rays fail");

	glm::vec3* pixels = new glm::vec3[cameraResolution];

	state->d_pixels = nullptr;
	cudaMalloc((void**)&(state->d_pixels), cameraResolution * sizeof(glm::vec3));
	cudaCheckErrors("cudaMalloc pixels fail");
	cudaMemcpy(state->d_pixels, pixels, cameraResolution * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy pixels fail");

	dim3 blockSize(16, 16, 1);
	dim3 gridSize;
	gridSize.x = (windowWidth / blockSize.x);// +1;
	gridSize.y = (windowHeight / blockSize.y);// +1;

	Camera* camera = new Camera();
	camera->m_position = glm::vec3(0.f, 0.f, 15.f);
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
		for (int i = 0; i < cameraResolution; ++i)
		{
			pixels[i] = glm::vec3(1.f, 0.75f, 0.79f);
		}
		
		cudaMemcpy(state->d_pixels, pixels, cameraResolution * sizeof(glm::vec3), cudaMemcpyHostToDevice);
		cudaCheckErrors("cudaMemcpy pixels to device fail");
		launchPathTrace << < gridSize, blockSize >> > (state);
		cudaDeviceSynchronize();
		cudaMemcpy(pixels, state->d_pixels, sizeof(glm::vec3) * cameraResolution, cudaMemcpyDeviceToHost);
		cudaCheckErrors("cudaMemcpy pixels to host fail");

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		viewer->Draw();

		glfwSwapBuffers(viewer->m_window);
		glfwPollEvents();
	}
	
	cleanCUDAMemory(state);
	delete[] pixels;
	delete viewer;
	delete triangleMeshGeometry;
	//cudaFree(hostTriangleData);
	return 0;
}
