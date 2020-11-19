
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utilities.h"

#include <cmath>

surface<void, cudaSurfaceType2D> surf;

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
	
		if (t > 0.0f) {
			intersect.m_t = t;
			intersect.m_intersectionPoint = P;
			intersect.m_normal = plane.m_normal;
			return true;
		}
		return false;
	}
	return false;
}

// fast Triangle intersection : https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
__device__ bool intersectTriangle(const Triangle& triangle, const Ray& ray, Intersect& intersect)
{
	// CLARIFICATION: all the rays need to be in object space; convert the ray to world space elsewhere
	const float EPSILON = 0.000001;
	glm::vec3 vertex0 = triangle.m_v0;
	glm::vec3 vertex1 = triangle.m_v1;
	glm::vec3 vertex2 = triangle.m_v2;
	glm::vec3 edge1, edge2, pvec, tvec, qvec;
	float det, invDet, u, v;
	edge1 = vertex1 - vertex0;
	edge2 = vertex2 - vertex0;

	pvec = glm::cross(ray.m_direction, edge2);
	det = glm::dot(edge1, pvec);

	// BACKFACE CULLING
	if (det < EPSILON) {
		return false;    // This ray is parallel to this triangle.
	}

	tvec = ray.m_origin - vertex0;
	u = glm::dot(tvec, pvec);

	if (u < 0.0f || u > det) {
		return false;
	}

	qvec = glm::cross(tvec, edge1);

	v = glm::dot(ray.m_direction, qvec);
	if (v < 0.0f || u + v > det) {
		return false;
	}

	float t = glm::dot(edge2, qvec);

	invDet = 1.0 / det;

	t *= invDet;
	u *= invDet;
	v *= invDet;

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

__device__ bool setIntersection(float& tMax, Intersect& intersectOut, const Intersect& objectSpaceIntersect, glm::mat4 invTransModelMatrix, glm::mat4 modelMatrix,const Ray& ray)
{
	// convert point of intersection into world space
	glm::vec3 worldPOI = modelMatrix * glm::vec4(objectSpaceIntersect.m_intersectionPoint, 1.0f);
	float distanceOfPOI = glm::distance(worldPOI, ray.m_origin);
	if (distanceOfPOI < tMax)
	{
		intersectOut.m_normal = invTransModelMatrix * glm::vec4(objectSpaceIntersect.m_normal, 0.f);
		intersectOut.m_intersectionPoint = worldPOI;
		intersectOut.m_t = distanceOfPOI;
		intersectOut.m_hit = true;
		tMax = distanceOfPOI;
		return true;
	}
	return false;
}

__device__ Intersect& intersectRays(const Ray& ray, Geometry* geometries, unsigned int raytracableObjects)
{
	// This is the global intersect that stores the intersect info in world space
	Intersect intersectOut;
	float tMax = INFINITY;
	// loop through all geometries, find the smallest "t" value for a single ray
	for (int i = 0; i < raytracableObjects; ++i)
	{
		Geometry& geometry = geometries[i];

		// Generate the ray in the object space of the geometry being intersected.
		Ray& objectSpaceRay = Ray(geometry.m_inverseModelMatrix * glm::vec4(ray.m_origin, 1.f), glm::normalize(geometry.m_inverseModelMatrix * glm::vec4(ray.m_direction, 0.f)));

		// This intersect is re-created each iteration and stores the intersect info in object space of the geometry
		Intersect objectSpaceIntersect;

		if (geometry.m_geometryType == GeometryType::TRIANGLEMESH)
		{
			for (int j = 0; j < geometry.m_numberOfTriangles; ++j)
			{

				if (intersectTriangle(geometry.m_triangles[j], objectSpaceRay, objectSpaceIntersect))
				{
					if (setIntersection(tMax, intersectOut, objectSpaceIntersect, geometry.m_invTransModelMatrix, geometry.m_modelMatrix, ray)) {
						intersectOut.geometryIndex = i;
						intersectOut.triangleIndex = j;
					}
				}
			}
		}
		else if (geometry.m_geometryType == GeometryType::PLANE)
		{
			if (intersectPlane(geometry, objectSpaceRay, objectSpaceIntersect))
			{
				if (setIntersection(tMax, intersectOut, objectSpaceIntersect, geometry.m_invTransModelMatrix, geometry.m_modelMatrix, ray)) {
					intersectOut.geometryIndex = i;
				}
			}
		}
		else if (geometry.m_geometryType == GeometryType::SPHERE)
		{
			printf("Sphere Geometry implemented yet!");
		}
		else
		{
			printf("No such Geometry implemented yet!");
		}
	}
	return intersectOut;
}

__device__ glm::vec3 shade(const Ray& incomingRay, const Intersect& intersect, glm::vec3& outgoingRayDirection, Geometry* geometries)
{
	Geometry hitGeometry = geometries[intersect.geometryIndex];

	//Ray& objectSpaceRay = Ray(hitGeometry.m_inverseModelMatrix * glm::vec4(incomingRay.m_origin, 1.f), hitGeometry.m_inverseModelMatrix * glm::vec4(incomingRay.m_direction, 0.f));
	return glm::abs(intersect.m_normal);// hitGeometry.m_bxdf->bsdf(-objectSpaceRay.m_direction, intersect.m_normal, outgoingRayDirection, intersect);
}

__device__ void generateRays(Camera camera, Geometry* geometries, unsigned int raytracableObjects)
{
	int x = blockIdx.x* blockDim.x + threadIdx.x;
	int y = blockIdx.y* blockDim.y + threadIdx.y;
	int pixelSize = camera.m_screenHeight * camera.m_screenWidth;
	int pixelIndex = y * camera.m_screenWidth + x;
	
	if (pixelIndex >= pixelSize)
	{
		return;
	}
	Ray ray;
	ray.m_origin = camera.m_position;
	
	float Px = (x / camera.m_screenWidth) * 2.f - 1.f;
	float Py = 1.f - (y / camera.m_screenHeight) * 2.f;
	
	glm::vec3 wLookAtPoint = camera.m_invViewProj * (glm::vec4(Px, Py, 1.f, 1.f) * camera.m_farClip);
	
	ray.m_direction = (wLookAtPoint - ray.m_origin);
	
	Intersect intersect = intersectRays(ray, geometries, raytracableObjects);
	
	if ( intersect.m_hit)
	{
		Ray outgoingRay;
		outgoingRay.m_origin = intersect.m_intersectionPoint;
		glm::vec3 pixelColor = shade(ray, intersect, outgoingRay.m_direction, geometries);
		
		surf2Dwrite(make_uchar4(pixelColor[0] * 255.f, pixelColor[1] * 255.f, pixelColor[2] * 255.f, 255.f), // even simpler: (unsigned int)clock()
		//surf2Dwrite(make_uchar4(255.f, 0.2f * 255.f, 0.9f * 255.f, 255.f), // even simpler: (unsigned int)clock
			surf,
			x * sizeof(uchar4),
			y,
			cudaBoundaryModeZero); // squelches out-of-bound writes
		//pbo[pixelIndex] = make_uchar3(pixelColor.x*255.f, pixelColor.y*255.f, pixelColor.z*255.f);
	}
}

__global__ void launchPathTrace(Geometry* geometries, Camera camera, int numberOfGeometries)
{
	generateRays(camera, geometries, numberOfGeometries);
}

cudaError_t pxl_kernel_launcher(cudaArray_const_t array,
	const int         width,
	const int         height,
	cudaEvent_t       event,
	cudaStream_t      stream,
	Geometry* geom,
	Camera camera,
	int numGeom)
{
	cudaError_t cuda_err;

	// cuda_err = cudaEventRecord(event,stream);

	cuda_err = cudaBindSurfaceToArray(surf, array);

	if (cuda_err)
		return cuda_err;

	dim3 blockSize(16, 16, 1);
	dim3 gridSize;
	gridSize.x = ((width + blockSize.x - 1) / blockSize.x);
	gridSize.y = ((height + blockSize.y -1) / blockSize.y);
	const int blocks = (width * height + 256 - 1) / 256;
	
	launchPathTrace << < gridSize, blockSize , 0, stream >> > (geom, camera, numGeom);

	return cudaSuccess;
}

int main()
{
	PathTracerState state;

	//std::vector<Triangle> trianglesInMesh;
	//LoadMesh(R"(..\..\sceneResources\rocketman.obj)", trianglesInMesh);
	//Geometry* triangleMeshGeometry = new Geometry(GeometryType::TRIANGLEMESH, glm::vec3(0), glm::vec3(0.0f, 90.0f, 180.0f), glm::vec3(1.0f), trianglesInMesh);

	Geometry*  planeLightGeometry = new Geometry(GeometryType::PLANE, glm::vec3(0.f, 0.f, 2.5f), glm::vec3(0.f, 0.f, 0.f), glm::vec3(5.f));
	Geometry* planeLightGeometry1 = new Geometry(GeometryType::PLANE, glm::vec3(0.f, 0.f, -2.5f), glm::vec3(0.f, 0.f, 0.f), glm::vec3(5.f));
	Geometry* planeLightGeometry2 = new Geometry(GeometryType::PLANE, glm::vec3(0.f, -2.5f, 0.f), glm::vec3(90.f, 0.f, 0.f), glm::vec3(5.f));
	Geometry* planeLightGeometry3 = new Geometry(GeometryType::PLANE, glm::vec3(0.f, 2.5f, 0.f), glm::vec3(90.f, 0.f, 0.f), glm::vec3(5.f));
	Geometry* planeLightGeometry4 = new Geometry(GeometryType::PLANE, glm::vec3(0.f, 0.0f, 0.f), glm::vec3(0.f, 25.f, 0.f), glm::vec3(5.f));

	BXDF* diffusebxdfMesh = new BXDF();
	diffusebxdfMesh->m_type = BXDFTyp::DIFFUSE;
	diffusebxdfMesh->m_albedo = { 1.f, 0.f, 0.f };

	BXDF* lightbxdfPlane = new BXDF();
	lightbxdfPlane->m_type = BXDFTyp::EMITTER;
	lightbxdfPlane->m_intensity = 2.0f;
	lightbxdfPlane->m_emissiveColor = { 1.f, 1.f, 1.f };

	//triangleMeshGeometry->m_bxdf = diffusebxdfMesh;
	planeLightGeometry->m_bxdf = diffusebxdfMesh;
	planeLightGeometry1->m_bxdf = diffusebxdfMesh;
	planeLightGeometry2->m_bxdf = diffusebxdfMesh;
	planeLightGeometry3->m_bxdf = diffusebxdfMesh;
	planeLightGeometry4->m_bxdf = diffusebxdfMesh;

	std::vector<Geometry> geometries;
	//geometries.push_back(*triangleMeshGeometry);
	geometries.push_back(*planeLightGeometry);
	geometries.push_back(*planeLightGeometry1);
	geometries.push_back(*planeLightGeometry2);
	geometries.push_back(*planeLightGeometry3);
	geometries.push_back(*planeLightGeometry4);

	// TODO: Load scene from file
	int windowWidth  = 800;
	int windowHeight = 800;
	int cameraResolution = windowWidth * windowHeight;

	int samplesPerPixel = 1;

	// First we will copy the base geometry object to device memory
	state.d_geometry = nullptr;
	cudaMalloc((void**)&(state.d_geometry), sizeof(Geometry) * geometries.size());
	cudaCheckErrors("cudaMalloc geometry fail");
	cudaMemcpy(state.d_geometry, geometries.data(), sizeof(Geometry) * geometries.size(), cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy geometry fail");
	state.d_raytracableObjects = geometries.size();

	// Now we will save the internal triangle data to device memory
	for (int i = 0; i < geometries.size(); ++i)
	{
		BXDF* hostBXDFData;
		cudaMallocManaged((void**)&hostBXDFData, sizeof(BXDF));
		cudaCheckErrors("cudaMalloc host bxdf data fail");
		cudaMemcpy(hostBXDFData, geometries[i].m_bxdf, sizeof(BXDF), cudaMemcpyHostToDevice);
		cudaCheckErrors("cudaMemcpy host bxdf data fail");
		cudaMemcpy(&(state.d_geometry[i].m_bxdf), &hostBXDFData, sizeof(BXDF*), cudaMemcpyHostToDevice);
		cudaCheckErrors("cudaMemcpy device bxdf data fail");

		if (geometries[i].m_geometryType == GeometryType::TRIANGLEMESH)
		{
			// TODO: Figure out a better way to allocate and deallocate this hostTriangleData
			Triangle* hostTriangleData;
			cudaMallocManaged((void**)&hostTriangleData, sizeof(Triangle) * geometries[i].m_numberOfTriangles);
			cudaCheckErrors("cudaMalloc host triangle data fail");
			cudaMemcpy(hostTriangleData, geometries[i].m_triangles, sizeof(Triangle) * geometries[i].m_numberOfTriangles, cudaMemcpyHostToDevice);
			cudaCheckErrors("cudaMemcpy host triangle data fail");
			cudaMemcpy(&(state.d_geometry[i].m_triangles), &hostTriangleData, sizeof(Triangle*), cudaMemcpyHostToDevice);
			cudaCheckErrors("cudaMemcpy device triangle data fail");
		}
	}

	state.d_raysToTrace = 0;
	cudaMalloc((void**)&(state.d_raysToTrace), cameraResolution * samplesPerPixel * sizeof(unsigned int));
	cudaCheckErrors("cudaMalloc rays fail");

	glm::vec3* pixels = new glm::vec3[cameraResolution];

	Camera camera;
	camera.m_position = glm::vec3(0.f, 0.f, 15.f);
	camera.m_forward = glm::vec3(0.f, 0.f, -1.f);
	camera.m_worldUp = glm::vec3(0.f, 1.f, 0.f);
	camera.m_fov = 70.f;
	camera.m_screenHeight = float(windowWidth);
	camera.m_screenWidth = float(windowHeight);
	camera.m_nearClip = 0.1f;
	camera.m_farClip = 1000.f;
	camera.m_pitch = 0.f;
	camera.m_yaw = -90.f;
	camera.UpdateBasisAxis();

	camera.m_invViewProj = camera.GetInverseViewMatrix() * camera.GetInverseProjectionMatrix();

	GLFWViewer* viewer = new GLFWViewer(windowWidth, windowHeight, pixels);
	//viewer->Create();

	state.d_camera = nullptr;
	cudaMalloc((void**)&(state.d_camera), sizeof(Camera));
	cudaCheckErrors("cudaMalloc camera fail");

	while (!glfwWindowShouldClose(viewer->m_window))
	{
		processInput(viewer->m_window, camera, pixels);
		camera.m_invViewProj = camera.GetInverseViewMatrix() * camera.GetInverseProjectionMatrix();

		//
		// EXECUTE CUDA KERNEL ON RENDER BUFFER
		//

		cudaGraphicsMapResources(1, &viewer->interop->cgr[viewer->interop->index], viewer->stream);
		{
			viewer->cuda_err = pxl_kernel_launcher(viewer->interop->ca[viewer->interop->index] ,
				windowWidth,
				windowHeight,
				viewer->event,
				viewer->stream,
				state.d_geometry, camera, geometries.size());
		}
		cudaGraphicsUnmapResources(1, &viewer->interop->cgr[viewer->interop->index], viewer->stream);

		std::string title = "Firefly";
		glfwSetWindowTitle(viewer->m_window, title.c_str());

		//
		// BLIT & SWAP FBO
		// 
		glBlitNamedFramebuffer(viewer->interop->fb[viewer->interop->index], 0,
			0, 0, viewer->interop->width, viewer->interop->height,
			0, viewer->interop->height, viewer->interop->width, 0,
			GL_COLOR_BUFFER_BIT,
			GL_NEAREST);

		const GLfloat clear_color[] = { 0.0f, 0.0f, 0.0f, 0.0f };
		glClearNamedFramebufferfv(viewer->interop->fb[viewer->interop->index], GL_COLOR, 0, clear_color);
		// pxl_interop_clear(interop);
		viewer->interop->index = (viewer->interop->index + 1) % viewer->interop->count;


		glfwSwapBuffers(viewer->m_window);
		glfwPollEvents();
	}

	glfwDestroyWindow(viewer->m_window);
	glfwTerminate();

	cudaFree(state.d_geometry);
	delete[] pixels;
	delete viewer;
	//delete triangleMeshGeometry;
	//cudaFree(hostTriangleData);
	return 0;
}
