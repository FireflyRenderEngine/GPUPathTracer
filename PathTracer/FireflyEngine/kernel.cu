
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
		glm::vec3 intersectPoint = ray.m_origin + ray.m_direction * t;
		intersect.m_intersectionPoint = intersectPoint;
		intersect.m_t = t;

		// Calculate the normal using barycentric coordinates
		float denom = (vertex1.y - vertex2.y) * (vertex0.x - vertex2.x) + (vertex2.x - vertex1.x) * (vertex0.y - vertex2.y);
		float wv1 = ((vertex1.y - vertex2.y) * (intersectPoint.x - vertex2.x) + (vertex2.x - vertex1.x) * (intersectPoint.y - vertex2.y)) / denom;
		float wv2 = ((vertex2.y - vertex0.y) * (intersectPoint.x - vertex2.x) + (vertex0.x - vertex2.x) * (intersectPoint.y - vertex2.y)) / denom;
		float wv3 = 1 - wv1 - wv2;
		intersect.m_normal = glm::normalize((wv1 * triangle.m_n0) + (wv2 * triangle.m_n1) + (wv3 * triangle.m_n2));
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
		// right now we are storing the object space normal. Later on we calculate the world space normal.
		intersectOut.m_normal = objectSpaceIntersect.m_normal;
		// This is the world space point of intersection
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
					if (setIntersection(tMax, intersectOut, objectSpaceIntersect, geometry.m_invTransModelMatrix, geometry.m_modelMatrix, ray)) 
					{
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
				if (setIntersection(tMax, intersectOut, objectSpaceIntersect, geometry.m_invTransModelMatrix, geometry.m_modelMatrix, ray)) 
				{
					// we store the geometry index so that we can access its BXDF later on
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

	// do matrix multiplication only once for calculating world space normal.
	// Instead of calculating a world space normal everytime a new tMax is found, do it only once at the end 
	if (intersectOut.m_hit)
	{
		intersectOut.m_normal = glm::normalize(glm::vec3(geometries[intersectOut.geometryIndex].m_invTransModelMatrix * glm::vec4(intersectOut.m_normal, 0.f)));
	}
	return intersectOut;
}

/**
 * @brief A helper function to call sampleBsdf of the intersected geometry.
	Same input/output parameters as BXDF::sampleBsdf. See utilities.h
*/
__device__ glm::vec3 getBXDF(const Ray& outgoingRay, const Intersect& intersect, glm::vec3& incomingRayDirection, Geometry* geometries, float& pdf, int depth)
{
	return (geometries[intersect.geometryIndex].m_bxdf->sampleBsdf((-outgoingRay.m_direction), incomingRayDirection, intersect, pdf, depth));
}

/**
 * @brief generates a ray that starts from the location of the sensor/camera pointing forward
 * @param camera (INPUT): contains all the forward, up, near/far clip to construct the ray direction
 * @param x (INPUT): horizontal thread index for a certain pixel
 * @param y (INPUT): vertical thread index for a certain pixel
 * @param iterations (INPUT): the current iteration. Used to generate a unique seed for the random number generator
 * @return: a Ray which contains the origin and the newly generated direction
*/
__device__ Ray& generateRay(Camera camera, int x, int y, int iterations)
{
	Ray ray;

	// TODO: add depth of field
	ray.m_origin = camera.m_position;

	curandState state1;
	curandState state2;

	curand_init((unsigned long long)clock() + x, x, 0, &state1);
	float jx = curand_uniform(&state1);
	float jy = curand_uniform(&state1);

	// Stratified sample
	float Px = ((x + jx) / camera.m_screenWidth) * 2.f - 1.f;
	float Py = 1.f - ((y + jy) / camera.m_screenHeight) * 2.f;

	glm::vec3 wLookAtPoint = camera.m_invViewProj * (glm::vec4(Px, Py, 1.f, 1.f) * camera.m_farClip);

	ray.m_direction = glm::normalize(wLookAtPoint - ray.m_origin);
	return ray;
}

/**
 * @brief Each frame calls this kernel to trace rays into the scene. 
 *		  Each calls this kernel width*height times to fill the renderbuffer.
 *		  This kernel calculates the radiance generated by "totalSamplesPerPixel" number of rays at a pixel x,y.
 *		  This radiance calculated is filled into the CUDA surface object which is bound to the renderbuffer before kernel invocation.
 * @param geometries (INPUT): a simple array of geometries present in the scene.
 * @param camera (INPUT): the camera object that contains the location, forward, up, near/far clip etc to generate the starting ray to be traced
 * @param numberOfGeometries (INPUT): the total geometries present in the scene and the array "geometries". 
 *									  Since CUDA doesn't take in std::vector, we need to manually send the size of geometries
 * @param iteration (INPUT): the current iteration. Used to average out the renderbuffer radiance (RGB) value.
 * @param maxDepth (INPUT): the maximum depth a ray is allowed to reach before terminating. 
 *							In complex scenes with lot of non-diffuse bsdfs, this needs to be high enough. But beware, the higher the maxDepth, the longer a path takes 
 * @param totalSamplesPerPixel (INPUT): to do anti aliasing. Shoots totalSamplesPerPixel for each kernel call to calculate radiance at pixel location.
 * @param d_pixelColor (INOUT): a device buffer to store radiance over iterations
 * @return : Fills the surface pointer that is bound to a certain (either FRONT or BACK) renderbuffer. This will be displayed by the framebuffer eventually
*/

__global__ void launchPathTrace(
	Geometry* geometries, 
	Camera camera, 
	int numberOfGeometries, 
	int iteration,
	int maxDepth,
	int totalSamplesPerPixel,
	glm::vec3* d_pixelColor)
{
	int x = /*200; */blockIdx.x* blockDim.x + threadIdx.x;
	int y = /*400; */blockIdx.y* blockDim.y + threadIdx.y;

	int pixelSize = camera.m_screenHeight * camera.m_screenWidth;
	int pixelIndex = y * camera.m_screenWidth + x;

	if (pixelIndex >= pixelSize)
	{
		return;
	}
	// Do Light transport here
	// Loop over total number of samples to be shot per pixel (gives us anti aliasing)
	//   A. Loop until we hit max depth or russian roulette termination
	//		1. Check if we hit a light
	//		  1.a if we hit light, then terminate
	//		2. Check what material we hit
	//		  2.a get bsdf and pdf
	//		  2.b get next ray (incoming)
	//		  2.c calculate thruput and calculate russian roulette
	//		  2.d Go bath to A

	// This is where we will store the final radiance that will be converted to RGB
	// to be stored and displayed by the render buffer
	glm::vec3 finalPixelColor(0.f);

	// when we begin tracing rays, we need to clear & reset the render buffer (done outside this kernel)
	// and clear and reset the device buffer we use for accumulation.
	// This happens every time iteration is 1.
	if (iteration == 1)
	{
		d_pixelColor[pixelIndex] = glm::vec3(0.f);
	}

	finalPixelColor.x = d_pixelColor[pixelIndex].x;
	finalPixelColor.y = d_pixelColor[pixelIndex].y;
	finalPixelColor.z = d_pixelColor[pixelIndex].z;

	int samplesPerPixel = 1;
	int pathsThatContributed = 0;
	glm::vec3 pixelColorPerPixel(0.f);
	
	while(samplesPerPixel <= totalSamplesPerPixel)
	{
		Ray& outgoingRay = generateRay(camera, x, y, iteration + samplesPerPixel);
		glm::vec3 pixelColorPerSample(0.f);
		int depth = 0;
		glm::vec3 thruput(1.f);

		do
		{
			Intersect intersect = intersectRays(outgoingRay, geometries, numberOfGeometries);
			if (!intersect.m_hit)
			{
				break;
			}
			else
			{
				Ray incomingRay;
				incomingRay.m_origin = intersect.m_intersectionPoint;

				float pdf;
				// getBXDF returns 3 things: the bsdf, pdf of that bsdf sample, and the new sampled direction
				glm::vec3 bxdf = getBXDF(outgoingRay, intersect, incomingRay.m_direction, geometries, pdf, depth);

				if (geometries[intersect.geometryIndex].m_bxdf->m_type == BXDFTyp::EMITTER)
				{
					// add to thruput and exit since we hit an emitter
					pixelColorPerSample += thruput * bxdf;
					pathsThatContributed++;
					break;
				}

				if (pdf > 0.0001f)
				{
					float dotProd = glm::abs(glm::dot(incomingRay.m_direction, intersect.m_normal));
					thruput *= dotProd * (bxdf / pdf);
				}
				else
				{
					break;
				}
				// set the next ray for tracing
				glm::vec3 originOffset = 0.005f * intersect.m_normal;
				incomingRay.m_origin += glm::dot(incomingRay.m_direction, originOffset) > 0 ? originOffset : -originOffset;

				outgoingRay = incomingRay;
			}
			depth++;
		} while (depth < maxDepth);

		pixelColorPerPixel += pixelColorPerSample;
		
		samplesPerPixel++;
	}
	
	if(pathsThatContributed > 0)
	{
		pixelColorPerPixel /= (float)(totalSamplesPerPixel);
	}

	finalPixelColor += pixelColorPerPixel;
	
	d_pixelColor[pixelIndex] = finalPixelColor;	
	finalPixelColor /= iteration;

	// clamp the final rgb color [0, 1]
	finalPixelColor = glm::vec3(glm::clamp(finalPixelColor.x, 0.f, 1.f), glm::clamp(finalPixelColor.y, 0.f, 1.f), glm::clamp(finalPixelColor.z, 0.f, 1.f));

	// write the color value to the pixel location x,y
	surf2Dwrite(make_uchar4(finalPixelColor[0] * 255, finalPixelColor[1] * 255, finalPixelColor[2] * 255, 255),
		surf,
		x * sizeof(uchar4),
		y,
		cudaBoundaryModeZero);
}

cudaError_t pxl_kernel_launcher(cudaArray_const_t array,
	const int         width,
	const int         height,
	cudaEvent_t       event,
	cudaStream_t      stream,
	Geometry* geom,
	Camera camera,
	int numGeom,
	int iteration,
	int maxDepth,
	int samplesPerPixel,
	glm::vec3* d_pixelColor)
{
	cudaError_t cuda_err;

	cuda_err = cudaBindSurfaceToArray(surf, array);

	if (cuda_err)
	{
		return cuda_err;
	}

	dim3 blockSize(16, 16, 1);
	dim3 gridSize;
	gridSize.x = ((width + blockSize.x - 1) / blockSize.x);
	gridSize.y = ((height + blockSize.y -1) / blockSize.y);
	
	launchPathTrace << </*1,1*/gridSize, blockSize, 0, stream >> > (geom, camera, numGeom, iteration, maxDepth, samplesPerPixel, d_pixelColor);

	return cudaSuccess;
}

__global__ void CreateWorldAABB(Geometry* geometries, size_t numGeometries, int numOfPrimitives, AABB* AABBArray) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (index > numOfPrimitives) {
		return;
	}

	int primitiveCount = 0;
	for (int i = 0; i < numGeometries; ++i)
	{
		primitiveCount += (geometries[i].m_geometryType == GeometryType::TRIANGLEMESH) ? geometries[i].m_numberOfTriangles : 1;
		if (index < primitiveCount) {
			if (geometries[i].m_geometryType != GeometryType::TRIANGLEMESH) {
				 BuildAABB(AABBArray[index], geometries[i], i, -1);
			}
			else {
				 BuildAABB(AABBArray[index], geometries[i], i, index - i);
			}
		}
	}

	//int primitiveIndex = 0;
	//for (int i = 0; i < numGeometries; ++i)
	//{
	//	if (index < numGeometries)
	//	{
	//		if (geometries[i].m_geometryType != GeometryType::TRIANGLEMESH)
	//		{
	//			if (index == i)
	//			{
	//				// need to break here
	//				BuildAABB(geometries[i], -1);
	//			}
	//			continue;
	//		}
	//		else
	//		{
	//			// need to break here
	//			BuildAABB(geometries[i], index);
	//			break;
	//		}
	//	}
	//	else
	//	{
	//		if (geometries[i].m_geometryType != GeometryType::TRIANGLEMESH)
	//		{
	//			if (primitiveIndex > 0)
	//			{
	//				if (index == primitiveIndex)
	//				{
	//					// found; need to break here
	//					BuildAABB(geometries[i], -1);
	//					break;
	//				}
	//				else
	//				{
	//					primitiveIndex++;
	//				}
	//			}
	//			continue;
	//		}
	//		else
	//		{
	//			if ((geometries[i].m_numTriangles + i + primitiveIndex) <= index)
	//			{
	//				primitiveIndex = geometries[i].m_numTriangles + i;
	//				continue;
	//			}
	//			else
	//			{
	//				// need to break here
	//				BuildAABB(geometries[i], index - primitiveIndex);
	//				break;
	//			}
	//		}
	//	}
	//}	 
}

int main()
{
	PathTracerState state;

	std::vector<Triangle> trianglesInMesh;
	LoadMesh(R"(..\..\sceneResources\sphere.obj)", trianglesInMesh);
	Geometry* triangleMeshGeometry = new Geometry(GeometryType::TRIANGLEMESH, glm::vec3(0.f, -0.5f, 0.f), glm::vec3(0.0f, 180.0f, 0.0f), glm::vec3(1.5f), trianglesInMesh);

	Geometry* topPlaneLightGeometry = new Geometry(GeometryType::PLANE, glm::vec3(0.f, 7.4f, 0.f), glm::vec3(90.f, 0.f, 0.f), glm::vec3(5.f));
	Geometry* leftPlaneLightGeometry = new Geometry(GeometryType::PLANE, glm::vec3(-5.f, 0.f, 0.f), glm::vec3(0.f, 90.f, 0.f), glm::vec3(5.f));
	Geometry* bottomPlaneWhiteGeometry = new Geometry(GeometryType::PLANE, glm::vec3(0.f, -7.5f, 0.f), glm::vec3(90.f, 0.f, 0.f), glm::vec3(15.f));
	Geometry* topPlaneWhiteGeometry = new Geometry(GeometryType::PLANE, glm::vec3(0.f, 7.5f, 0.f), glm::vec3(90.f, 0.f, 0.f), glm::vec3(15.f));
	Geometry* backPlaneWhiteGeometry = new Geometry(GeometryType::PLANE, glm::vec3(0.f, 0.f, -7.5f), glm::vec3(0.f), glm::vec3(15.f));
	Geometry* leftPlaneRedGeometry = new Geometry(GeometryType::PLANE, glm::vec3(-7.5f, 0.f, 0.f), glm::vec3(0.f, 90.f, 0.f), glm::vec3(15.f));
	Geometry* rightPlaneGreenGeometry = new Geometry(GeometryType::PLANE, glm::vec3(7.5f, 0.f, 0.f), glm::vec3(0.f, 90.f, 0.f), glm::vec3(15.f));


	BXDF* diffusebxdfREDMesh = new BXDF();
	diffusebxdfREDMesh->m_type = BXDFTyp::DIFFUSE;
	diffusebxdfREDMesh->m_albedo = { 1.f, 0.f, 0.f };

	BXDF* diffusebxdfGREENMesh = new BXDF();
	diffusebxdfGREENMesh->m_type = BXDFTyp::DIFFUSE;
	diffusebxdfGREENMesh->m_albedo = { 0.f, 1.f, 0.f };

	BXDF* diffusebxdfBLUEMesh = new BXDF();
	diffusebxdfBLUEMesh->m_type = BXDFTyp::DIFFUSE;
	diffusebxdfBLUEMesh->m_albedo = { 0.f, 0.f, 1.f };

	BXDF* diffusebxdfPURPLEMesh = new BXDF();
	diffusebxdfPURPLEMesh->m_type = BXDFTyp::DIFFUSE;
	diffusebxdfPURPLEMesh->m_albedo = { 1.f, 0.f, 1.f };

	BXDF* diffusebxdfWHITEMesh = new BXDF();
	diffusebxdfWHITEMesh->m_type = BXDFTyp::DIFFUSE;
	diffusebxdfWHITEMesh->m_albedo = { 1.f, 1.f, 1.f };

	BXDF* lightbxdfPlane = new BXDF();
	lightbxdfPlane->m_type = BXDFTyp::EMITTER;
	lightbxdfPlane->m_intensity = 5.0f;
	lightbxdfPlane->m_emissiveColor = { 1.f, 1.f, 1.f };

	triangleMeshGeometry->m_bxdf = diffusebxdfWHITEMesh;
	bottomPlaneWhiteGeometry->m_bxdf = diffusebxdfWHITEMesh;
	backPlaneWhiteGeometry->m_bxdf = diffusebxdfWHITEMesh;
	topPlaneWhiteGeometry->m_bxdf = diffusebxdfWHITEMesh;
	leftPlaneRedGeometry->m_bxdf = diffusebxdfREDMesh;
	rightPlaneGreenGeometry->m_bxdf = diffusebxdfGREENMesh;
	topPlaneLightGeometry->m_bxdf = lightbxdfPlane;
	leftPlaneLightGeometry->m_bxdf = lightbxdfPlane;
	
	std::vector<Geometry> geometries;
	geometries.push_back(*triangleMeshGeometry);
	geometries.push_back(*topPlaneLightGeometry);
	//geometries.push_back(*leftPlaneLightGeometry);
	geometries.push_back(*bottomPlaneWhiteGeometry);
	geometries.push_back(*backPlaneWhiteGeometry);
	geometries.push_back(*topPlaneWhiteGeometry);
	geometries.push_back(*rightPlaneGreenGeometry);
	geometries.push_back(*leftPlaneRedGeometry);

	// TODO: Load scene from file
	int windowWidth  = 800;
	int windowHeight = 800;
	int cameraResolution = windowWidth * windowHeight;

	// First we will copy the base geometry object to device memory
	state.d_geometry = nullptr;
	cudaMalloc((void**)&(state.d_geometry), sizeof(Geometry) * geometries.size());
	cudaCheckErrors("cudaMalloc geometry fail");
	cudaMemcpy(state.d_geometry, geometries.data(), sizeof(Geometry) * geometries.size(), cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy geometry fail");
	state.d_raytracableObjects = geometries.size();

	// Create and load AABB
	state.d_AABB = nullptr;
	cudaMalloc((void**)&(state.d_AABB), sizeof(AABB) * GetTotalPrimitiveCount(geometries));
	cudaCheckErrors("cudaMalloc geometry fail");
	int blockSize = 256;
	int numBlocks = (GetTotalPrimitiveCount(geometries) + 1) / blockSize;
	CreateWorldAABB << <numBlocks, blockSize >> > (state.d_geometry, geometries.size(), GetTotalPrimitiveCount(geometries), state.d_AABB);
	// Get the memory back from the GPU to CPU
	std::vector<AABB> aabbArray; 
	aabbArray.resize(GetTotalPrimitiveCount(geometries));
	cudaMemcpy(aabbArray.data(), state.d_AABB, GetTotalPrimitiveCount(geometries), cudaMemcpyDeviceToHost);
	//Create K-D Tree
	std::shared_ptr<KDTree> kdTree = std::make_shared<KDTree>(aabbArray, 0, 2);


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

	GLFWViewer* viewer = new GLFWViewer(windowWidth, windowHeight);

	glm::vec3* d_pixelColor = nullptr;
	cudaMalloc((void**)&(d_pixelColor), sizeof(glm::vec3) * windowWidth * windowHeight);
	cudaCheckErrors("cudaMalloc d_pixelColor fail");

	state.d_camera = nullptr;
	cudaMalloc((void**)&(state.d_camera), sizeof(Camera));
	cudaCheckErrors("cudaMalloc camera fail");

	int iteration = 1;

	int maxDepth = 4;
	int samplesPerPixel = 16;

	GpuTimer timer;
	float time = 0.f;

	while (!glfwWindowShouldClose(viewer->m_window))
	{
		processInput(viewer->m_window, camera, viewer, iteration, time);
		camera.m_invViewProj = camera.GetInverseViewMatrix() * camera.GetInverseProjectionMatrix();

		//
		// EXECUTE CUDA KERNEL ON RENDER BUFFER
		//

		cudaGraphicsMapResources(1, &viewer->interop->cgr[viewer->interop->index], viewer->stream);
		{
			timer.Start();
			viewer->cuda_err = pxl_kernel_launcher(viewer->interop->ca[viewer->interop->index] ,
				windowWidth,
				windowHeight,
				viewer->event,
				viewer->stream,
				state.d_geometry, 
				camera, 
				geometries.size(),
				iteration, 
				maxDepth,
				samplesPerPixel,
				d_pixelColor);
			timer.Stop();
		}
		cudaGraphicsUnmapResources(1, &viewer->interop->cgr[viewer->interop->index], viewer->stream);

		char title[256];
		time = timer.Elapsed();
		sprintf(title, "Firefly | FPS %f | iteration: %d | kernel took: %.2fs | samples per pixel: %d | max depth: %d", 1.0f/time, iteration, time/iteration, samplesPerPixel, maxDepth);
		glfwSetWindowTitle(viewer->m_window, title);

		//
		// BLIT & SWAP FBO
		// 
		glBlitNamedFramebuffer(viewer->interop->fb[viewer->interop->index], 0,
			0, 0, viewer->interop->width, viewer->interop->height,
			0, viewer->interop->height, viewer->interop->width, 0,
			GL_COLOR_BUFFER_BIT,
			GL_NEAREST);

		viewer->interop->index = (viewer->interop->index + 1) % viewer->interop->count;
		iteration++;

		glfwSwapBuffers(viewer->m_window);
		glfwPollEvents();
	}

	glfwDestroyWindow(viewer->m_window);
	glfwTerminate();

	cudaFree(state.d_geometry);
	delete viewer;
	cudaFree(d_pixelColor);
	return 0;
}
