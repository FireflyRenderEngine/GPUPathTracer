#include <iostream>
#include <fstream>
#include "vector_types.h"
#include <vector>
#include <stdio.h>
#include <sstream>

#include "vec3.hpp"
#include "glm.hpp"
#include "gtc/matrix_transform.hpp"
#include "glad.h"
#include "glfw/glfw3.h"

#include "math_constants.h"

#include "cuda_gl_interop.h"

#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <functional>
#include <time.h>

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tiny_obj_loader.h"

#include "kernel.h"

// Timing

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		// returns the time elapsed in seconds
		return elapsed / 1000.f;
	}
};

// Error Reporting
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__device__ void print(const char* label, float value)
{
	printf("%s: %f\n", label, value);
}

__device__ void print(const char* label, int value)
{
	printf("%s: %d\n", label, value);
}

__device__ void print(const char* label, glm::vec3 value)
{
	printf("%s: %f, %f, %f\n", label, value[0], value[1], value[2]);
}

__device__ void print(const char* label, uchar4 value)
{
	printf("%s: %d, %d, %d, %d\n", label, value.x, value.y, value.z, value.w);
}

// ------------------DATA CONTAINER STRUCTS------------------

__device__ bool isBlack(glm::vec3 color)
{
	return color.x <= 0.0001f && color.y <= 0.0001f && color.z <= 0.0001f;
}

__device__ glm::vec2 ConcentricSampleDisk(float u1, float u2) {
	glm::vec2 uOffset = 2.0f * glm::vec2(u1, u2) - glm::vec2(1.0f);
	if (uOffset.x == 0.0f && uOffset.y == 0.0f) {
		return glm::vec2(0.0f);
	}

	float theta, r;
	if (glm::abs(uOffset.x) > glm::abs(uOffset.y)) {
		r = uOffset.x;
		theta = CUDART_PIO4_F * (uOffset.y / uOffset.x);
	}
	else {
		r = uOffset.y;
		theta = CUDART_PIO2_F - CUDART_PIO4_F * (uOffset.x / uOffset.y);
	}
	return r * glm::vec2(glm::cos(theta), glm::sin(theta));
}

__device__ glm::vec3 CosineSampleHemisphere(float u1, float u2)
{
	glm::vec2 d = ConcentricSampleDisk(u1, u2);
	float z = glm::sqrt(glm::max(0.f, 1.0f - d.x * d.x - d.y * d.y));
	return glm::vec3(d.x, d.y, z);
}

__device__ glm::vec3 UniformHemisphereSample(float u1, float u2)
{
	const float r = sqrt(1.0f - u1 * u1);
	const float phi = 2 * CUDART_PI_F * u2;

	return glm::vec3(cos(phi) * r, sin(phi) * r, u1);
}

struct Intersect
{
	Intersect() = default;
	glm::vec3 m_intersectionPoint;
	glm::vec3 m_normal;
	float m_t{ 0.f };
	bool m_hit{ false };
	int geometryIndex{ -1 };
	int triangleIndex{ -1 };
};

enum BXDFTyp
{
	EMITTER,
	DIFFUSE,
	MIRROR,
	GLASS,
	COUNT
};


/*******************************************************************************
 * Calculates an orthogonal coordinate axis system from a single normalized
 * vector.
 * We use the normal from the intersection object to calculate its tangent
 * and bitangent vectors. We do this by assigning one of the x,y,z components
 * to 0. This assumption is based off the fact that the dot product of 2
 * orthogonal vectors is 0. so if "a" and "b" are the 2 vectors,
 * a.x*b.x + a.y*b.y + a.z+b.z = 0. If we make the assumption that
 * (in local space) b.y is 0, then for a.x*bx + a.z*b.z to be zero we can assign
 * b.x as -a.z and b.z as a.x, then a.x*(-a.z) + a.z*a.x will be 0.
 * this forms the tangent. To get the bitangent, all we need to do is get the
 * cross product between normal and tangent.
 ******************************************************************************/
__device__ void calculateCoordinateAxes(glm::vec3 normal, glm::vec3& tangent, glm::vec3& bitangent)
{
	if (normal.x > normal.y)
	{
		tangent = glm::normalize(glm::vec3(-normal.z, 0.f, normal.x));
	}
	else
	{
		tangent = glm::normalize(glm::vec3(0.f, -normal.z, normal.y));
	}
	bitangent = glm::normalize(glm::cross(normal, tangent));
}

struct BXDF
{
	BXDF() = default;
	
	BXDFTyp m_type{ BXDFTyp::COUNT };

	glm::vec3 m_albedo{ -1,-1,-1 };
	glm::vec3 m_specularColor{ -1,-1,-1 };
	float m_refractiveIndex{ -1 };
	glm::vec3 m_emissiveColor{ -1,-1,-1 };
	float m_intensity{ -1 };
	glm::vec3 m_transmittanceColor{ -1,-1,-1 };
	
	/**
	 * @brief calculates the PDF of a given outgoing (newly sampled bsdf direction) direction
	 * @param incoming (INPUT): tangent space direction that is going out of the point of intersection from the previous ray
	 * @param outgoing (INPUT): tangent space direction that is going out to be traced into the next scene
	 * @return floating point value of the PDF of the sampled outgoing ray
	*/
	__device__ float pdf(const glm::vec3& outgoing, const glm::vec3& incoming)
	{
		// CLARIFICATION: all the rays need to be in object space; convert the ray to world space elsewhere

		//only diffuse for now
		//TODO: add pdf for every other material type
		if (m_type == BXDFTyp::DIFFUSE)
		{
			// cosine weighted hemisphere sampling: cosTheta / PI
			return incoming.z * outgoing.z > 0.f ? glm::abs(incoming.z) / CUDART_PI_F : 0.f;

			// uniform hemisphere sampling: 1 / 2*PI
			//return CUDART_2_OVER_PI_F * 0.25f;
		}
	}

	/**
	 * @brief sampleBsdf samples the bsdf depending on which one it is and sends back the 
	 *		  glm::vec3 scalar. Along with the randomly sampled bsdf direction and its 
	 *        (the newly generated sample's) corresponding pdf.
	 * @param outgoing (INPUT): the direction of the previous path segment that intersected with current geom that is going out of the point of intersection
	 * @param incoming (OUTPUT): the direction that will be sampled based on the bsdf lobe. Incoming here is denoted as incoming because it is assumed this ray is incoming from the emitter
	 * @param intersect (INPUT): the intersect object containing the normal at the point of intersection
	 * @param bsdfPDF (OUTPUT): the PDF of the sample we generate based on the lobe
	 * @param depth (INPUT): depth of the trace currently. Used to generate a unique random seed
	 * @return : glm::vec3 scalar value of the object's color information
	*/
	__device__ glm::vec3 sampleBsdf(const glm::vec3& outgoing, glm::vec3& incoming, const Intersect& intersect, float& bsdfPDF, int depth)
	{
		//ASSUMPTION: incoming vector points away from the point of intersection

		if (m_type == BXDFTyp::EMITTER)
		{
			// CLARIFICATION: we assume that all area lights are two-sided
			bool twoSided = true;
			incoming = glm::vec3(0, 0, 0);
			// means we have a light source
			return (intersect.m_t >= 0 && (twoSided || glm::dot(intersect.m_normal, outgoing) > 0)) ? m_emissiveColor * m_intensity : glm::vec3(0.f);// noHitColor();
		}

		//else other materials

		//only diffuse for now
		//TODO: add bsdf for every other material type
		if (m_type == BXDFTyp::DIFFUSE)
		{
			// sample a point on hemisphere to return an outgoing ray
			glm::vec2 sample;

			int x = blockIdx.x * blockDim.x + threadIdx.x;
			/* Each thread gets same seed, a different sequence
			   number, no offset */
			curandState state1;
			curand_init((unsigned long long)clock() + x, x, 0, &state1);
			sample[0] = curand_uniform(&state1);
			sample[1] = curand_uniform(&state1);
			// do warp from square to cosine weighted hemisphere
			glm::vec3 tangent, bitangent;

			// we calculate tangent and bitangent only here where we need it
			// 2 matrix multiplications need to happen with a world space normal:
			// transpose of the TBN (TangentBitangentNormal) matrix to calculate
			// the tangent space outgoing direction. So when we sample the bsdf, 
			// technically we sample the lobe in tangent space.
			// This means that once we have a tangent space sampled direction (incoming),
			// we need to convert this to world space

			calculateCoordinateAxes(intersect.m_normal, tangent, bitangent);
			glm::mat3 worldToLocal = glm::transpose(glm::mat3(tangent, bitangent, intersect.m_normal));
			glm::vec3 tangentSpaceOutgoing = worldToLocal * outgoing;
			//outgoing = UniformHemisphereSample(sample[0], sample[1]); 
			incoming = CosineSampleHemisphere(sample[0], sample[1]);
			if (tangentSpaceOutgoing.z < 0.f)
			{
				incoming.z *= -1.f;
			}
			bsdfPDF = pdf(tangentSpaceOutgoing, incoming);
			// convert tangent space bsdf sampled direction (incoming) to world space
			incoming = glm::mat3(tangent, bitangent, intersect.m_normal) * incoming;
			// albedo / PI
			return m_albedo * CUDART_2_OVER_PI_F * 0.5f;
		}
	}
};

enum GeometryType
{
	SPHERE,
	PLANE,
	TRIANGLEMESH
};

struct Triangle
{
	Triangle() = default;
	Triangle(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2,
		glm::vec2 uv0, glm::vec2 uv1, glm::vec2 uv2,
		glm::vec3 n0, glm::vec3 n1, glm::vec3 n2)
		:m_v0(v0), m_v1(v1), m_v2(v2),
		m_uv0(uv0), m_uv1(uv1), m_uv2(uv2),
		m_n0(n0), m_n1(n1), m_n2(n2)
	{
	}
	// Vertices
	glm::vec3 m_v0;
	glm::vec3 m_v1;
	glm::vec3 m_v2;
	// UV's
	glm::vec2 m_uv0;
	glm::vec2 m_uv1;
	glm::vec2 m_uv2;
	// Normals
	glm::vec3 m_n0;
	glm::vec3 m_n1;
	glm::vec3 m_n2;
};

struct Geometry
{
	Geometry() = default;
	Geometry(GeometryType geometryType, glm::vec3 position, glm::vec3 rotation, glm::vec3 scale, std::vector<Triangle> triangles = std::vector<Triangle>(), float radius = 0.f)
		: m_geometryType(geometryType), m_position(position), m_rotation(rotation), m_scale(scale)
	{
		// Translate Matrix
		glm::mat4 translateM = glm::translate(glm::mat4(1.0f), m_position);
		// Rotate Matrix
		glm::mat4 rotateM = glm::rotate(glm::mat4(1.0f), glm::radians(m_rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		rotateM *= glm::rotate(glm::mat4(1.0f), glm::radians(m_rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		rotateM *= glm::rotate(glm::mat4(1.0f), glm::radians(m_rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));
		// Scale Matrix
		glm::mat4 scaleM = glm::scale(glm::mat4(1.0f), m_scale);
		m_modelMatrix = translateM * rotateM * scaleM;

		m_inverseModelMatrix = glm::inverse(m_modelMatrix);
		m_invTransModelMatrix = glm::inverse(glm::transpose(m_modelMatrix));

		switch (m_geometryType)
		{
			case GeometryType::SPHERE:
				m_sphereRadius = radius;
				break;
			case GeometryType::PLANE:
				break;
			case GeometryType::TRIANGLEMESH:
			{
				if (triangles.empty())
					return;
				m_numberOfTriangles = (triangles).size();
				m_triangles = new Triangle[m_numberOfTriangles];
				for (int i = 0; i < m_numberOfTriangles; ++i)
				{
					m_triangles[i] = triangles[i];
				}
				break;
			}
			default:
				std::cout << "Geometry type is not supported yet!" << std::endl;
		}
	}

	// Types:
	// 1 - Plane
	// 2 - Triangle Mesh
	// 3 - Sphere
	GeometryType m_geometryType;
	glm::vec3 m_position;
	glm::vec3 m_rotation;
	glm::vec3 m_scale;
	glm::mat4 m_modelMatrix;

	glm::mat4 m_inverseModelMatrix;

	glm::mat4 m_invTransModelMatrix;

	float m_sphereRadius;
	// CLARIFICATION: normal of geometry is in its object space, this will be used in intersections/shading
	glm::vec3 m_normal{0,0,1};
	Triangle* m_triangles{ nullptr };
	int m_numberOfTriangles{ 0 };

	BXDF* m_bxdf{nullptr};
};

struct GeometryIndex {
	int m_geometryArrayIndex, m_geometryArrayTriangleIndex;
};

struct AABB {
	glm::vec3 m_minBound;
	glm::vec3 m_maxBound;
	GeometryIndex m_gometryIndex;

	// Default const.
	__host__ __device__ AABB() {}

	__host__ __device__ AABB(glm::vec3 minBound, glm::vec3 maxBound) 
	{
		m_minBound = minBound;
		m_maxBound = maxBound;
	}

	__host__ __device__ void operator=(const AABB& aabb) 
	{
		m_minBound = aabb.m_minBound;
		m_maxBound = aabb.m_maxBound;
	}
	
	__host__ __device__ void UpdateBounds(glm::vec3 point)
	{
		//X
		if (m_minBound.x > point.x) {
			m_minBound.x = point.x;
		}
		if (m_maxBound.x < point.x) {
			m_maxBound.x = point.x;
		}

		//Y
		if (m_minBound.y > point.y) {
			m_minBound.y = point.y;
		}
		if (m_maxBound.y < point.y) {
			m_maxBound.y = point.y;
		}

		//Z
		if (m_minBound.z > point.z) {
			m_minBound.z = point.z;
		}
		if (m_maxBound.z < point.z) {
			m_maxBound.z = point.z;
		}
	}

	__host__ __device__ void ResetMinAndMaxBound() {
		m_minBound = glm::vec3(FLT_MAX);
		m_maxBound = glm::vec3(-FLT_MAX);
	}

	__host__ __device__ glm::vec3 Diameter() {
		return m_maxBound - m_minBound;
	}

	__host__ __device__ float SurfaceArea() {
		glm::vec3 d = Diameter();
		return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
	}

	__device__ Intersect& GetIntersection() {}
};

struct Edge {
	float mT;
	int mAABBArrayIndex;
	// This variable keeps track of which bounding T value this edge has
	int edgePlane; // 0 - minBound, 1 - maxBound

	Edge(float t, int ep, int index) : mT(t), edgePlane(ep), mAABBArrayIndex(index) {}
};

struct sortByT {
	inline bool operator()(const Edge& a, const Edge& b) 
	{
		return (a.mT < b.mT);
	}
};

void SortEdge(std::vector<Edge>& EdgeArray)
{
	std::sort(EdgeArray.begin(), EdgeArray.end(), sortByT());
}

__host__ __device__ void BuildAABB(AABB& aabb, Geometry& geometry, int geometryArrayIndex, int triangleIndex)
{

	//aabb.ResetMinAndMaxBound();
	if (geometry.m_geometryType == GeometryType::PLANE)
	{
		// The plane is bound between -0.5 <-> 0.5 on the X and Y plane in the object space.
		// We will use these points get the min max and convert them to world space to get final AABB min & max.
		// We will give the AABB of the plane a width so as to make it 3D
		glm::vec3 minPoint = geometry.m_modelMatrix * glm::vec4(-0.5f, -0.5f, -0.01f, 1.0f);
		glm::vec3 maxPoint = geometry.m_modelMatrix * glm::vec4(0.5f, 0.5f, 0.01f, 1.0f);

		aabb.UpdateBounds(minPoint);
		aabb.UpdateBounds(maxPoint);
	}
	else if (geometry.m_geometryType == GeometryType::SPHERE)
	{
		// The sphere is represented with a radius centered around the origin.
		glm::vec3 minPoint = geometry.m_modelMatrix * glm::vec4(-geometry.m_sphereRadius, -geometry.m_sphereRadius, -geometry.m_sphereRadius, 1.0f);
		glm::vec3 maxPoint = geometry.m_modelMatrix * glm::vec4(geometry.m_sphereRadius, geometry.m_sphereRadius, geometry.m_sphereRadius, 1.0f);

		aabb.UpdateBounds(minPoint);
		aabb.UpdateBounds(maxPoint);
	}
	else if (geometry.m_geometryType == GeometryType::TRIANGLEMESH)
	{
		// We will convert the triangle from the object space to world space and use those points to build the AABB
		glm::vec3 vertex1 = geometry.m_modelMatrix * glm::vec4(geometry.m_triangles[triangleIndex].m_v0, 1.0f);
		glm::vec3 vertex2 = geometry.m_modelMatrix * glm::vec4(geometry.m_triangles[triangleIndex].m_v1, 1.0f);
		glm::vec3 vertex3 = geometry.m_modelMatrix * glm::vec4(geometry.m_triangles[triangleIndex].m_v2, 1.0f);

		aabb.UpdateBounds(vertex1);
		aabb.UpdateBounds(vertex2);
		aabb.UpdateBounds(vertex3);
	}
	aabb.m_gometryIndex.m_geometryArrayIndex = geometryArrayIndex;
	aabb.m_gometryIndex.m_geometryArrayTriangleIndex = triangleIndex;
}

void BuildAABBCPU(std::vector<AABB>& aabbArray, std::vector<Geometry>& geometries) {
	// Loop through the geometries and set each's AABB
	for (int geometryArrayindex = 0; geometryArrayindex < geometries.size(); ++geometryArrayindex) {
		if (geometries[geometryArrayindex].m_geometryType == GeometryType::TRIANGLEMESH) {
			for (int triangleIndex = 0; triangleIndex < geometries[geometryArrayindex].m_numberOfTriangles; ++triangleIndex) {
				AABB aabb;
				aabb.ResetMinAndMaxBound();
				BuildAABB(aabb, geometries[geometryArrayindex], geometryArrayindex, triangleIndex);
				aabbArray.push_back(aabb);
			}
		}
		else {
			AABB aabb;
			aabb.ResetMinAndMaxBound();
			BuildAABB(aabb, geometries[geometryArrayindex], geometryArrayindex, -1);
			aabbArray.push_back(aabb);
		}
	}
}

void CheckAABBCPU(std::vector<AABB>& aabbArray) {
	for (int i = 0; i < aabbArray.size(); ++i) {
		
	}
}

struct Scene
{
	Scene() = default;
	Scene(std::vector<Geometry*> geometries)
	{
		m_geometrySize = geometries.size();
		m_geometries = new Geometry[m_geometrySize];
		for (int i = 0; i < m_geometrySize; ++i) 
		{
			m_geometries[i] = *(geometries[i]);
		}
	}

	~Scene() 
	{
		delete m_geometries;
	}
	Geometry* m_geometries;
	int m_geometrySize;
};

struct Ray
{
	Ray() = default;
	__device__ Ray(glm::vec3 origin, glm::vec3 direction)
		:m_origin(origin), m_direction(direction)
	{
	}
	__device__ Ray& operator=(const Ray& otherRay) 
	{
		m_origin = otherRay.m_origin;
		m_direction = otherRay.m_direction;
		return *this;
	}
	__device__ Ray(const Ray& otherRay) 
	{
		m_origin = otherRay.m_origin;
		m_direction = otherRay.m_direction;
	}

	glm::vec3 m_origin;
	glm::vec3 m_direction;
	// TODO: Padding
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
	// Camera options
	float m_cameraMovementSpeed = 0.2f;
	float m_cameraMouseSensitivity = 0.2f;
	bool m_cameraFirstMouseInput = false;
	float m_xDelta = 0.f;
	float m_yDelta = 0.f;

	glm::mat4 m_invViewProj;

	void UpdateCameraScreenWidthAndHeight(float screenWidth, float screenHeight)
	{
		m_screenWidth = screenWidth;
		m_screenHeight = screenHeight;
	}

	glm::mat4 GetViewMatrix()
	{
		return glm::lookAtRH(m_position, m_position + m_forward, m_up);
	}

	glm::mat4 GetInverseViewMatrix()
	{
		return glm::inverse(GetViewMatrix());
	}

	glm::mat4 GetProjectionMatrix()
	{
		return glm::perspectiveFovRH(glm::radians(m_fov), m_screenWidth, m_screenHeight, m_nearClip, m_farClip);
	}

	glm::mat4 GetInverseProjectionMatrix()
	{
		return glm::inverse(GetProjectionMatrix());
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
	void ProcessKeyboard(int direction)
	{
		float velocity = m_cameraMouseSensitivity;
		if (direction == 0) // FORWARD
			m_position += m_forward * velocity;
		if (direction == 1) // BACKWARD
			m_position -= m_forward * velocity;
		if (direction == 2) // LEFT
			m_position -= m_right * velocity;
		if (direction == 3) // RIGHT
			m_position += m_right * velocity;
		if (direction == 4) // UP
			m_position += m_up * velocity;
		if (direction == 5) // DOWN
			m_position -= m_up * velocity;
		if (direction == 6) // YAWLEFT
		{
			m_yaw -= 0.5f;
			UpdateBasisAxis();
		}
		if (direction == 7) // YAWRIGHT
		{
			m_yaw += 0.5f;
			UpdateBasisAxis();
		}
		if (direction == 8) // PITCHUP
		{
			m_pitch += 0.5f;
			UpdateBasisAxis();
		}
		if (direction == 9) // PITCHDOWN
		{
			m_pitch -= 0.5f;
			UpdateBasisAxis();
		}
		if (direction == 10) // PITCHDOWN
		{
			resetCamera();
		}
	}

	// Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
	void ProcessMouseMovement(bool constrainPitch = true)
	{
		m_xDelta *= m_cameraMouseSensitivity;
		m_yDelta *= m_cameraMouseSensitivity;

		m_yaw += m_xDelta;
		m_pitch += m_yDelta;

		// Make sure that when pitch is out of bounds, screen doesn't get flipped
		if (constrainPitch)
		{
			if (m_pitch > 89.0f)
				m_pitch = 89.0f;
			if (m_pitch < -89.0f)
				m_pitch = -89.0f;
		}

		// Update Front, Right and Up Vectors using the updated Euler angles
		UpdateBasisAxis();
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

	void resetCamera()
	{
		m_position = glm::vec3(0.f, 0.f, 15.f);
		m_pitch = 0.f;
		m_yaw = -90.f;
		UpdateBasisAxis();
	}
};
static void glfw_error_callback(int error, const char* description)
{
	fputs(description, stderr);
}

struct pxl_interop
{
	// split GPUs?
	bool                    multi_gpu;

	// number of fbo's
	int                     count;
	int                     index;

	// w x h
	int                     width;
	int                     height;

	// GL buffers
	GLuint* fb;
	GLuint* rb;

	// CUDA resources
	cudaGraphicsResource_t* cgr;
	cudaArray_t* ca;
};

cudaError_t pxl_interop_size_set(struct pxl_interop* const interop, const int width, const int height)
{
	cudaError_t cuda_err = cudaSuccess;

	// save new size
	interop->width = width;
	interop->height = height;

	// resize color buffer
	for (int index = 0; index < interop->count; index++)
	{
		// unregister resource
		if (interop->cgr[index] != NULL)
			cuda_err = cudaGraphicsUnregisterResource(interop->cgr[index]);

		// resize rbo
		glNamedRenderbufferStorage(interop->rb[index], GL_RGBA8, width, height);

		// register rbo
		cuda_err = cudaGraphicsGLRegisterImage(&interop->cgr[index],
			interop->rb[index],
			GL_RENDERBUFFER,
			cudaGraphicsRegisterFlagsSurfaceLoadStore |
			cudaGraphicsRegisterFlagsWriteDiscard);
	}

	// map graphics resources
	cuda_err = cudaGraphicsMapResources(interop->count, interop->cgr, 0);

	// get CUDA Array refernces
	for (int index = 0; index < interop->count; index++)
	{
		cuda_err = cudaGraphicsSubResourceGetMappedArray(&interop->ca[index],
			interop->cgr[index],
			0, 0);
	}

	// unmap graphics resources
	cuda_err = cudaGraphicsUnmapResources(interop->count, interop->cgr, 0);

	return cuda_err;
}


void pxl_glfw_window_size_callback(GLFWwindow* window, int width, int height)
{
	// get context
	struct pxl_interop* const interop = (struct pxl_interop* const) glfwGetWindowUserPointer(window);

	pxl_interop_size_set(interop, width, height);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
struct GLFWViewer 
{
	GLFWViewer(int windowWidth, int windowHeight)
		: m_windowWidth(windowWidth), m_windowHeight(windowHeight)
	{
		glfwSetErrorCallback(glfw_error_callback);
		// Init the viewer
		if (!glfwInit())
			exit(EXIT_FAILURE);

		glfwWindowHint(GLFW_DEPTH_BITS, 0);
		glfwWindowHint(GLFW_STENCIL_BITS, 0);

		glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
		m_window = glfwCreateWindow(m_windowWidth, m_windowHeight, "Viewer", NULL, NULL);
		if (m_window == NULL)
		{
			std::cout << "Failed to create GLFW window" << std::endl;
			glfwTerminate();
			exit(EXIT_FAILURE);
			return;
		}
		glfwMakeContextCurrent(m_window);

		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
		{
			std::cout << "Failed to initialize GLAD" << std::endl;
			return;
		}

		// ignore vsync for now
		glfwSwapInterval(0);
		// only copy r/g/b
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);

		InitCUDA();
	}

	void InitCUDA()
	{
		int gl_device_id;
		unsigned int gl_device_count;
		cuda_err = (cudaGLGetDevices(&gl_device_count, &gl_device_id, 1, cudaGLDeviceListAll));

		int cuda_device_id = gl_device_id;
		cuda_err = (cudaSetDevice(cuda_device_id));

		//
		// MULTI-GPU?
		//
		const bool multi_gpu = gl_device_id != cuda_device_id;

		//
		// INFO
		//
		struct cudaDeviceProp props;

		cuda_err = (cudaGetDeviceProperties(&props, gl_device_id));
		printf("GL   : %-24s (%2d)\n", props.name, props.multiProcessorCount);

		cuda_err = (cudaGetDeviceProperties(&props, cuda_device_id));
		printf("CUDA : %-24s (%2d)\n", props.name, props.multiProcessorCount);

		//
		// CREATE CUDA STREAM & EVENT
		//


		cuda_err = (cudaStreamCreateWithFlags(&stream, cudaStreamDefault));   // optionally ignore default stream behavior
		cuda_err = (cudaEventCreateWithFlags(&event, cudaEventBlockingSync)); // | cudaEventDisableTiming);

		CreateInterop();
	}
	
	struct pxl_interop*
		pxl_interop_create(const bool multi_gpu, const int fbo_count)
	{
		struct pxl_interop* const interop = (struct pxl_interop* const)calloc(1, sizeof(*interop));

		interop->multi_gpu = multi_gpu;
		interop->count = fbo_count;
		interop->index = 0;

		// allocate arrays
		interop->fb = (GLuint*)calloc(fbo_count, sizeof(*(interop->fb)));
		interop->rb = (GLuint*)calloc(fbo_count, sizeof(*(interop->rb)));
		interop->cgr = (cudaGraphicsResource_t*)calloc(fbo_count, sizeof(*(interop->cgr)));
		interop->ca = (cudaArray_t*)calloc(fbo_count, sizeof(*(interop->ca)));

		// render buffer object w/a color buffer
		glCreateRenderbuffers(fbo_count, interop->rb);

		// frame buffer object
		glCreateFramebuffers(fbo_count, interop->fb);

		// attach rbo to fbo
		for (int index = 0; index < fbo_count; index++)
		{
			glNamedFramebufferRenderbuffer(interop->fb[index],
				GL_COLOR_ATTACHMENT0,
				GL_RENDERBUFFER,
				interop->rb[index]);
		}

		// return it
		return interop;
	}

	void CreateInterop()
	{
		//
		// CREATE INTEROP
		//
		// TESTING -- DO NOT SET TO FALSE, ONLY TRUE IS RELIABLE
		interop = pxl_interop_create(true /*multi_gpu*/, 2);

		//
		// RESIZE INTEROP
		//

		int width, height;

		// get initial width/height
		glfwGetFramebufferSize(m_window, &width, &height);

		// resize with initial window dimensions
		cuda_err = pxl_interop_size_set(interop, width, height);

		glfwSetWindowUserPointer(m_window, interop);
		glfwSetFramebufferSizeCallback(m_window, pxl_glfw_window_size_callback);

	}

	~GLFWViewer() 
	{
		glfwTerminate();
		glfwDestroyWindow(m_window);
	}

	// This is the quad on the screen that will be used for showing the path traced image
	int m_windowWidth, m_windowHeight;
	GLFWwindow* m_window;

	cudaStream_t stream;
	cudaEvent_t  event;

	cudaError_t cuda_err;

	struct pxl_interop* interop;
};

// ------------------UTILITY FUNCTIONS------------------
void LoadMesh(std::string meshFilePath, std::vector<Triangle> &trianglesInMesh)
{
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warn;
	std::string err;

	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, meshFilePath.c_str());

	if (!warn.empty())
	{
		std::cout << warn << std::endl;
	}

	if (!err.empty())
	{
		std::cerr << err << std::endl;
	}

	if (!ret)
	{
		std::cout << "Error loading mesh";
		return;
	}

	// Loop over shapes
	for (size_t s = 0; s < shapes.size(); s++) {
		// Loop over faces(polygon)
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			int fv = shapes[s].mesh.num_face_vertices[f];

			std::vector<glm::vec3> triangleVertices;
			std::vector<glm::vec2> triangleUVS;
			std::vector<glm::vec3> triangleNormals;
			// Loop over vertices in the face.
			for (size_t v = 0; v < fv; v++) {
				// access to vertex
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
				triangleVertices.push_back(glm::vec3(attrib.vertices[3 * idx.vertex_index + 0], attrib.vertices[3 * idx.vertex_index + 1], attrib.vertices[3 * idx.vertex_index + 2]));
				triangleNormals.push_back(glm::vec3(attrib.normals[3 * idx.normal_index + 0], attrib.normals[3 * idx.normal_index + 1], attrib.normals[3 * idx.normal_index + 2]));
				triangleUVS.push_back(glm::vec2(attrib.texcoords[2 * idx.texcoord_index + 0], attrib.texcoords[2 * idx.texcoord_index + 1]));
				// Optional: vertex colors
				// tinyobj::real_t red = attrib.colors[3*idx.vertex_index+0];
				// tinyobj::real_t green = attrib.colors[3*idx.vertex_index+1];
				// tinyobj::real_t blue = attrib.colors[3*idx.vertex_index+2];
			}
			index_offset += fv;

			// loading new trinagle
			Triangle newTriangle(triangleVertices[0], triangleVertices[1], triangleVertices[2], triangleUVS[0], triangleUVS[1], triangleUVS[2], triangleNormals[0], triangleNormals[1], triangleNormals[2]);
			trianglesInMesh.push_back(newTriangle);

			// per-face material
			shapes[s].mesh.material_ids[f];
		}
	}
}

void saveToPPM(GLFWViewer* viewer)
{
	int width = viewer->interop->width;
	int height = viewer->interop->height;
	uchar4* pixels4 = new uchar4[width * height];

	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels4);
	std::ofstream renderFile;
	renderFile.open("render.ppm");

	renderFile << "P3" << std::endl;
	renderFile << width << " " << height << std::endl;
	renderFile << 255 << std::endl;

	for (int i = (width * height) - 1; i >=0; --i)
	{
		renderFile << static_cast<int>(pixels4[i].x) << " " << static_cast<int>(pixels4[i].y) << " " << static_cast<int>(pixels4[i].z) << std::endl;
	}
	renderFile.close();
	delete[] pixels4;
}

int GetTotalPrimitiveCount(std::vector<Geometry> geometries) {
	int primitiveCount= -1;
	for (auto geometry : geometries) {
		primitiveCount += (geometry.m_numberOfTriangles == 0) ? 1 : geometry.m_numberOfTriangles;
	}
	return primitiveCount;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) 
{
	glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window, Camera& camera, GLFWViewer* viewer, int& iteration, float& time)
{
	bool cameraMoved = false;
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(0);
		cameraMoved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(1);
		cameraMoved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(2);
		cameraMoved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(3);
		cameraMoved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(4);
		cameraMoved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(5);
		cameraMoved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(6);
		cameraMoved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(7);
		cameraMoved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(8);
		cameraMoved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(9);
		cameraMoved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
	{
		camera.ProcessKeyboard(10);
		cameraMoved = true;
	}
	if ((glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS) && glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		saveToPPM(viewer);
	}

	// if the camera moved, then we need to clear our framebuffer and reset the iteration and time
	if (cameraMoved)
	{
		const GLfloat clear_color[] = { 0.0f, 0.0f, 0.0f, 0.0f };
		glClearNamedFramebufferfv(viewer->interop->fb[viewer->interop->index], GL_COLOR, 0, clear_color);
		iteration = 1;
		time = 0.f;
	}
}

struct KDNode {
	KDNode() 
	{ 
		m_leafNodeSize = -1;
		m_isLeafNode = false;
		m_indexArray = nullptr;
	}

	KDNode(size_t leafNodeSize) : m_leafNodeSize(leafNodeSize) 
	{ 
		m_isLeafNode = false;
		m_indexArray = nullptr; 
	}

	KDNode* m_lChild = nullptr;
	KDNode* m_rChild = nullptr;
	AABB m_AABB; // This is the AABB that encompasses all the meshes inside
	int m_SplitAxis; // 0 - X, 1 - Y, 2 - Z
	float m_SplitT; // this is the distance of the split along the axis

	bool m_isLeafNode;
	size_t m_leafNodeSize;
	GeometryIndex* m_indexArray;
};

struct KDTree {
	KDTree(std::vector<AABB>& AABBArray, int minLeafNodeCount) {
		m_rootNode = BuildKDTree(AABBArray, minLeafNodeCount);
	}

	~KDTree() {
		// TODO: Recursively delete the KD Tree nodes
	}
	
	// 0 = X-Axis, 1 = Y-Axis, 2 = Z-Axis
	int GetSplitAxis(AABB aabb) 
	{
		float splitAxis = -1;
		glm::vec3 diameter = aabb.Diameter();
		if (diameter.x >= diameter.y && diameter.x >= diameter.z) {
			splitAxis = 0;
		}
		else if (diameter.y >= diameter.x && diameter.y >= diameter.z) {
			splitAxis = 1;
		}
		else {
			splitAxis = 2;
		}
		return splitAxis;
	}

	void ConsolidateAABB(std::vector<AABB>& AABBArray, KDNode* rootNode) {
		for (int i = 0; i < AABBArray.size(); ++i) {
			rootNode->m_AABB.UpdateBounds(AABBArray[i].m_minBound);
			rootNode->m_AABB.UpdateBounds(AABBArray[i].m_maxBound);
		}
	}

	// initialize and fill the geomery index's of the geometry inside the node AABB
	void SetGeometryIndixes(KDNode* rootNode, std::vector<AABB>& AABBArray) {
		if (AABBArray.size() > 0) {
			rootNode->m_indexArray = new GeometryIndex[AABBArray.size()];
			for (int i = 0; i < AABBArray.size(); ++i) {
				rootNode->m_indexArray[i].m_geometryArrayIndex = AABBArray[i].m_gometryIndex.m_geometryArrayIndex;
				rootNode->m_indexArray[i].m_geometryArrayTriangleIndex = AABBArray[i].m_gometryIndex.m_geometryArrayTriangleIndex;
			}
		}
	}

	// Surface Area Huristic for finding the minimum cost of splitting the node along a given axis
	void FindLeastCostlySplittingPlaneSAH(KDNode* rootNode, int divAxis, std::vector<Edge>& edgeArray, float& bestSplitCost, int& splitIndex, int& splitAxis) {
		// Iterate through all the min planes along the div axis formed by a sorted list of AABB's
		// Find the cost of splitting for each min plane for each geometry's AABB inside the root node using SAH
		// Update the minimum cost split plane  
	
		// We will update the number of primitives left and right based on where we find the spitting plane
		size_t nLeft = (int)edgeArray.size() / 2;
		size_t nRight = 0;

		int divAxisOther1 = (divAxis + 1) % 3;
		int divAxisOther2 = (divAxis + 2) % 3;

		// The total surface area of the node bounding box encompassing all the geometries
		float TSArea = rootNode->m_AABB.SurfaceArea();
		float InvTSArea = 1.0f / TSArea;
		glm::vec3 d = rootNode->m_AABB.Diameter();

		float traversalCost = 1;
		float intersetCost = 80;

		for (int index = 0; index < edgeArray.size(); ++index) {
			if (edgeArray[index].edgePlane == 1) --nRight;
			float edgeT = edgeArray[index].mT;
			if (edgeT > rootNode->m_AABB.m_minBound[divAxis] && edgeT < rootNode->m_AABB.m_maxBound[divAxis]) {

				float leftSA = 2 * (d[divAxisOther1] * d[divAxisOther2] +
					(edgeT - rootNode->m_AABB.m_minBound[divAxis]) *
					(d[divAxisOther1] + d[divAxisOther2]));

				float rightSA = 2 * (d[divAxisOther1] * d[divAxisOther2] +
					(rootNode->m_AABB.m_maxBound[divAxis] - edgeT) *
					(d[divAxisOther1] + d[divAxisOther2]));

				float pLeft = leftSA * InvTSArea;
				float pRight = rightSA * InvTSArea;

				float cost = traversalCost + intersetCost * (pLeft * nLeft + pRight * nRight);

				if (cost < bestSplitCost) {
					bestSplitCost = cost;
					splitIndex = edgeArray[index].edgePlane;
				}
			}
			if (edgeArray[index].edgePlane == 0) ++nLeft;
		}
	}

	KDNode* BuildKDTree(std::vector<AABB>& AABBArray, int minLeafNodeCount)
	{
		// Create a new node
		KDNode* rootNode = new KDNode(AABBArray.size());

		// Create a big AABB surrounding all the geometries inside the node
		ConsolidateAABB(AABBArray, rootNode);

		// Get the split axis
		int divAxis = GetSplitAxis(rootNode->m_AABB);

		// Find the plane that gives the least cost for splitting the root node into left and right child
		int bestSplitAxis = -1;
		float bestSplitCost = FLT_MAX;
		int bestSplitIndex = -1;
		int retries = 0;
		float oldCost = AABBArray.size() * 80; // Intersect cost is set to 80. We picked this number airbitrarily
		while (bestSplitAxis == -1 && retries <= 2) {
			// Create a new array of edges from the bounds of AABB filled by the div axis T
			std::vector<Edge> edgeArray;
			for (int i = 0; i < AABBArray.size(); ++i) {
				edgeArray.push_back(Edge(AABBArray[i].m_minBound[divAxis], 0, i));
				edgeArray.push_back(Edge(AABBArray[i].m_maxBound[divAxis], 1, i));
			}
			// Sort the Edge Array
			SortEdge(edgeArray);

			FindLeastCostlySplittingPlaneSAH(rootNode, divAxis, edgeArray, bestSplitCost, bestSplitIndex, bestSplitAxis);
			divAxis = (divAxis + 1) % 3;
			retries++;
		}

		if (bestSplitAxis == -1 || bestSplitCost > (4 * oldCost) || AABBArray.size() <= minLeafNodeCount)
		{
			// LEAF NODE:
			// TODO: Add a check for max depth as a criteria for terminantion
			// Create the array of geometry indixes that are used to index into the main geometry array
			SetGeometryIndixes(rootNode, AABBArray);

			ConsolidateAABB(AABBArray, rootNode);

			rootNode->m_isLeafNode = true;
			return rootNode;
		}
		else {
			// Create local sub arrays that will be used to generate the left and right sections of the K-D Tree
			// LEFT
			std::vector<AABB> leftAABBArray;
			// fill in the array
			leftAABBArray.insert(leftAABBArray.begin(), AABBArray.begin(), AABBArray.begin() + (1));

			// RIGHT
			std::vector<AABB> rightAABBArray;
			// FIll in the array
			rightAABBArray.insert(rightAABBArray.begin(), AABBArray.begin() + 1, AABBArray.end());

			// Recursively call the BuildKDtree function for the left and right sub tree
			int newDivAxis = ((divAxis + 1) % 3);
			rootNode->m_lChild = BuildKDTree(leftAABBArray, minLeafNodeCount);
			rootNode->m_rChild = BuildKDTree(rightAABBArray, minLeafNodeCount);
		}
			
		return rootNode;
 	}

	__device__ Intersect& FindIntersection() {}

	KDNode* m_rootNode;
};