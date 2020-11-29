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

#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tiny_obj_loader.h"

#include "kernel.h"

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

__device__ bool isZero(const glm::vec3& v)
{
	return v.x == 0 && v.y == 0 && v.z == 0;
}

__device__ glm::vec3 CosineSampleHemisphere(float u1, float u2)
{
	const float r = sqrt(u1);
	const float theta = 2 * CUDART_PI_F * u2;

	const float x = r * cos(theta);
	const float y = r * sin(theta);

	return glm::vec3(x, y, sqrt(glm::max(0.0f, 1 - u1)));
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
	glm::vec3 m_tangent;
	glm::vec3 m_bitangent;
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
	
	__device__ glm::vec3 bsdf(const glm::vec3& incoming, glm::vec3& outgoing, const Intersect& intersect, int depth)
	{
		//ASSUMPTION: incoming vector points away from the point of intersection

		if (m_type == BXDFTyp::EMITTER)
		{
			// CLARIFICATION: we assume that all area lights are two-sided
			bool twoSided = true;
			outgoing = glm::vec3(0, 0, 0);
			// means we have a light source
			return (intersect.m_t >= 0 && (twoSided || glm::dot(intersect.m_normal, incoming) > 0)) ? m_emissiveColor * m_intensity : glm::vec3(0.f);// noHitColor();
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
			glm::mat3 worldToLocal = glm::transpose(glm::mat3(intersect.m_tangent, intersect.m_bitangent, intersect.m_normal));
			glm::vec3 tangentSpaceIncoming = worldToLocal * incoming;
			//outgoing = UniformHemisphereSample(sample[0], sample[1]); 
			outgoing = CosineSampleHemisphere(sample[0], sample[1]);
			if (tangentSpaceIncoming.z < 0.f)
			{
				outgoing.z *= -1.f;
			}
			outgoing = glm::mat3(intersect.m_tangent, intersect.m_bitangent, intersect.m_normal) * outgoing;
			// albedo / PI
			return m_albedo * CUDART_2_OVER_PI_F * 0.5f;
		}
	}

	__device__ float pdf(const glm::vec3& incoming, const glm::vec3& outgoing, const Intersect& intersect)
	{
		// CLARIFICATION: all the rays need to be in object space; convert the ray to world space elsewhere

		//only diffuse for now
		//TODO: add pdf for every other material type
		if (m_type == BXDFTyp::DIFFUSE )
		{
			glm::mat3 worldToLocal = glm::transpose(glm::mat3(intersect.m_tangent, intersect.m_bitangent, intersect.m_normal));
			glm::vec3 tangentSpaceIncoming = worldToLocal * incoming;
			glm::vec3 tangentSpaceOutgoing = worldToLocal * outgoing;
			
			// cosine weighted hemisphere sampling: cosTheta / PI
			return tangentSpaceIncoming.z * tangentSpaceOutgoing.z > 0.f ? glm::abs(tangentSpaceOutgoing.z) / CUDART_PI_F : 0.f;
			
			// uniform hemisphere sampling: 1 / 2*PI
			//return CUDART_2_OVER_PI_F * 0.25f;
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
	int m_numberOfTriangles;

	BXDF* m_bxdf{nullptr};
};

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

void framebuffer_size_callback(GLFWwindow* window, int width, int height) 
{
	glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window, Camera& camera, GLFWViewer* viewer, int& iterations)
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
	if (cameraMoved)
	{
		const GLfloat clear_color[] = { 0.0f, 0.0f, 0.0f, 0.0f };
		glClearNamedFramebufferfv(viewer->interop->fb[viewer->interop->index], GL_COLOR, 0, clear_color);
		iterations = 1;
	}
}