struct Geometry;
struct Ray;
struct Camera;
#include "vec3.hpp"
#include "glm.hpp"

__device__ glm::vec3 noHitColor()
{
	return glm::vec3(1.f, 0.75f, 0.79f);
}

struct PathTracerState
{
	// device side variables
	Geometry* d_geometry{ nullptr };
	glm::vec3* d_pixels{ nullptr };
	Camera* d_camera{ nullptr };
	unsigned int d_raytracableObjects{ 0 };
};