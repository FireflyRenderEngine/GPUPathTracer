struct Geometry;
struct Ray;
struct Camera;
#include "vec3.hpp"
#include "glm.hpp"


struct PathTracerState
{
	// device side variables
	Geometry* d_geometry{ nullptr };
	glm::vec3* d_pixels{ nullptr };
	Camera* d_camera{ nullptr };
	unsigned int d_raysToTrace{ 0 };
};