#pragma once

#include <gtc/matrix_transform.hpp>

#include <vector>

#include "../vector_math.h"


struct Primitive
{
	size_t index;

	std::vector<float3>       vertices;
	std::vector<unsigned int> indices;
	std::vector<float3>       normals;

	size_t vertex_count;

	glm::mat4 transform;
};

class Primitives
{
public:
	Primitives() {}
	~Primitives() {}

	Primitive createPlane(float3 position);

private:
	glm::mat4 getInitialTransform(float3 position);

	std::vector<Primitive> m_primitives;
	size_t m_primitive_count = 0;
};