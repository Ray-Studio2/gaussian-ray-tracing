#pragma once

#include <optix.h>

#include <gtc/matrix_transform.hpp>

#include <vector>
#include <string>
#include <iostream>

#include "../vector_math.h"


struct Primitive
{
	size_t      index;
	std::string type;
	size_t      instanceIndex;

	std::vector<float3>       vertices;
	std::vector<unsigned int> indices;
	std::vector<float3>       normals;

	size_t vertex_count;

	glm::mat4 transform;

	OptixTraversableHandle gas;
};

class Primitives
{
public:
	Primitives() {}
	~Primitives() {}

	Primitive createPlane(float3 position);
	Primitive createSphere(float3 position);
	Primitive createLoadMesh(std::string filename, float3 position);

	size_t getMeshCount() const { return numberOfMesh; }
	std::vector<Primitive>& getPrimitives() { return m_primitives; }

private:
	glm::mat4 getInitialTransform(float3 position);

	std::vector<Primitive> m_primitives;

	size_t numberOfMesh   = 0;
	size_t numberOfPlane  = 0;
	size_t numberOfSphere = 0;
	size_t numberOfLoaded = 0;
};