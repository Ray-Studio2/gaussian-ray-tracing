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

	std::vector<Primitive>& getPrimitives() { return m_primitives; }
	void clearPrimitives()
	{ 
		m_primitives.clear(); 

		numberOfMesh   = 0;
		numberOfPlane  = 0;
		numberOfSphere = 0;
		numberOfLoaded = 0;
	}

	size_t& getMeshCount() { return numberOfMesh; }
	size_t& getPlaneCount() { return numberOfPlane; }
	size_t& getSphereCount() { return numberOfSphere; }
	size_t& getLoadedCount() { return numberOfLoaded; }

private:
	glm::mat4 getInitialTransform(float3 position);

	std::vector<Primitive> m_primitives;

	size_t numberOfMesh   = 0;
	size_t numberOfPlane  = 0;
	size_t numberOfSphere = 0;
	size_t numberOfLoaded = 0;
};