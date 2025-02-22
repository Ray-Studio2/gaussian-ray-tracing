#pragma once

#include <vector>
#include <string>

#include <vector_functions.h>
#include "Sphere.h"
#include "Plane.h"

struct Vertex
{
	float3 position;
	float3 normal;
};

class Mesh
{
public:
	Mesh();
	Mesh(Sphere& s);
	Mesh(Plane& s);
	~Mesh();

	size_t getVertexCount() const;

	std::vector<float3> m_positions;
	std::vector<float3> m_normals;
	std::vector<uint3> m_primitives;
};