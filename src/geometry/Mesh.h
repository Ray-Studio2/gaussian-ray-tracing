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

struct Offset
{
	size_t vertex_offset;
	size_t primitive_offset;
};

class MeshData
{
public:
	MeshData();
	~MeshData();

	size_t getMeshCount() const;
	size_t getVertexCount() const;
	size_t getPrimitiveCount() const;

	void addMesh(Sphere& s);
	void addMesh(Plane& p);

	float3 transform_position(float3 position, glm::mat4 transform);
	float3 transform_normal(float3 normal, glm::mat4 transform);

	std::vector<Offset> m_offsets;
	std::vector<Vertex> m_vertices;
	std::vector<uint3> m_primitives;
};