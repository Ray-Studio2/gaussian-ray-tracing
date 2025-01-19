#pragma once

#include <vector>
#include <vector_functions.h>

class Plane
{
public:
	Plane()
	{
		vertices = {
			{ 0.4f,  0.5f, 0.0f},	// top right
			{ 0.4f, -0.5f, 0.0f},	// bottom right
			{-0.4f, -0.5f, 0.0f},	// bottom left
			{-0.4f,  0.5f, 0.0f}	// top left
		};

		indices = {
			0, 1, 3,	// first triangle
			1, 2, 3		// second triangle
		};
	}
	~Plane() {}

	std::vector<float3> getVertices() const { return vertices; }
	std::vector<unsigned int> getIndices() const { return indices; }

private:
	std::vector<float3> vertices;
	std::vector<unsigned int> indices;
};