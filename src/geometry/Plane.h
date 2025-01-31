#pragma once

#include <vector>
#include <vector_functions.h>

class Plane
{
public:
	Plane() {}
	~Plane() {};

	std::vector<float3>& getVertices() { return vertices; }
	std::vector<unsigned int>& getIndices() { return indices; }

private:
	std::vector<float3> vertices = {
		{ 0.4f,  0.5f, 0.0f},	// top right
		{ 0.4f, -0.5f, 0.0f},	// bottom right
		{-0.4f, -0.5f, 0.0f},	// bottom left
		{-0.4f,  0.5f, 0.0f}	// top left
	};

	std::vector<unsigned int> indices = {
		0, 1, 3,	// first triangle
		1, 2, 3		// second triangle
	};
};
