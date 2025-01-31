#pragma once

#include <vector>
#include <vector_functions.h>

#include "../vector_math.h"

class Plane
{
public:
	Plane() { createPlane(); }
	~Plane() {};

	std::vector<float3>& getVertices() { return vertices; }
	std::vector<unsigned int>& getIndices() { return indices; }

private:
	void createPlane()
	{
		unsigned int tessU = 1;
		unsigned int tessV = 1;

		const float uTile = 0.5f / float(tessU);
		const float vTile = 0.3f / float(tessV);

		float3 corner = make_float3(-1.0f, -1.0f, 0.0f); // left front corner of the plane. texcoord (0.0f, 0.0f).
		float3 normal = make_float3(0.0f, 0.0f, 1.0f);

		for (unsigned int j = 0; j <= tessV; ++j)
		{
			const float v = float(j) * vTile;

			for (unsigned int i = 0; i <= tessU; ++i)
			{
				const float u = float(i) * uTile;

				float3 vertex = corner + make_float3(u, v, 0.0f);

				vertices.push_back(vertex);
			}
		}

		const unsigned int stride = tessU + 1;
		for (unsigned int j = 0; j < tessV; ++j)
		{
			for (unsigned int i = 0; i < tessU; ++i)
			{
				indices.push_back(j * stride + i);
				indices.push_back(j * stride + i + 1);
				indices.push_back((j + 1) * stride + i + 1);

				indices.push_back((j + 1) * stride + i + 1);
				indices.push_back((j + 1) * stride + i);
				indices.push_back(j * stride + i);
			}
		}
	}

	std::vector<float3>		  vertices = {};
	std::vector<unsigned int> indices  = {};
};
