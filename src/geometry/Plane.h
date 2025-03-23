#pragma once

#include <gtc/matrix_transform.hpp>

#include <vector>
#include <vector_functions.h>

#include "../vector_math.h"
#include "Mesh.h"

class Plane : public Mesh
{
public:
	Plane(float3 center)
	{
		createGeometry();
		setPosition(center);
		setInitialTransform();
	}
	Plane(float3 center, std::string filename) {}
	~Plane() {};

private:
	void createGeometry()
	{
		unsigned int tessU = 1;
		unsigned int tessV = 1;

		const float width  = 0.3f;
		const float height = 0.5f;

		const float uTile = width / float(tessU);
		const float vTile = height / float(tessV);

		float3 corner = make_float3(-width * 0.5f, -height * 0.5f, 0.0f);
		float3 normal = make_float3(0.0f, 0.0f, 1.0f);

		for (unsigned int j = 0; j <= tessV; ++j)
		{
			const float v = float(j) * vTile;

			for (unsigned int i = 0; i <= tessU; ++i)
			{
				const float u = float(i) * uTile;

				float3 vertex = corner + make_float3(u, v, 0.0f);

				vertices.push_back(vertex);
				normals.push_back(normal);
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
};
