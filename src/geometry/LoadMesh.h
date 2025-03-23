#pragma once

#include <gtc/matrix_transform.hpp>

#include <vector>
#include <vector_functions.h>
#include <iostream>

#include "../vector_math.h"
#include "Mesh.h"

class LoadMesh : public Mesh
{
public:
    LoadMesh(float3 center, std::string filename)
	{
		this->filename = filename;

		createGeometry();
		setPosition(center);
		setInitialTransform();
	}
	~LoadMesh() {};

private:
	void createGeometry();

	std::string filename = "";
};
