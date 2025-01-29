#pragma once

#include <optix.h>

#include <vector>
#include <vector_functions.h>

#include "../Exception.h"

class Plane
{
public:
	Plane() {}
	~Plane();

	void setOptixBuildInput();
	OptixBuildInput* getOptixBuildInput() { return &plane_input; }
	OptixAccelBuildOptions* getAccelOptions() { return &accel_options; }
	void createTraversableHandle(OptixDeviceContext &ctx);
	void createOptixInstance(unsigned int id, 
							 float tx, float ty, float tz, 
							 float yaw, float pitch, float roll);
	OptixInstance getInstance() const { return m_instance; }

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

	CUdeviceptr d_vertices = 0;
	CUdeviceptr d_indices  = 0;

	OptixBuildInput plane_input          = {};
	unsigned int triangleInputFlags[1]   = {};
	OptixAccelBuildOptions accel_options = {};

	OptixTraversableHandle m_gas           = 0;
	OptixAccelBufferSizes gas_buffer_sizes = {};
	CUdeviceptr d_gas                      = 0;

	OptixInstance m_instance = {};
};
