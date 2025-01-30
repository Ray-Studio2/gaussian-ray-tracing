#pragma once

#include <optix.h>

#include <vector>
#include <vector_functions.h>

#include "../Exception.h"

class Sphere
{
public:
	Sphere();
	~Sphere();

	void setOptixBuildInput();
	void createTraversableHandle(OptixDeviceContext& ctx);
	void createOptixInstance(unsigned int id,
							 float tx, float ty, float tz,
							 float yaw, float pitch, float roll);
	OptixInstance getInstance() const { return m_instance; }

private:
	void createSphere();

	std::vector<float3>       vertices = {};
	std::vector<unsigned int> indices  = {};

	CUdeviceptr d_vertices = 0;
	CUdeviceptr d_indices  = 0;

	OptixBuildInput sphere_input         = {};
	unsigned int triangleInputFlags[1]   = {};
	OptixAccelBuildOptions accel_options = {};

	OptixTraversableHandle m_gas           = 0;
	OptixAccelBufferSizes gas_buffer_sizes = {};
	CUdeviceptr d_gas                      = 0;

	OptixInstance m_instance = {};
};