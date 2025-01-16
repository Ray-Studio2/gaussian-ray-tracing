#pragma once

#include <optix.h>

#include <glm.hpp>
#include <glad/glad.h>

#include <map>
#include <string>
#include <iomanip>
#include <iostream>

#include "Camera.h"
#include "Icosahedron.h"
#include "Parameters.h"
#include "GaussianData.h"
#include "CUDAOutputBuffer.h"

class GaussianTracer
{
public:

	GaussianTracer(const std::string& filename);
	~GaussianTracer();

	void setSize(unsigned int width, unsigned int height);

	void initializeOptix();
	void initParams();
	void render(CUDAOutputBuffer& output_buffer);
	
	void updateCamera(Camera& camera, bool& camera_changed);

	void addMirrorSphere();

	Params	 params;
	CUstream stream;

private:
	void createContext();
	void buildAccelationStructure();
	void createModule();
	void createProgramGroups();
	void createPipeline();
	void createSBT();

	void filterGaussians();

	// Gaussian data
	GaussianData			    m_gsData;
	std::vector<GaussianIndice> m_gsIndice;
	size_t						vertex_count;
	float						alpha_min;

	// Optix state
	OptixDeviceContext		   m_context;
	OptixBuildInput			   triangle_input;
	OptixTraversableHandle	   m_gas;
	OptixTraversableHandle	   m_ias;
	OptixTraversableHandle	   m_root;
	std::vector<OptixInstance> instances;

	OptixModule                 ptx_module;
	OptixPipelineCompileOptions pipeline_compile_options;
	OptixProgramGroup           raygen_prog_group;
	OptixProgramGroup           miss_prog_group;
	OptixProgramGroup           anyhit_prog_group;
	OptixPipeline               pipeline;
	OptixShaderBindingTable     sbt;

	Params* d_params;

	// Geometry
	std::vector<float3>       vertices;
	std::vector<unsigned int> indices;
	CUdeviceptr               d_vertices;
	CUdeviceptr               d_indices;

	// Primitives
	int numberMirrorSpheres = 0;
};