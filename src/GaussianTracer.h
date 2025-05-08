#pragma once

#include <optix.h>

#include <glm.hpp>
#include <glad/glad.h>

#include <map>
#include <string>
#include <iomanip>
#include <iostream>

#include "Camera.h"
#include "geometry/Icosahedron.h"
#include "Parameters.h"
#include "GaussianData.h"
#include "CUDAOutputBuffer.h"
#include "geometry/Primitives.h"

enum ReflectionPrimitiveType
{
	PLANE = 0,
	SPHERE,
	CUSTOM
};

class GaussianTracer
{
public:
	GaussianTracer(const std::string& filename);
	~GaussianTracer();

	void initializeOptix();

	void render(CUDAOutputBuffer& output_buffer);
	
	void updateCamera(Camera& camera, bool& camera_changed);

	void updateInstanceTransforms(Primitive& p);

	std::vector<Primitive>& getPrimitives() { return primitives->getPrimitives(); }
	void removePrimitive();

	void setRenderType(unsigned int renderType);

	void setSize(unsigned int width, unsigned int height);
	float3 getGaussianCenter() { return m_gsData.getCenter(); }

	Params	 params;
	CUstream stream;

	// Primitives
	Primitives* primitives = new Primitives();
	void createPlane();
	void createSphere();
	void createLoadMesh(std::string filename);

private:
	// Initialize Optix pipeline.
	void createContext();
	void createModule();
	void createProgramGroups();
	void createPipeline();
	void createSBT();
	void initializeParams();

	void createGaussianParticlesBVH();

	OptixTraversableHandle createGAS(std::vector<float3> const& vs,
								     std::vector<unsigned int> const& is);
	OptixInstance createIAS(OptixTraversableHandle const& gas,
							glm::mat4 transform,
							size_t index);
	void buildAccelationStructure(std::vector<OptixInstance>& instances, OptixTraversableHandle& handle);

	void updateParamsTraversableHandle();
	void sendGeometryAttributesToDevice(Primitive p);
	void updateGeometryAttributesToDevice(Primitive& p);

	// Optix state
	OptixDeviceContext		    m_context;
	OptixModule                 ptx_module;
	OptixPipelineCompileOptions pipeline_compile_options;
	OptixProgramGroup           raygen_prog_group;
	OptixProgramGroup           miss_prog_group;
	OptixProgramGroup           hit_prog_group;
	OptixPipeline               pipeline;
	OptixShaderBindingTable     sbt;

	// Geometry
	OptixTraversableHandle	   gaussian_handle;
	std::vector<OptixInstance> gaussian_instances;

	std::vector<OptixInstance> mesh_instances;
	OptixTraversableHandle     mesh_handle;
	
	// Gaussian data
	GaussianData m_gsData;
	size_t		 particle_count;
	float		 alpha_min;	
	std::vector<float3>       vertices;
	std::vector<unsigned int> indices;	

	// Mesh
	std::vector<Mesh> meshes;

	Params* d_params;

	// ETC
	float3 current_lookat;
};