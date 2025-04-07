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

	void setSize(unsigned int width, unsigned int height);
	float3 getGaussianCenter() { return m_gsData.getCenter(); }

	void initializeOptix();
	void initParams();
	void render(CUDAOutputBuffer& output_buffer);
	
	void updateCamera(Camera& camera, bool& camera_changed);

	void updateInstanceTransforms(Primitive& p);

	std::vector<Primitive>& getPrimitives() { return primitives->getPrimitives(); }
	void removePrimitive();

	void setReflectionMeshRenderNormal(bool val);

	Params	 params;
	CUstream stream;

	// Primitives
	Primitives* primitives = new Primitives();
	void createPlane();
	void createSphere();
	void createLoadMesh(std::string filename);

private:
	void createContext();
	void buildAccelationStructure();
	void buildReflectionAccelationStructure();
	void createModule();
	void createProgramGroups();
	void createPipeline();
	void createSBT();

	void createGaussiansASV1();
	void createGaussiansASV2();
	void updateParamsTraversableHandle();

	OptixTraversableHandle createGAS(std::vector<float3> const& vs, std::vector<unsigned int> const& is);
	OptixInstance createIAS(OptixTraversableHandle const& gas, glm::mat4 transform);

	void sendGeometryAttributesToDevice(Primitive p);
	void updateGeometryAttributesToDevice(Primitive& p);

	void filterGaussians();

	// Gaussian data
	GaussianData			    m_gsData;
	std::vector<GaussianIndice> m_gsIndice;
	size_t						particle_count;
	float						alpha_min;
	
	OptixTraversableHandle	   gaussian_handle;
	std::vector<OptixInstance> gaussian_instances;

	std::vector<float3>       vertices;
	std::vector<unsigned int> indices;
	CUdeviceptr               d_vertices;
	CUdeviceptr               d_indices;

	// Optix state
	OptixDeviceContext		    m_context;
	OptixModule                 ptx_module;
	OptixPipelineCompileOptions pipeline_compile_options;
	OptixProgramGroup           raygen_prog_group;
	OptixProgramGroup           miss_prog_group;
	OptixProgramGroup           hit_prog_group;
	OptixPipeline               pipeline;
	OptixShaderBindingTable     sbt;

	// Reflection state
	std::vector<OptixInstance> reflection_instances;
	OptixTraversableHandle     reflection_ias = 0;

	Params* d_params;

	// Reflection meshes
	std::vector<Mesh> meshes;

	float3 current_lookat;
};