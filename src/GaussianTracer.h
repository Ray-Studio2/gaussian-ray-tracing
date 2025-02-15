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
#include "geometry/Plane.h"
#include "geometry/Sphere.h"
#include "Parameters.h"
#include "GaussianData.h"
#include "CUDAOutputBuffer.h"

struct Primitive
{
	std::string			   type;
	size_t				   index;
	float3 				   position;	// tx, ty, tz
	float3 				   rotation;	// yaw, pitch, roll
	float3 				   scale;		// sx, sy, sz
	size_t				   instanceIndex;
	OptixTraversableHandle gas;
};

enum ReflectionPrimitiveType
{
	PLANE = 0,
	SPHERE
};

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

	void createReflectionPrimitive();
	void createPlane();
	void createSphere();
	void updateInstanceTransforms(Primitive& p);

	std::vector<Primitive>& getPrimitives() { return primitives; }
	void removePrimitive(std::string primitiveType, size_t primitiveIndex, size_t instanceIndex);

	Params	 params;
	CUstream stream;	

private:
	void createContext();
	void buildAccelationStructure();
	void createModule();
	void createProgramGroups();
	void createPipeline();
	void createSBT();

	void createGaussiansAS();
	void updateParamsTraversableHandle();

	OptixBuildInput createBuildInput();

	OptixTraversableHandle createGAS(std::vector<float3> const& vs, std::vector<unsigned int> const& is);
	OptixInstance createIAS(OptixTraversableHandle const& gas, glm::mat4 transform);

	void updateSBT();

	void filterGaussians();

	// Gaussian data
	GaussianData			    m_gsData;
	std::vector<GaussianIndice> m_gsIndice;
	size_t						vertex_count;
	float						alpha_min;

	// Optix state
	OptixDeviceContext		   m_context;
	OptixBuildInput			   triangle_input;
	OptixTraversableHandle	   m_root;
	std::vector<OptixInstance> instances;

	OptixModule                 ptx_module;
	OptixPipelineCompileOptions pipeline_compile_options;
	OptixProgramGroup           raygen_prog_group;
	OptixProgramGroup           miss_prog_group;
	OptixProgramGroup           hit_prog_group;
	OptixPipeline               pipeline;
	OptixShaderBindingTable     sbt;

	Params* d_params;

	// Geometry
	std::vector<float3>       vertices;
	std::vector<unsigned int> indices;
	CUdeviceptr               d_vertices;
	CUdeviceptr               d_indices;

	// Reflection Primitives
	std::vector<OptixBuildInput> reflection_inputs;


	std::vector<Primitive> primitives;
	unsigned int		   numberOfPlanes  = 0;
	unsigned int		   numberOfSpheres = 0;
};