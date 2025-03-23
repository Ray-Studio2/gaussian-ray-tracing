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
#include "geometry/LoadMesh.h"
#include "Parameters.h"
#include "GaussianData.h"
#include "CUDAOutputBuffer.h"
#include "geometry/Mesh.h"

struct Primitive
{
	std::string			   type;
	size_t				   index;
	glm::mat4			   transform;
	size_t				   instanceIndex;
	OptixTraversableHandle gas;
};

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

	template <typename T>
	void createGeometry(std::string geometry_name);
	template <typename T>
	void createGeometry(std::string geometry_name, std::string filename);

	void updateInstanceTransforms(Primitive& p);

	std::vector<Primitive>& getPrimitives() { return primitives; }
	void removePrimitive(std::string primitiveType, size_t primitiveIndex, size_t instanceIndex);

	Params	 params;
	CUstream stream;	

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

	void sendGeometryAttributesToDevice(glm::mat4 transform);
	void addPrimitives(OptixTraversableHandle gas, Mesh& geometry, std::string geometry_name);

	void filterGaussians();

	// Gaussian data
	GaussianData			    m_gsData;
	std::vector<GaussianIndice> m_gsIndice;
	size_t						vertex_count;
	float						alpha_min;

	// Reflection data
	MeshData					m_meshData;

	// Optix state
	OptixDeviceContext		   m_context;
	OptixBuildInput			   triangle_input;
	OptixTraversableHandle	   m_root;
	std::vector<OptixInstance> instances;

	// Reflection state
	std::vector<OptixInstance> reflection_instances;
	OptixTraversableHandle     reflection_ias = 0;

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
	std::vector<Primitive> primitives;
	unsigned int		   numberOfPlanes  = 0;
	unsigned int		   numberOfSpheres = 0;
};


template <typename T>
void GaussianTracer::createGeometry(std::string geometry_name)
{
	float3 gaussianCenter = getGaussianCenter();
	float3 cameraPosition = params.eye;
	float cameraWeight = 0.75f;
	float gaussianWeight = 1.0f - cameraWeight;

	float3 midPoint = {
		gaussianCenter.x * gaussianWeight + cameraPosition.x * cameraWeight,
		gaussianCenter.y * gaussianWeight + cameraPosition.y * cameraWeight,
		gaussianCenter.z * gaussianWeight + cameraPosition.z * cameraWeight
	};

	T geometry = T(midPoint);
	m_meshData.addMesh(geometry);

	OptixTraversableHandle gas = createGAS(geometry.getVertices(), geometry.getIndices());
	OptixInstance          ias = createIAS(gas, geometry.getTransform());

	reflection_instances.push_back(ias);

	buildReflectionAccelationStructure();
	updateParamsTraversableHandle();

	sendGeometryAttributesToDevice(geometry.getTransform());

	addPrimitives(gas, geometry, geometry_name);

	params.has_reflection_objects = true;
}

template <typename T>
void GaussianTracer::createGeometry(std::string geometry_name, std::string filename)
{
	float3 gaussianCenter = getGaussianCenter();
	float3 cameraPosition = params.eye;
	float cameraWeight = 0.75f;
	float gaussianWeight = 1.0f - cameraWeight;

	float3 midPoint = {
		gaussianCenter.x * gaussianWeight + cameraPosition.x * cameraWeight,
		gaussianCenter.y * gaussianWeight + cameraPosition.y * cameraWeight,
		gaussianCenter.z * gaussianWeight + cameraPosition.z * cameraWeight
	};

	T geometry = T(midPoint, filename);
	m_meshData.addMesh(geometry);

	OptixTraversableHandle gas = createGAS(geometry.getVertices(), geometry.getIndices());
	OptixInstance          ias = createIAS(gas, geometry.getTransform());

	reflection_instances.push_back(ias);

	buildReflectionAccelationStructure();
	updateParamsTraversableHandle();

	sendGeometryAttributesToDevice(geometry.getTransform());

	addPrimitives(gas, geometry, geometry_name);

	params.has_reflection_objects = true;
}