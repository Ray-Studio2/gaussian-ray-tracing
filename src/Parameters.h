#pragma once

#include <optix.h>

#include <vector_functions.h>

#include "GaussianData.h"
#include "geometry/Primitives.h"

#define SH_C0     0.28209479177387814f
#define SH_C1     0.4886025119029199f
#define SH_C2_0   1.0925484305920792f
#define SH_C2_1  -1.0925484305920792f
#define SH_C2_2   0.31539156525252005f
#define SH_C2_3  -1.0925484305920792f
#define SH_C2_4   0.5462742152960396f
#define SH_C3_0  -0.5900435899266435f
#define SH_C3_1   2.890611442640554f
#define SH_C3_2  -0.4570457994644658f
#define SH_C3_3   0.3731763325901154f
#define SH_C3_4  -0.4570457994644658f
#define SH_C3_5   1.445305721320277f
#define SH_C3_6  -0.5900435899266435f

struct RayGenData { };
struct MissData   { };
struct HitData    { };

template <typename T>
struct Record
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

struct Mesh
{
	uint3*  faces;
	float3* vertex_normals;
};

struct Params
{
	uchar3* output_buffer;

	unsigned int width;
	unsigned int height;
	unsigned int sh_degree_max;

	float3 eye;
	float3 U;
	float3 V;
	float3 W;

	float t_min;
	float t_max;
	float minTransmittance;
	float alpha_min;

	OptixTraversableHandle handle;
	GaussianParticle* d_particles;

	// Mesh
	OptixTraversableHandle mesh_handle;

	// FishEye
	bool mode_fisheye;

	Mesh* d_meshes;

    int32_t type;

	unsigned int* traceState;

	bool onShadow;
};

typedef Record<RayGenData> RayGenRecord;
typedef Record<MissData>   MissRecord;
typedef Record<HitData>    HitRecord;

enum MeshType
{
    MIRROR = 0,
	NORMAL = 1,
	GLASS  = 2
};

enum TraceState
{
	TraceLastGaussianPass = 0,
	TraceGaussianPass     = 1,
	TraceMeshPass         = 2,
	TraceTerminate        = 3
};