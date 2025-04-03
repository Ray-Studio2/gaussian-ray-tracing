#pragma once

#include <optix.h>

#include <vector_functions.h>

#include "GaussianData.h"
#include "geometry/Primitives.h"

#define MAX_K 6
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

enum RayType
{
	RAY_TYPE_RADIANCE = 0,
	RAY_TYPE_COUNT
};

struct RayGenData { };
struct MissData   { };
struct HitData    { };

template <typename T>
struct Record
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

struct HitInfo
{
	float t;
	int particleIndex;
};

struct Vertex
{
	float3 position;
	float3 normal;
};

struct Face
{
	uint3 indices;
};

struct Mesh
{
	Vertex* vertices;
	Face*   faces;
};

struct Params
{
	uchar3* output_buffer;

	unsigned int width;
	unsigned int height;
	int			 k;
	unsigned int sh_degree_max;

	float3 eye;
	float3 U;
	float3 V;
	float3 W;

	float t_min;
	float t_max;
	float T_min;
	float alpha_min;

	OptixTraversableHandle handle;
	GaussianParticle* d_particles;

	// Reflection
	OptixTraversableHandle reflection_handle;
	bool has_reflection_objects;
	bool reflection_render_normals;

	// FishEye
	bool mode_fisheye;

	Mesh* d_meshes;
};

struct RayPayload
{
	HitInfo      k_closest[MAX_K + 1];
	unsigned int hit_count;

	bool hit_reflection_primitive;
	float t_hit_reflection;
	Vertex reflection_vertex;
};

struct GaussianIndice
{
	size_t index;
};

typedef Record<RayGenData> RayGenRecord;
typedef Record<MissData>   MissRecord;
typedef Record<HitData>    HitRecord;