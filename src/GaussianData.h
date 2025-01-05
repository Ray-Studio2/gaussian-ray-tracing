#pragma once

#include <happly.h>

#include <vector>
#include <string>

#include <vector_functions.h>

struct GaussianParticle
{
	float3 position;
	float3 scale;
	float4 rotation;
	float  opacity;
	float3 sh[16];
};

class GaussianData
{
public:
	GaussianData();
	~GaussianData();

	size_t getVertexCount() const;

	std::vector<GaussianParticle> m_particles;

private:
	void loadPly();
	void parse();

	std::string m_filename;
	std::string plyElementName;
	happly::PLYData* plydata;
};