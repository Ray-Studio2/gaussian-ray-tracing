#pragma once

#include <happly.h>
#include <gtc/matrix_transform.hpp>
#include <gtc/quaternion.hpp>

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
	glm::mat4 transform;
	glm::mat3 rotation_mat;
};

class GaussianData
{
public:
	GaussianData(const std::string& filename);
	~GaussianData();

	size_t getVertexCount() const;
	float3 getCenter();

	std::vector<GaussianParticle> m_particles;

private:
	void loadPly();
	void parse();

	std::string m_filename;
	std::string plyElementName;
	happly::PLYData* plydata;

	float3 center;
};