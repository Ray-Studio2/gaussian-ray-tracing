#pragma once

#include <happly.h>
#include <gtc/matrix_transform.hpp>
#include <gtc/quaternion.hpp>

#include <vector>
#include <string>

#include <vector_functions.h>

struct GaussianParticle
{
    glm::vec3 position;
    glm::vec3 scale;
    glm::quat rotation;

    float  opacity;
    float3 sh[16];
};

class GaussianData
{
public:
    GaussianData(const std::string& filename);
    ~GaussianData();

    size_t getVertexCount() const;
    float3 getCenter();

    std::vector<GaussianParticle> particles;

private:
    void loadPly();
    void parse();

    std::string m_filename;
    std::string plyElementName;
    happly::PLYData* plydata;

    float3 center;
};