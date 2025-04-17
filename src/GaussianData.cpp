#include "GaussianData.h"

GaussianData::GaussianData(const std::string& filename)
{
	std::cout << "Loading Gaussian data from " << filename << "\n";

	m_filename = filename;
	plyElementName = "vertex";

	loadPly();
	parse();
}

GaussianData::~GaussianData()
{
	if (plydata) {
		delete plydata;
	}
}

void GaussianData::loadPly()
{
	plydata = new happly::PLYData(m_filename);
}

void GaussianData::parse()
{
	auto& element = plydata->getElement(plyElementName);

    auto xProps = element.getProperty<float>("x");
    auto yProps = element.getProperty<float>("y");
    auto zProps = element.getProperty<float>("z");

    auto scale0Props = element.getProperty<float>("scale_0");
    auto scale1Props = element.getProperty<float>("scale_1");
    auto scale2Props = element.getProperty<float>("scale_2");
    
    auto rot0Props = element.getProperty<float>("rot_0");
    auto rot1Props = element.getProperty<float>("rot_1");
    auto rot2Props = element.getProperty<float>("rot_2");
    auto rot3Props = element.getProperty<float>("rot_3");
    
    auto opacityProps = element.getProperty<float>("opacity");

    auto f_dc0Props = element.getProperty<float>("f_dc_0");
    auto f_dc1Props = element.getProperty<float>("f_dc_1");
    auto f_dc2Props = element.getProperty<float>("f_dc_2");

	size_t vertexCount = getVertexCount();
	std::cout << "Number of vertices: " << vertexCount << std::endl;

    for (size_t i = 0; i < vertexCount; ++i) {
        GaussianParticle p;

        p.position = make_float3(xProps[i], yProps[i], zProps[i]);
		p.scale = make_float3(expf(scale0Props[i]),
			                  expf(scale1Props[i]),
			                  expf(scale2Props[i]));
        const float norm = std::sqrt(rot0Props[i] * rot0Props[i] +
                                     rot1Props[i] * rot1Props[i] +
                                     rot2Props[i] * rot2Props[i] +
                                     rot3Props[i] * rot3Props[i]);
        p.rotation = make_float4(rot0Props[i] / norm,
                                 rot1Props[i] / norm,
                                 rot2Props[i] / norm,
                                 rot3Props[i] / norm);
        p.opacity = 1.0f / (1.0f + std::exp(-opacityProps[i]));
		p.sh[0] = make_float3(f_dc0Props[i], f_dc1Props[i], f_dc2Props[i]);
		for (int j = 1; j < 16; ++j)
        {
			p.sh[j] = make_float3(0.0f, 0.0f, 0.0f);
		}

		particles.push_back(p);
    }
}

size_t GaussianData::getVertexCount() const
{
	auto& element = plydata->getElement(plyElementName);
	return element.count;
}

float3 GaussianData::getCenter()
{
	float3 center = make_float3(0.0f, 0.0f, 0.0f);
    for (auto& p : particles)
    {
        center.x += p.position.x;
        center.y += p.position.y;
        center.z += p.position.z;
    }
    center.x /= static_cast<float>(particles.size());
    center.y /= static_cast<float>(particles.size());
    center.z /= static_cast<float>(particles.size());
    return center;
}