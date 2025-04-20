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
    if (plydata)
	    delete plydata;
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

    auto f_rest0Props = element.getProperty<float>("f_rest_0");
    auto f_rest1Props = element.getProperty<float>("f_rest_1");
    auto f_rest2Props = element.getProperty<float>("f_rest_2");
    auto f_rest3Props = element.getProperty<float>("f_rest_3");
    auto f_rest4Props = element.getProperty<float>("f_rest_4");
    auto f_rest5Props = element.getProperty<float>("f_rest_5");
    auto f_rest6Props = element.getProperty<float>("f_rest_6");
    auto f_rest7Props = element.getProperty<float>("f_rest_7");
    auto f_rest8Props = element.getProperty<float>("f_rest_8");
    auto f_rest9Props = element.getProperty<float>("f_rest_9");
    auto f_rest10Props = element.getProperty<float>("f_rest_10");
    auto f_rest11Props = element.getProperty<float>("f_rest_11");
    auto f_rest12Props = element.getProperty<float>("f_rest_12");
    auto f_rest13Props = element.getProperty<float>("f_rest_13");
    auto f_rest14Props = element.getProperty<float>("f_rest_14");
    auto f_rest15Props = element.getProperty<float>("f_rest_15");
    auto f_rest16Props = element.getProperty<float>("f_rest_16");
    auto f_rest17Props = element.getProperty<float>("f_rest_17");
    auto f_rest18Props = element.getProperty<float>("f_rest_18");
    auto f_rest19Props = element.getProperty<float>("f_rest_19");
    auto f_rest20Props = element.getProperty<float>("f_rest_20");
    auto f_rest21Props = element.getProperty<float>("f_rest_21");
    auto f_rest22Props = element.getProperty<float>("f_rest_22");
    auto f_rest23Props = element.getProperty<float>("f_rest_23");
    auto f_rest24Props = element.getProperty<float>("f_rest_24");
    auto f_rest25Props = element.getProperty<float>("f_rest_25");
    auto f_rest26Props = element.getProperty<float>("f_rest_26");
    auto f_rest27Props = element.getProperty<float>("f_rest_27");
    auto f_rest28Props = element.getProperty<float>("f_rest_28");
    auto f_rest29Props = element.getProperty<float>("f_rest_29");
    auto f_rest30Props = element.getProperty<float>("f_rest_30");
    auto f_rest31Props = element.getProperty<float>("f_rest_31");
    auto f_rest32Props = element.getProperty<float>("f_rest_32");
    auto f_rest33Props = element.getProperty<float>("f_rest_33");
    auto f_rest34Props = element.getProperty<float>("f_rest_34");
    auto f_rest35Props = element.getProperty<float>("f_rest_35");
    auto f_rest36Props = element.getProperty<float>("f_rest_36");
    auto f_rest37Props = element.getProperty<float>("f_rest_37");
    auto f_rest38Props = element.getProperty<float>("f_rest_38");
    auto f_rest39Props = element.getProperty<float>("f_rest_39");
    auto f_rest40Props = element.getProperty<float>("f_rest_40");
    auto f_rest41Props = element.getProperty<float>("f_rest_41");
    auto f_rest42Props = element.getProperty<float>("f_rest_42");
    auto f_rest43Props = element.getProperty<float>("f_rest_43");
    auto f_rest44Props = element.getProperty<float>("f_rest_44");

    size_t vertexCount = getVertexCount();
    std::cout << "Number of vertices: " << vertexCount << std::endl;

    for (size_t i = 0; i < vertexCount; ++i) {
        GaussianParticle p;

        p.position = glm::vec3(xProps[i], yProps[i], zProps[i]);
        p.scale = glm::vec3(expf(scale0Props[i]),
                            expf(scale1Props[i]),
                            expf(scale2Props[i]));
        const float norm = sqrtf(rot0Props[i] * rot0Props[i] +
								 rot1Props[i] * rot1Props[i] +
								 rot2Props[i] * rot2Props[i] +
								 rot3Props[i] * rot3Props[i]);
        p.rotation = glm::quat(rot0Props[i] / norm,
                               rot1Props[i] / norm,
                               rot2Props[i] / norm,
                               rot3Props[i] / norm);
        p.opacity = 1.0f / (1.0f + expf(-opacityProps[i]));
        p.sh[0]  = make_float3(f_dc0Props[i], f_dc1Props[i], f_dc2Props[i]);
        p.sh[1]  = make_float3(f_rest0Props[i], f_rest15Props[i], f_rest30Props[i]);
        p.sh[2]  = make_float3(f_rest1Props[i], f_rest16Props[i], f_rest31Props[i]);
        p.sh[3]  = make_float3(f_rest2Props[i], f_rest17Props[i], f_rest32Props[i]);
        p.sh[4]  = make_float3(f_rest3Props[i], f_rest18Props[i], f_rest33Props[i]);
        p.sh[5]  = make_float3(f_rest4Props[i], f_rest19Props[i], f_rest34Props[i]);
        p.sh[6]  = make_float3(f_rest5Props[i], f_rest20Props[i], f_rest35Props[i]);
        p.sh[7]  = make_float3(f_rest6Props[i], f_rest21Props[i], f_rest36Props[i]);
        p.sh[8]  = make_float3(f_rest7Props[i], f_rest22Props[i], f_rest37Props[i]);
        p.sh[9]  = make_float3(f_rest8Props[i], f_rest23Props[i], f_rest38Props[i]);
        p.sh[10] = make_float3(f_rest9Props[i], f_rest24Props[i], f_rest39Props[i]);
        p.sh[11] = make_float3(f_rest10Props[i], f_rest25Props[i], f_rest40Props[i]);
        p.sh[12] = make_float3(f_rest11Props[i], f_rest26Props[i], f_rest41Props[i]);
        p.sh[13] = make_float3(f_rest12Props[i], f_rest27Props[i], f_rest42Props[i]);
        p.sh[14] = make_float3(f_rest13Props[i], f_rest28Props[i], f_rest43Props[i]);
        p.sh[15] = make_float3(f_rest14Props[i], f_rest29Props[i], f_rest44Props[i]);

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