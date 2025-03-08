#pragma once

#include <gtc/matrix_transform.hpp>

#include <vector>
#include <vector_functions.h>

#include "../vector_math.h"
#include "Mesh.h"

class Sphere: public Mesh
{
public:
    Sphere()
    {
        createGeometry();
        setPosition();
        setRotation();
        setScale();
    }
    ~Sphere() {}


private:
    void createGeometry()
    {
        // TODO: Hardcoded..
        unsigned int tessU = 180;
        unsigned int tessV = 90;
        float        radius = 0.3f;
        float        maxTheta = M_PIf;

        vertices.reserve((tessU + 1) * tessV);
        indices.reserve(6 * tessU * (tessV - 1));

        float phi_step = 2.0f * M_PIf / (float)tessU;
        float theta_step = maxTheta / (float)(tessV - 1);

        // Latitudinal rings.
        // Starting at the south pole going upwards on the y-axis.
        for (unsigned int latitude = 0; latitude < tessV; ++latitude) // theta angle
        {
            float theta = (float)latitude * theta_step;
            float sinTheta = sinf(theta);
            float cosTheta = cosf(theta);

            float texv = (float)latitude / (float)(tessV - 1); // Range [0.0f, 1.0f]

            // Generate vertices along the latitudinal rings.
            // On each latitude there are tessU + 1 vertices.
            // The last one and the first one are on identical positions, but have different texture coordinates!
            // Note that each second triangle connected to the two poles has zero area if the sphere is closed.
            // But since this is also used for open spheres (maxTheta < 1.0f) this is required.
            for (unsigned int longitude = 0; longitude <= tessU; ++longitude) // phi angle
            {
                float phi = (float)longitude * phi_step;
                float sinPhi = sinf(phi);
                float cosPhi = cosf(phi);

                float texu = (float)longitude / (float)tessU; // Range [0.0f, 1.0f]

                // Unit sphere coordinates are the normals.
                float3 normal = make_float3(cosPhi * sinTheta,
                                            cosTheta,           // -y to start at the south pole.
                                            sinPhi * sinTheta);
                float3 vertex;
                vertex = normal * radius;
                vertices.push_back(vertex);
				normals.push_back(normal);
            }
        }

        // We have generated tessU + 1 vertices per latitude.
        const unsigned int columns = tessU + 1;

        // Calculate indices.
        for (unsigned int latitude = 0; latitude < tessV - 1; ++latitude)
        {
            for (unsigned int longitude = 0; longitude < tessU; ++longitude)
            {
                indices.push_back(latitude * columns + longitude);  // lower left
                indices.push_back(latitude * columns + longitude + 1);  // lower right
                indices.push_back((latitude + 1) * columns + longitude + 1);  // upper right 

                indices.push_back((latitude + 1) * columns + longitude + 1);  // upper right 
                indices.push_back((latitude + 1) * columns + longitude);  // upper left
                indices.push_back(latitude * columns + longitude);  // lower left
            }
        }
    }
};