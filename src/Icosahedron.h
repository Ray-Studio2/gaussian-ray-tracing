#pragma once

#include <vector>
#include <cmath>

#include <vector_functions.h>

class Icosahedron
{
public:
    Icosahedron() 
    { 
        const static float rr = (3 + std::sqrt(5.0f)) / (2 * std::sqrt(3.0f));
        const static float ss = 1.0f / rr;
        const static float tt = (1.0f + std::sqrt(5.0f)) / (2.0f * rr);

        vertices = {
            {-ss,  tt,  0},
            { ss,  tt,  0},
            {-ss, -tt,  0},
            { ss, -tt,  0},
            {  0, -ss,  tt},
            {  0,  ss,  tt},
            {  0, -ss, -tt},
            {  0,  ss, -tt},
            { tt,  0, -ss},
            { tt,  0,  ss},
            {-tt,  0, -ss},
            {-tt,  0,  ss}
        };

        indices = {
            0, 11, 5,  0, 5, 1,  0, 1, 7,  0, 7, 10,  0, 10, 11,
            1, 5, 9,  5, 11, 4,  11, 10, 2,  10, 7, 6,  7, 1, 8,
            3, 9, 4,  3, 4, 2,  3, 2, 6,  3, 6, 8,  3, 8, 9,
            4, 9, 5,  2, 4, 11,  6, 2, 10,  8, 6, 7,  9, 8, 1
        };
    }
    ~Icosahedron()
	{
		vertices.clear();
		indices.clear();
	}

	std::vector<float3> getVertices() const { return vertices; }
	std::vector<unsigned int> getIndices() const { return indices; }

private:
	std::vector<float3>       vertices;
	std::vector<unsigned int> indices;

};