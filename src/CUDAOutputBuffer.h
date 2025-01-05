#pragma once

#include <glad/glad.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <vector>

#include "Exception.h"

class CUDAOutputBuffer
{
public:
	CUDAOutputBuffer(int32_t width, int32_t height);
	~CUDAOutputBuffer();

    void setStream(CUstream stream) { m_stream = stream; }

	void resize(int32_t width, int32_t height);

    uchar3* map();
    void unmap();

    int32_t width() const { return m_width; }
    int32_t height() const { return m_height; }

    GLuint getPBO();

private:
    cudaGraphicsResource* m_cuda_gfx_resource = nullptr;
    GLuint                m_pbo = 0u;
    CUstream              m_stream = 0u;
    uchar3*               m_device_pixels = nullptr;
    uchar3*               m_host_zcopy_pixels = nullptr;
    std::vector<uchar3>   m_host_pixels;

    int32_t m_width  = 0u;
    int32_t m_height = 0u;
};