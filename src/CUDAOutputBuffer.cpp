#include "CUDAOutputBuffer.h"

CUDAOutputBuffer::CUDAOutputBuffer(int32_t width, int32_t height)
{
    int current_device, is_display_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    CUDA_CHECK(cudaDeviceGetAttribute(&is_display_device, cudaDevAttrKernelExecTimeout, current_device));
    if (!is_display_device)
    {
        throw std::runtime_error(
            "GL interop is only available on display device, please use display device for optimal "
            "performance.  Alternatively you can disable GL interop with --no-gl-interop and run with "
            "degraded performance."
        );
    }

    resize(width, height);
}

CUDAOutputBuffer::~CUDAOutputBuffer()
{
}

void CUDAOutputBuffer::resize(int32_t width, int32_t height)
{
    if (m_width == width && m_height == height)
        return;

    m_width = width;
    m_height = height;

    // GL buffer gets resized below
    GL_CHECK(glGenBuffers(1, &m_pbo));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_pbo));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(uchar3) * m_width * m_height, nullptr, GL_STREAM_DRAW));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0u));

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
        &m_cuda_gfx_resource,
        m_pbo,
        cudaGraphicsMapFlagsWriteDiscard
    ));

    if (!m_host_pixels.empty())
        m_host_pixels.resize(m_width * m_height);
}

uchar3* CUDAOutputBuffer::map()
{
    size_t buffer_size = 0u;
    CUDA_CHECK(cudaGraphicsMapResources(1, &m_cuda_gfx_resource, m_stream));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
        reinterpret_cast<void**>(&m_device_pixels),
        &buffer_size,
        m_cuda_gfx_resource
    ));

    return m_device_pixels;
}

void CUDAOutputBuffer::unmap()
{
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cuda_gfx_resource, m_stream));
}

GLuint CUDAOutputBuffer::getPBO()
{
	return m_pbo;
}