#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <optix.h>
#include <glad/glad.h>

#include <iostream>
#include <sstream>
#include <string>

#define GL_CHECK( call )                                                       \
    do                                                                         \
    {                                                                          \
        call;                                                                  \
        glCheck( #call, __FILE__, __LINE__ );                                  \
    } while( false )
#define GL_CHECK_ERRORS() glCheckErrors( __FILE__, __LINE__ )
#define CUDA_CHECK(call)  cudaCheck(call, #call, __FILE__, __LINE__)
#define CUDA_SYNC_CHECK() cudaSyncCheck( __FILE__, __LINE__ )
#define OPTIX_CHECK(call) optixCheck(call, #call, __FILE__, __LINE__)
#define OPTIX_CHECK_LOG(call)                                                  \
    do                                                                         \
    {                                                                          \
        char   LOG[2048];                                                      \
        size_t LOG_SIZE = sizeof( LOG );                                       \
        optixCheckLog( call, LOG, sizeof( LOG ), LOG_SIZE, #call,              \
                                __FILE__, __LINE__ );                          \
    } while( false )

inline void cudaCheck(cudaError_t error, const char* call, const char* file, unsigned int line)
{
    if (error != cudaSuccess)
    {
        std::stringstream ss;
        ss << "CUDA call (" << call << " ) failed with error: '"
            << cudaGetErrorString(error) << "' (" << file << ":" << line << ")\n";
        throw std::runtime_error(ss.str());
    }
}

inline void cudaSyncCheck(const char* file, unsigned int line)
{
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::stringstream ss;
        ss << "CUDA error on synchronize with error '"
            << cudaGetErrorString(error) << "' (" << file << ":" << line << ")\n";
        throw std::runtime_error(ss.str().c_str());
    }
}

inline void optixCheck(OptixResult res, const char* call, const char* file, unsigned int line)
{
    if (res != OPTIX_SUCCESS)
    {
        std::stringstream ss;
        ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\n";
        throw std::runtime_error(ss.str());
    }
}

inline void optixCheckLog(OptixResult  res,
    const char* log,
    size_t       sizeof_log,
    size_t       sizeof_log_returned,
    const char* call,
    const char* file,
    unsigned int line)
{
    if (res != OPTIX_SUCCESS)
    {
        std::stringstream ss;
        ss << "Optix call '" << call << "' failed: " << file << ':' << line << ")\nLog:\n"
            << log << (sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "") << '\n';
        throw std::runtime_error(ss.str());
    }
}

inline const char* getGLErrorString(GLenum error)
{
    switch (error)
    {
    case GL_NO_ERROR:
        return "No error";
    case GL_INVALID_ENUM:
        return "Invalid enum";
    case GL_INVALID_VALUE:
        return "Invalid value";
    case GL_INVALID_OPERATION:
        return "Invalid operation";
        //case GL_STACK_OVERFLOW:      return "Stack overflow";
        //case GL_STACK_UNDERFLOW:     return "Stack underflow";
    case GL_OUT_OF_MEMORY:
        return "Out of memory";
        //case GL_TABLE_TOO_LARGE:     return "Table too large";
    default:
        return "Unknown GL error";
    }
}

inline void glCheck(const char* call, const char* file, unsigned int line)
{
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
    {
        std::stringstream ss;
        ss << "GL error " << getGLErrorString(err) << " at " << file << "("
            << line << "): " << call << '\n';
        std::cerr << ss.str() << std::endl;
        throw std::runtime_error(ss.str().c_str());
    }
}

inline void glCheckErrors(const char* file, unsigned int line)
{
    GLenum err = glGetError();
    if (err != GL_NO_ERROR)
    {
        std::stringstream ss;
        ss << "GL error " << getGLErrorString(err) << " at " << file << "("
            << line << ")";
        std::cerr << ss.str() << std::endl;
        throw std::runtime_error(ss.str().c_str());
    }
}