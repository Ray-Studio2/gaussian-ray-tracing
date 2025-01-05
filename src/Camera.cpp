#include "Camera.h"

void Camera::UVWFrame(float3& U, float3& V, float3& W) const
{
    W = m_lookat - m_eye; // Do not normalize W -- it implies focal length
    float wlen = length(W);
    U = normalize(cross(W, m_up));
    V = normalize(cross(U, W));

    float vlen = wlen * tanf(0.5f * m_fovY * M_PIf / 180.0f);
    V *= vlen;
    float ulen = vlen * m_aspectRatio;
    U *= ulen;
}