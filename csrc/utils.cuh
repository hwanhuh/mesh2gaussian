#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// operators
__device__ __forceinline__ float2 operator+(const float2& a, const float2& b) { return {a.x + b.x, a.y + b.y}; }
__device__ __forceinline__ float2 operator*(float s, const float2& a) { return {s * a.x, s * a.y}; }

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ __forceinline__ float3 operator*(float s, const float3& a) { return {s * a.x, s * a.y, s * a.z}; }
__device__ __forceinline__ float dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float3 cross(const float3& a, const float3& b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
__device__ __forceinline__ float length(const float3& a) { return sqrtf(dot(a, a)); }
__device__ __forceinline__ float3 normalize(const float3& a) { float l = length(a); return (l > 1e-8) ? (1.0f / l) * a : float3{0,0,0}; }

// Matrix to quaternion conversion
__device__ __forceinline__ float4 rot_to_quat(const float3 R[3]) {
    float4 q;
    float trace = R[0].x + R[1].y + R[2].z;
    if (trace > 0.0f) {
        float s = 0.5f / sqrtf(trace + 1.0f);
        q.w = 0.25f / s;
        q.x = (R[2].y - R[1].z) * s;
        q.y = (R[0].z - R[2].x) * s;
        q.z = (R[1].x - R[0].y) * s;
    } else {
        if (R[0].x > R[1].y && R[0].x > R[2].z) {
            float s = 2.0f * sqrtf(1.0f + R[0].x - R[1].y - R[2].z);
            q.w = (R[2].y - R[1].z) / s;
            q.x = 0.25f * s;
            q.y = (R[0].y + R[1].x) / s;
            q.z = (R[0].z + R[2].x) / s;
        } else if (R[1].y > R[2].z) {
            float s = 2.0f * sqrtf(1.0f + R[1].y - R[0].x - R[2].z);
            q.w = (R[0].z - R[2].x) / s;
            q.x = (R[0].y + R[1].x) / s;
            q.y = 0.25f * s;
            q.z = (R[1].z + R[2].y) / s;
        } else {
            float s = 2.0f * sqrtf(1.0f + R[2].z - R[0].x - R[1].y);
            q.w = (R[1].x - R[0].y) / s;
            q.x = (R[0].z + R[2].x) / s;
            q.y = (R[1].z + R[2].y) / s;
            q.z = 0.25f * s;
        }
    }
    return q;
}

// gamma color correction
__device__ inline float3 srgb_to_linear(float3 srgb_color) {
    return {
        powf(srgb_color.x, 2.2f),
        powf(srgb_color.y, 2.2f),
        powf(srgb_color.z, 2.2f)
    };
}

__device__ inline float3 srgb_to_linear_inverse(float3 linear_color) {
    return {
        powf(fmaxf(linear_color.x, 0.0f), 1.0f / 2.2f),
        powf(fmaxf(linear_color.y, 0.0f), 1.0f / 2.2f),
        powf(fmaxf(linear_color.z, 0.0f), 1.0f / 2.2f)
    };
}