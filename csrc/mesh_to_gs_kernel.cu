#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <torch/extension.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <random>

#include "utils.cuh"

namespace cg = cooperative_groups;

__global__ void count_gaussians_kernel(
    const float* vertices,
    const int* faces,
    unsigned int* counts,
    int num_faces,
    float k,
    float R
) {
    int face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= num_faces) return;

    // Load face vertex indices
    int i0 = faces[face_idx * 3 + 0];
    int i1 = faces[face_idx * 3 + 1];
    int i2 = faces[face_idx * 3 + 2];

    // Load vertex positions
    float3 v0 = {vertices[i0 * 3 + 0], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]};
    float3 v1 = {vertices[i1 * 3 + 0], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]};
    float3 v2 = {vertices[i2 * 3 + 0], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]};

    // Calculate face area
    float3 cross_prod = cross(v1 - v0, v2 - v0);
    float area = 0.5f * length(cross_prod);

    if (area < 1e-12f) {
        counts[face_idx] = 0;
        return;
    }
    
    float A_target = (1.0f / R) * (1.0f / R);
    unsigned int n_gaussians = static_cast<unsigned int>(ceilf(k * (area / A_target)));
    counts[face_idx] = n_gaussians;
}


__global__ void generate_gaussians_kernel(
    // Output arrays
    float* out_xyz,
    float* out_colors,
    float* out_scales,
    float* out_rotations,
    float* out_opacities,
    // Input mesh data
    const float* vertices,
    const int* faces,
    // Per-face gaussian counts and offsets
    const unsigned int* counts,
    const unsigned int* offsets,
    // Optional color data
    const float* vertex_colors, // can be nullptr
    const float* uvs,           // can be nullptr
    cudaTextureObject_t texture, // can be 0
    // Parameters
    int num_faces,
    float k,
    float R,
    float epsilon,
    float min_scale,
    float max_scale,
    float opacity_val,
    unsigned long long seed
) {
    int face_idx = blockIdx.x;
    int thread_id = threadIdx.x;
    cg::thread_block block = cg::this_thread_block();

    unsigned int n_gaussians_in_face = counts[face_idx];
    if (n_gaussians_in_face == 0) return;
    
    unsigned int base_offset = offsets[face_idx];

    // --- Load all necessary face data into shared memory for performance ---
    __shared__ int smem_indices[3];
    __shared__ float3 smem_verts[3];
    __shared__ float3 smem_colors[3];
    __shared__ float2 smem_uvs[3];

    if (thread_id == 0) {
        // Load vertex indices
        smem_indices[0] = faces[face_idx * 3 + 0];
        smem_indices[1] = faces[face_idx * 3 + 1];
        smem_indices[2] = faces[face_idx * 3 + 2];

        // Load vertex positions using indices
        smem_verts[0] = {vertices[smem_indices[0] * 3 + 0], vertices[smem_indices[0] * 3 + 1], vertices[smem_indices[0] * 3 + 2]};
        smem_verts[1] = {vertices[smem_indices[1] * 3 + 0], vertices[smem_indices[1] * 3 + 1], vertices[smem_indices[1] * 3 + 2]};
        smem_verts[2] = {vertices[smem_indices[2] * 3 + 0], vertices[smem_indices[2] * 3 + 1], vertices[smem_indices[2] * 3 + 2]};

        // Load vertex colors if available
        if (vertex_colors) {
            smem_colors[0] = {vertex_colors[smem_indices[0] * 3 + 0], vertex_colors[smem_indices[0] * 3 + 1], vertex_colors[smem_indices[0] * 3 + 2]};
            smem_colors[1] = {vertex_colors[smem_indices[1] * 3 + 0], vertex_colors[smem_indices[1] * 3 + 1], vertex_colors[smem_indices[1] * 3 + 2]};
            smem_colors[2] = {vertex_colors[smem_indices[2] * 3 + 0], vertex_colors[smem_indices[2] * 3 + 1], vertex_colors[smem_indices[2] * 3 + 2]};
        } 
        // Load UVs if available (and no vertex colors)
        else if (uvs) {
            smem_uvs[0] = {uvs[smem_indices[0] * 2 + 0], uvs[smem_indices[0] * 2 + 1]};
            smem_uvs[1] = {uvs[smem_indices[1] * 2 + 0], uvs[smem_indices[1] * 2 + 1]};
            smem_uvs[2] = {uvs[smem_indices[2] * 2 + 0], uvs[smem_indices[2] * 2 + 1]};
        }
    }
    cg::sync(block);

    float3 v0 = smem_verts[0];
    float3 v1 = smem_verts[1];
    float3 v2 = smem_verts[2];

    // --- Compute common face properties using data from shared memory ---
    float3 face_normal = normalize(cross(v1 - v0, v2 - v0));
    
    // Rotation
    float3 edges[3] = {v1 - v0, v2 - v1, v0 - v2};
    float3 best_tangent;
    float best_len = -1.0f;
    for (int i=0; i<3; ++i) {
        float3 proj = edges[i] - dot(edges[i], face_normal) * face_normal;
        float l = length(proj);
        if (l > best_len) {
            best_len = l;
            best_tangent = normalize(proj);
        }
    }
    float3 bitangent = normalize(cross(face_normal, best_tangent));
    float3 Rmat_cols[3] = {best_tangent, bitangent, face_normal};
    float4 quat = rot_to_quat(Rmat_cols);

    // Scaling
    float edge_lengths[3] = {length(v1-v0), length(v2-v1), length(v0-v2)};
    float max_edge_len = fmaxf(edge_lengths[0], fmaxf(edge_lengths[1], edge_lengths[2]));
    float area = 0.5f * length(cross(v1 - v0, v2 - v0));
    float rf = (max_edge_len * max_edge_len) / fmaxf(4.0f * area, 1e-12f);
    rf = fmaxf(rf, 1e-6f);

    float s1 = sqrtf(rf / (M_PI * k)) * (1.0f / R);
    float s2 = 1.0f / (sqrtf(M_PI * k * rf) * R);
    s1 = fminf(fmaxf(s1, min_scale), max_scale);
    s2 = fminf(fmaxf(s2, min_scale), max_scale);
    float3 scale = {s1, s2, epsilon};

    // --- Generate Gaussians in parallel for this face ---
    curandState_t rand_state;
    curand_init(seed, face_idx, thread_id, &rand_state);
    
    for (int i = thread_id; i < n_gaussians_in_face; i += blockDim.x) {
        unsigned int out_idx = base_offset + i;

        // Barycentric sampling
        float r1 = curand_uniform(&rand_state);
        float r2 = curand_uniform(&rand_state);
        float u = 1.0f - sqrtf(r1);
        float v = r2 * sqrtf(r1);
        float w = 1.0f - u - v;
        
        // Position
        float3 pos = u * v0 + v * v1 + w * v2;
        out_xyz[out_idx * 3 + 0] = pos.x;
        out_xyz[out_idx * 3 + 1] = pos.y;
        out_xyz[out_idx * 3 + 2] = pos.z;

        // Color (using data from shared memory)
        float3 color = {0.5f, 0.5f, 0.5f}; // Default gray
        if (vertex_colors) {
            float3 c0 = smem_colors[0];
            float3 c1 = smem_colors[1];
            float3 c2 = smem_colors[2];
            color = u*c0 + v*c1 + w*c2;
        } else if (uvs && texture) {
            float2 uv0 = smem_uvs[0];
            float2 uv1 = smem_uvs[1];
            float2 uv2 = smem_uvs[2];
            float2 uv_sample = u*uv0 + v*uv1 + w*uv2;
            float4 tex_color = tex2D<float4>(texture, uv_sample.x, 1.0f - uv_sample.y);
            color = {tex_color.x, tex_color.y, tex_color.z};
        }
        
        out_colors[out_idx * 3 + 0] = color.x;
        out_colors[out_idx * 3 + 1] = color.y;
        out_colors[out_idx * 3 + 2] = color.z;
        
        // Scale, Rotation, Opacity
        out_scales[out_idx * 3 + 0] = scale.x;
        out_scales[out_idx * 3 + 1] = scale.y;
        out_scales[out_idx * 3 + 2] = scale.z;

        out_rotations[out_idx * 4 + 0] = quat.w;
        out_rotations[out_idx * 4 + 1] = quat.x;
        out_rotations[out_idx * 4 + 2] = quat.y;
        out_rotations[out_idx * 4 + 3] = quat.z;

        out_opacities[out_idx] = opacity_val;
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mesh_to_gaussians_cuda(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor vertex_colors, // Optional
    torch::Tensor uvs,           // Optional
    torch::Tensor texture,       // Optional
    float R,
    float k,
    float epsilon,
    float min_scale,
    float max_scale,
    unsigned long long seed
) {
    if (!vertices.is_cuda()) throw std::runtime_error("Vertices must be on CUDA device");
    if (!faces.is_cuda()) throw std::runtime_error("Faces must be on CUDA device");
    if (vertices.dtype() != torch::kFloat32) throw std::runtime_error("Vertices must be float32");
    if (faces.dtype() != torch::kInt32) throw std::runtime_error("Faces must be int32");

    const int num_faces = faces.size(0);
    const int threads_per_block = 256;

    // 1. Count gaussians per face
    torch::Tensor counts = torch::zeros({num_faces}, torch::dtype(torch::kUInt32).device(torch::kCUDA));
    int num_blocks = (num_faces + threads_per_block - 1) / threads_per_block;
    count_gaussians_kernel<<<num_blocks, threads_per_block>>>(
        vertices.data_ptr<float>(),
        faces.data_ptr<int>(),
        counts.data_ptr<unsigned int>(),
        num_faces,
        k, R
    );
    
    // 2. Compute prefix sum to get offsets and total count
    torch::Tensor offsets = torch::zeros_like(counts);
    thrust::device_ptr<unsigned int> counts_ptr(counts.data_ptr<unsigned int>());
    thrust::device_ptr<unsigned int> offsets_ptr(offsets.data_ptr<unsigned int>());
    thrust::exclusive_scan(counts_ptr, counts_ptr + num_faces, offsets_ptr);
    
    // [FIXED] Cast to 64-bit int before sum to prevent potential overflow
    int64_t total_gaussians = counts.to(torch::kInt64).sum().item<int64_t>();
    if (total_gaussians == 0) {
        return {torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }

    // 3. Allocate output tensors
    auto opts = torch::dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor out_xyz = torch::empty({total_gaussians, 3}, opts);
    torch::Tensor out_colors = torch::empty({total_gaussians, 3}, opts);
    torch::Tensor out_scales = torch::empty({total_gaussians, 3}, opts);
    torch::Tensor out_rotations = torch::empty({total_gaussians, 4}, opts);
    torch::Tensor out_opacities = torch::empty({total_gaussians}, opts).unsqueeze(1);

    // 4. Handle texture object if provided
    cudaTextureObject_t texture_obj = 0;
    cudaArray* cuArray = nullptr;

    if (texture.defined() && texture.numel() > 0) {
        if (!(texture.is_cuda() && texture.dtype() == torch::kFloat32)) throw std::runtime_error("Texture must be a float32 CUDA tensor.");
        if (texture.dim() != 3) throw std::runtime_error("Texture must have 3 dimensions (H, W, C)");

        const int height = texture.size(0);
        const int width = texture.size(1);
        const int channels = texture.size(2);
        
        torch::Tensor texture_rgba;
        if (channels == 3) {
            auto ones = torch::ones({height, width, 1}, texture.options());
            texture_rgba = torch::cat({texture, ones}, 2).contiguous();
        } else if (channels == 4) {
            texture_rgba = texture.contiguous();
        } else {
            throw std::runtime_error("Texture must have 3 (RGB) or 4 (RGBA) channels.");
        }

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
        cudaMallocArray(&cuArray, &channelDesc, width, height, 0);
        cudaMemcpy2DToArray(
            cuArray, 0, 0,
            texture_rgba.data_ptr(),
            width * sizeof(float4),
            width * sizeof(float4),
            height,
            cudaMemcpyDeviceToDevice
        );
        
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;

        cudaCreateTextureObject(&texture_obj, &resDesc, &texDesc, NULL);
    }

    // 5. Launch generation kernel
    // [FIXED] Correctly calculate inverse sigmoid for opacity
    float opacity_val = logf(0.99f / (1.0f - 0.99f));

    generate_gaussians_kernel<<<num_faces, threads_per_block>>>(
        out_xyz.data_ptr<float>(), out_colors.data_ptr<float>(),
        out_scales.data_ptr<float>(), out_rotations.data_ptr<float>(),
        out_opacities.data_ptr<float>(),
        vertices.data_ptr<float>(), faces.data_ptr<int>(),
        counts.data_ptr<unsigned int>(), offsets.data_ptr<unsigned int>(),
        vertex_colors.defined() ? vertex_colors.data_ptr<float>() : nullptr,
        uvs.defined() ? uvs.data_ptr<float>() : nullptr,
        texture_obj,
        num_faces, k, R, epsilon, min_scale, max_scale, opacity_val,
        seed // [FIXED] Use seed passed as argument
    );
    
    // A synchronization is good practice here to ensure all CUDA errors are caught.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    // 6. Clean up CUDA resources
    if (texture_obj) {
        cudaDestroyTextureObject(texture_obj);
    }
    if (cuArray) {
        cudaFreeArray(cuArray);
    }
    
    // Ensure output opacities have the correct shape [N, 1]
    return {out_xyz, out_colors, out_scales, out_rotations, out_opacities.squeeze(-1)};
}