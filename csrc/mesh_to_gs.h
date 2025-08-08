#pragma once
#include <torch/extension.h>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
mesh_to_gaussians_cuda(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor vertex_colors,
    torch::Tensor uvs,
    torch::Tensor texture,
    float R,
    float k,
    float epsilon,
    float min_scale,
    float max_scale,
    unsigned long long seed
);