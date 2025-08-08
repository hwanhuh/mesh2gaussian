#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <random>
#include "mesh_to_gs.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    std::random_device rd;
    unsigned long long default_seed = rd();

    m.def("convert", &mesh_to_gaussians_cuda, "Convert mesh to gaussians (CUDA)",
        py::arg("vertices"),
        py::arg("faces"),
        py::arg("vertex_colors"),
        py::arg("uvs"),
        py::arg("texture"),
        py::arg("R"),
        py::arg("k"),
        py::arg("epsilon"),
        py::arg("min_scale"),
        py::arg("max_scale"),
        py::arg("seed") = default_seed
    );
}