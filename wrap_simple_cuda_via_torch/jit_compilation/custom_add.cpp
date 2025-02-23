#include <torch/extension.h>
#include "custom_add.h"

void torch_launch_add(torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       int64_t n) {
    launch_add((float *)c.data_ptr(),
                (const float *)a.data_ptr(),
                (const float *)b.data_ptr(),
                n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_add",
          &torch_launch_add,
          "add kernel wrapper");
}

