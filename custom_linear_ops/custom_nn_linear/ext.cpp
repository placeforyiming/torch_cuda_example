#include <torch/extension.h>
#include "custom_nn_linear.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("custom_nn_linear", &CustomNNLinearCUDA);
  m.def("custom_nn_linear_backward", &CustomNNLinearBackwardCUDA);
}