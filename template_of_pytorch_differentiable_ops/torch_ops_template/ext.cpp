#include <torch/extension.h>
#include "torch_ops_template.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_ops_template", &TorchOpsTemplateCUDA);
  m.def("torch_ops_template_backward", &TorchOpsTemplateBackwardCUDA);
}