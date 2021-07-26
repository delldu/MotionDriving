#include <torch/extension.h>
#include <iostream>


torch::Tensor leftsv(const torch::Tensor &self)
{
    torch::Tensor U, S, V;
    std::tie(U, S, V) = self.svd();
    torch::Tensor d = S.pow(0.5).diag_embed(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1);
    
    return torch::matmul(U, d);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("leftsv", &leftsv, "leftsv document ...");
}
