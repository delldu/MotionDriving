#include <iostream>
#include <torch/extension.h>

torch::Tensor GetSubWindow(torch::Tensor &self, torch::Tensor pos) {
  float *f = (float *)self.data_ptr();
  *f += 1.0;

  return self;
}


torch::Tensor leftsv(const torch::Tensor &self) {
  torch::Tensor U, S, V;
  std::tie(U, S, V) = self.svd();
  torch::Tensor d =
      S.pow(0.5).diag_embed(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1);

  return torch::matmul(U, d);
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.doc("Support leftsv, Siamese::GetSubWindow");
  m.def("leftsv", &leftsv, "leftsv document ...");
  m.def("GetSubWindow", GetSubWindow, "GetSubWindow Function");
}
