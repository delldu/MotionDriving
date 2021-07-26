#include <torch/extension.h>

#include <iostream>

// void do_leftsv((const float *)x.data_ptr(), (float *)y.data_ptr(), int n)
// {

// }

void leftsv(const torch::Tensor &x, torch::Tensor &y)
{

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("leftsv", &leftsv, "leftsv document ...");
}
