#include <iostream>
#include <torch/extension.h>


// def set_reference(image, reference, target):
//     """Reference image: Tensor (1x3xHxW format, range: 0, 255, uint8"""
//     # xxxx8888, get_subwindow -- via reference and target
//     r = int(target[0])
//     c = int(target[1])
//     h = int(target[2])
//     w = int(target[3])

//     height = int(reference.size(2))
//     width = int(reference.size(3))
//     image.image_height = height
//     image.image_width = width

//     image.set_target(r, c, h, w)

//     target_e = get_scale_size(h, w)
//     z_crop = get_subwindow(reference, r, c, image.template_size, target_e)


// def get_subwindow(
//     image, target_rc: int, target_cc: int, target_size: int, search_size: int):
//     # batch = int(image.size(0))
//     # chan = int(image.size(1))
//     height = int(image.size(2))
//     width = int(image.size(3))

//     x1, x2, left_pad, right_pad = get_range_pad(target_cc, search_size, width)
//     y1, y2, top_pad, bottom_pad = get_range_pad(target_rc, search_size, height)

//     # padding_left,padding_right, padding_top, padding_bottom
//     big = F.pad(image, (left_pad, right_pad, top_pad, bottom_pad))

//     patch = big[:, :, y1 : y2 + 1, x1 : x2 + 1]

//     return F.interpolate(patch, size=(target_size, target_size), mode="nearest")

int64_t get_scale_size(int64_t h, int64_t w)
{
    double pad = (w + h) * 0.5;
    double sz2 = (w + pad) * (h + pad);

    return (int64_t)sqrt(sz2);
}

std::vector<int64_t> get_range_pad(int64_t y, int64_t d, int64_t max)
{
    std::vector<int64_t> result;
    int64_t y1 = y - d/2;
    int64_t y2 = y1 + d - 1;
    int64_t pad1 = (-y1 > 0)? -y1 : 0;
    int64_t pad2 = (y2 - max + 1 > 0)? y2 - max + 1: 0;
    y1 += pad1;
    y2 += pad2;

    result.push_back(y1);
    result.push_back(y2);
    result.push_back(pad1);
    result.push_back(pad2);

    return result;
}

torch::Tensor subwindow(const torch::Tensor &image, torch::Tensor position)
{
  // 1. image.dim() == 4 && kNearest mode
  // 2. position position.dim() == 1 with 4 elements, rc, cc, template_size, search_size

  float *data = position.data_ptr<float>();

  int64_t height = image.size(2);
  int64_t width = image.size(3);

  int64_t target_e = get_scale_size(height, width);

  int64_t target_rc = (int64_t)data[0];
  int64_t target_cc = (int64_t)data[1];
  int64_t template_size = (int64_t)data[2];
  int64_t search_size = (int64_t)data[3];

  // y1, y2, pad1, pad2
  std::vector<int64_t> top_bottom_pad = get_range_pad(target_rc, template_size, search_size);
  std::vector<int64_t> left_right_pad = get_range_pad(target_cc, template_size, search_size);

  // Padding, F.pad(image, (left_pad, right_pad, top_pad, bottom_pad))
  namespace F = torch::nn::functional;
  std::vector<int64_t> padding;
  padding.push_back(left_right_pad.at(2)); // left
  padding.push_back(left_right_pad.at(3)); // right
  padding.push_back(top_bottom_pad.at(2)); // top
  padding.push_back(top_bottom_pad.at(3)); // bottom
  torch::Tensor pad_data = F::pad(image, F::PadFuncOptions(padding).mode(torch::kReplicate));

  // Slice, pad_data[:, :, y1 : y2 + 1, x1 : x2 + 1]
  int64_t y1 = top_bottom_pad.at(0);
  int64_t y2 = top_bottom_pad.at(1);
  int64_t x1 = left_right_pad.at(0);
  int64_t x2 = left_right_pad.at(1);
  torch::Tensor patch = pad_data.slice(2, y1, y2 + 1).slice(3, x1, x2 + 1);

  // Sample, F.interpolate(patch, size=(target_size, target_size), mode="nearest")
  std::vector<int64_t> output_size;
  output_size.push_back(template_size);
  output_size.push_back(template_size);

  return torch::upsample_nearest2d(pad_data, output_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("subwindow", subwindow, "Subwindow Function");
}
