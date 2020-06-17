#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> rasterize_cuda_forward(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor shape);

std::vector<torch::Tensor> rasterize_cuda_backward(
    torch::Tensor volume,
    torch::Tensor grad_volume, 
    torch::Tensor vertices,    
    torch::Tensor faces,
    torch::Tensor shape);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> rasterize_forward(
    torch::Tensor vertices,
    torch::Tensor faces,
    torch::Tensor shape) {
  CHECK_INPUT(vertices);
  CHECK_INPUT(faces);
  CHECK_INPUT(shape); 
 

  return rasterize_cuda_forward(vertices, faces, shape);
}



std::vector<torch::Tensor> rasterize_backward(
    torch::Tensor volume,
    torch::Tensor grad_volume, 
    torch::Tensor vertices,    
    torch::Tensor faces,
    torch::Tensor shape) {
  CHECK_INPUT(grad_volume);  
  CHECK_INPUT(vertices);  
  CHECK_INPUT(faces);  
  CHECK_INPUT(shape);  
  CHECK_INPUT(volume);  

  return rasterize_cuda_backward(volume, grad_volume, vertices, faces, shape);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rasterize_forward, "Rasterize forward (CUDA)");
  m.def("backward", &rasterize_backward, "Rasterize backward (CUDA)");
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("forward", &rasterize_forward, "Rasterize forward (CUDA)");
// }
