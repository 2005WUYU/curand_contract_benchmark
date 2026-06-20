#include <torch/extension.h>

#include <cstdint>

#include <curand_kernel.h>

namespace {

__global__ void philox_raw_u32_kernel(
    int32_t* __restrict__ out,
    int64_t n,
    unsigned long long seed,
    unsigned long long offset) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  curandStatePhilox4_32_10_t state;
  curand_init(seed, static_cast<unsigned long long>(idx), offset, &state);
  out[idx] = static_cast<int32_t>(curand(&state));
}

__global__ void philox_uniform_kernel(
    float* __restrict__ out,
    int64_t n,
    unsigned long long seed,
    unsigned long long offset) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  curandStatePhilox4_32_10_t state;
  curand_init(seed, static_cast<unsigned long long>(idx), offset, &state);
  out[idx] = curand_uniform(&state);
}

__global__ void philox_add_uniform_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t n,
    unsigned long long seed,
    unsigned long long offset,
    float alpha) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  curandStatePhilox4_32_10_t state;
  curand_init(seed, static_cast<unsigned long long>(idx), offset, &state);
  float u = curand_uniform(&state);
  out[idx] = x[idx] + alpha * (u - 0.5f);
}

__global__ void philox_threshold_kernel(
    unsigned char* __restrict__ mask,
    int64_t n,
    unsigned long long seed,
    unsigned long long offset,
    float p) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  curandStatePhilox4_32_10_t state;
  curand_init(seed, static_cast<unsigned long long>(idx), offset, &state);
  float u = curand_uniform(&state);
  mask[idx] = static_cast<unsigned char>(u <= p);
}

__global__ void philox_dropout_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    unsigned char* __restrict__ mask,
    int64_t n,
    unsigned long long seed,
    unsigned long long offset,
    float p) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  curandStatePhilox4_32_10_t state;
  curand_init(seed, static_cast<unsigned long long>(idx), offset, &state);
  float u = curand_uniform(&state);
  bool keep = u <= p;
  mask[idx] = static_cast<unsigned char>(keep);
  out[idx] = keep ? x[idx] / p : 0.0f;
}

int blocks_for(int64_t n, int threads) {
  return static_cast<int>((n + threads - 1) / threads);
}

void check_cuda_tensor(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void philox_raw_u32(torch::Tensor out, unsigned long long seed, unsigned long long offset) {
  check_cuda_tensor(out, "out");
  TORCH_CHECK(out.scalar_type() == torch::kInt32, "out must be int32");
  int threads = 256;
  philox_raw_u32_kernel<<<blocks_for(out.numel(), threads), threads>>>(
      out.data_ptr<int32_t>(), out.numel(), seed, offset);
}

void philox_uniform(torch::Tensor out, unsigned long long seed, unsigned long long offset) {
  check_cuda_tensor(out, "out");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
  int threads = 256;
  philox_uniform_kernel<<<blocks_for(out.numel(), threads), threads>>>(
      out.data_ptr<float>(), out.numel(), seed, offset);
}

void philox_add_uniform(torch::Tensor x, torch::Tensor out, unsigned long long seed, unsigned long long offset, double alpha) {
  check_cuda_tensor(x, "x");
  check_cuda_tensor(out, "out");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
  TORCH_CHECK(x.numel() == out.numel(), "x and out must have the same numel");
  int threads = 256;
  philox_add_uniform_kernel<<<blocks_for(out.numel(), threads), threads>>>(
      x.data_ptr<float>(), out.data_ptr<float>(), out.numel(), seed, offset, static_cast<float>(alpha));
}

void philox_threshold(torch::Tensor mask, unsigned long long seed, unsigned long long offset, double p) {
  check_cuda_tensor(mask, "mask");
  TORCH_CHECK(mask.scalar_type() == torch::kUInt8, "mask must be uint8");
  int threads = 256;
  philox_threshold_kernel<<<blocks_for(mask.numel(), threads), threads>>>(
      mask.data_ptr<unsigned char>(), mask.numel(), seed, offset, static_cast<float>(p));
}

void philox_dropout(torch::Tensor x, torch::Tensor out, torch::Tensor mask, unsigned long long seed, unsigned long long offset, double p) {
  check_cuda_tensor(x, "x");
  check_cuda_tensor(out, "out");
  check_cuda_tensor(mask, "mask");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
  TORCH_CHECK(mask.scalar_type() == torch::kUInt8, "mask must be uint8");
  TORCH_CHECK(x.numel() == out.numel() && x.numel() == mask.numel(), "x/out/mask numel mismatch");
  int threads = 256;
  philox_dropout_kernel<<<blocks_for(out.numel(), threads), threads>>>(
      x.data_ptr<float>(), out.data_ptr<float>(), mask.data_ptr<unsigned char>(),
      out.numel(), seed, offset, static_cast<float>(p));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("philox_raw_u32", &philox_raw_u32, "legacy cuRAND Device API Philox raw uint32 output");
  m.def("philox_uniform", &philox_uniform, "legacy cuRAND Device API Philox uniform output");
  m.def("philox_add_uniform", &philox_add_uniform, "legacy cuRAND Device API Philox add-uniform fused");
  m.def("philox_threshold", &philox_threshold, "legacy cuRAND Device API Philox threshold fused");
  m.def("philox_dropout", &philox_dropout, "legacy cuRAND Device API Philox dropout fused");
}
