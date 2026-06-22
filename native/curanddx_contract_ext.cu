#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>

#include <curanddx.hpp>

#ifndef CURANDDX_TARGET_SM
#define CURANDDX_TARGET_SM 900
#endif

namespace {

constexpr unsigned int kSubsequences = 65536;
constexpr int kThreads = 256;

using PhiloxRng = decltype(curanddx::Generator<curanddx::philox4_32>() +
                           curanddx::PhiloxRounds<10>() +
                           curanddx::SM<CURANDDX_TARGET_SM>() +
                           curanddx::Thread());

__device__ PhiloxRng make_rng(unsigned long long seed, unsigned long long offset4, int64_t group_idx) {
  const auto absolute_group = offset4 + static_cast<unsigned long long>(group_idx);
  return PhiloxRng(seed, absolute_group % kSubsequences, absolute_group / kSubsequences);
}

__global__ void philox_raw_u32_kernel(
    uint4* __restrict__ out,
    int64_t groups,
    unsigned long long seed,
    unsigned long long offset4) {
  int64_t group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_idx >= groups) {
    return;
  }
  auto rng = make_rng(seed, offset4, group_idx);
  out[group_idx] = rng.generate4();
}

__global__ void philox_uniform_kernel(
    float4* __restrict__ out,
    int64_t groups,
    unsigned long long seed,
    unsigned long long offset4) {
  int64_t group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_idx >= groups) {
    return;
  }
  auto rng = make_rng(seed, offset4, group_idx);
  curanddx::uniform<float> dist(0.0f, 1.0f);
  out[group_idx] = dist.generate4(rng);
}

__global__ void philox_add_uniform_kernel(
    const float4* __restrict__ x,
    float4* __restrict__ out,
    int64_t groups,
    unsigned long long seed,
    unsigned long long offset4,
    float alpha) {
  int64_t group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_idx >= groups) {
    return;
  }
  auto rng = make_rng(seed, offset4, group_idx);
  curanddx::uniform<float> dist(0.0f, 1.0f);
  const float4 u = dist.generate4(rng);
  const float4 xv = x[group_idx];
  out[group_idx] = make_float4(
      xv.x + alpha * (u.x - 0.5f),
      xv.y + alpha * (u.y - 0.5f),
      xv.z + alpha * (u.z - 0.5f),
      xv.w + alpha * (u.w - 0.5f));
}

__global__ void philox_threshold_kernel(
    unsigned char* __restrict__ mask,
    int64_t groups,
    unsigned long long seed,
    unsigned long long offset4,
    float p) {
  int64_t group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_idx >= groups) {
    return;
  }
  auto rng = make_rng(seed, offset4, group_idx);
  curanddx::uniform<float> dist(0.0f, 1.0f);
  const float4 u = dist.generate4(rng);
  const int64_t base = group_idx * 4;
  mask[base] = static_cast<unsigned char>(u.x <= p);
  mask[base + 1] = static_cast<unsigned char>(u.y <= p);
  mask[base + 2] = static_cast<unsigned char>(u.z <= p);
  mask[base + 3] = static_cast<unsigned char>(u.w <= p);
}

__global__ void philox_dropout_kernel(
    const float4* __restrict__ x,
    float4* __restrict__ out,
    unsigned char* __restrict__ mask,
    int64_t groups,
    unsigned long long seed,
    unsigned long long offset4,
    float p) {
  int64_t group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_idx >= groups) {
    return;
  }
  auto rng = make_rng(seed, offset4, group_idx);
  curanddx::uniform<float> dist(0.0f, 1.0f);
  const float4 u = dist.generate4(rng);
  const float4 xv = x[group_idx];
  const bool keep0 = u.x <= p;
  const bool keep1 = u.y <= p;
  const bool keep2 = u.z <= p;
  const bool keep3 = u.w <= p;
  const int64_t base = group_idx * 4;
  mask[base] = static_cast<unsigned char>(keep0);
  mask[base + 1] = static_cast<unsigned char>(keep1);
  mask[base + 2] = static_cast<unsigned char>(keep2);
  mask[base + 3] = static_cast<unsigned char>(keep3);
  out[group_idx] = make_float4(
      keep0 ? xv.x / p : 0.0f,
      keep1 ? xv.y / p : 0.0f,
      keep2 ? xv.z / p : 0.0f,
      keep3 ? xv.w / p : 0.0f);
}

int blocks_for_groups(int64_t groups) {
  return static_cast<int>((groups + kThreads - 1) / kThreads);
}

unsigned long long offset4_from_elements(unsigned long long element_offset) {
  TORCH_CHECK(element_offset % 4 == 0, "cuRANDDx Philox offset must be a multiple of 4 elements");
  return element_offset / 4;
}

void check_cuda_tensor(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_multiple_of_four(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.numel() % 4 == 0, name, " numel must be a multiple of 4 for cuRANDDx Philox generate4");
}

void check_probability(float p) {
  TORCH_CHECK(p > 0.0f && p <= 1.0f, "p must be in (0, 1]");
}

}  // namespace

void philox_raw_u32(torch::Tensor out, unsigned long long seed, unsigned long long offset) {
  check_cuda_tensor(out, "out");
  TORCH_CHECK(out.scalar_type() == torch::kInt32, "out must be int32");
  check_multiple_of_four(out, "out");
  const int64_t groups = out.numel() / 4;
  philox_raw_u32_kernel<<<blocks_for_groups(groups), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<uint4*>(out.data_ptr<int32_t>()), groups, seed, offset4_from_elements(offset));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void philox_uniform(torch::Tensor out, unsigned long long seed, unsigned long long offset) {
  check_cuda_tensor(out, "out");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
  check_multiple_of_four(out, "out");
  const int64_t groups = out.numel() / 4;
  philox_uniform_kernel<<<blocks_for_groups(groups), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<float4*>(out.data_ptr<float>()), groups, seed, offset4_from_elements(offset));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void philox_add_uniform(torch::Tensor x, torch::Tensor out, unsigned long long seed, unsigned long long offset, double alpha) {
  check_cuda_tensor(x, "x");
  check_cuda_tensor(out, "out");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
  TORCH_CHECK(x.numel() == out.numel(), "x and out must have the same numel");
  check_multiple_of_four(out, "out");
  const int64_t groups = out.numel() / 4;
  philox_add_uniform_kernel<<<blocks_for_groups(groups), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()),
      reinterpret_cast<float4*>(out.data_ptr<float>()),
      groups,
      seed,
      offset4_from_elements(offset),
      static_cast<float>(alpha));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void philox_threshold(torch::Tensor mask, unsigned long long seed, unsigned long long offset, double p) {
  check_cuda_tensor(mask, "mask");
  TORCH_CHECK(mask.scalar_type() == torch::kUInt8, "mask must be uint8");
  check_multiple_of_four(mask, "mask");
  check_probability(static_cast<float>(p));
  const int64_t groups = mask.numel() / 4;
  philox_threshold_kernel<<<blocks_for_groups(groups), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      mask.data_ptr<unsigned char>(), groups, seed, offset4_from_elements(offset), static_cast<float>(p));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void philox_dropout(torch::Tensor x, torch::Tensor out, torch::Tensor mask, unsigned long long seed, unsigned long long offset, double p) {
  check_cuda_tensor(x, "x");
  check_cuda_tensor(out, "out");
  check_cuda_tensor(mask, "mask");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(out.scalar_type() == torch::kFloat32, "out must be float32");
  TORCH_CHECK(mask.scalar_type() == torch::kUInt8, "mask must be uint8");
  TORCH_CHECK(x.numel() == out.numel() && x.numel() == mask.numel(), "x/out/mask numel mismatch");
  check_multiple_of_four(out, "out");
  check_probability(static_cast<float>(p));
  const int64_t groups = out.numel() / 4;
  philox_dropout_kernel<<<blocks_for_groups(groups), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const float4*>(x.data_ptr<float>()),
      reinterpret_cast<float4*>(out.data_ptr<float>()),
      mask.data_ptr<unsigned char>(),
      groups,
      seed,
      offset4_from_elements(offset),
      static_cast<float>(p));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("philox_raw_u32", &philox_raw_u32, "cuRANDDx Philox raw uint32 output");
  m.def("philox_uniform", &philox_uniform, "cuRANDDx Philox uniform output");
  m.def("philox_add_uniform", &philox_add_uniform, "cuRANDDx Philox add-uniform fused");
  m.def("philox_threshold", &philox_threshold, "cuRANDDx Philox threshold fused");
  m.def("philox_dropout", &philox_dropout, "cuRANDDx Philox dropout fused");
}
