#pragma once

#include <array>
#include <iostream>

#include "Mat.h"
#include "VulkanBuffer.h"
#include "VulkanCompute.h"

namespace core {
namespace vulkan {

class ComputeGaussianBlur : public VulkanCompute {
 public:
  ComputeGaussianBlur(VulkanContext* context, VulkanBuffer& src, VulkanBuffer& dst, const int width,
                      const int height);

  void Init() override;
  void Run(const VkCommandBuffer command_buffer);

  VulkanBuffer& src_buffer;
  VulkanBuffer& dst_buffer;

 protected:
  std::vector<BindingInfo> GetBindingInfo() const override;
  const std::vector<uint32_t>& LoadShaderCode() const override;
  const std::string GetPipelineCache() const override;

 private:
  VulkanBuffer uniform_buffer_;
#if !defined(__ANDROID__)
  std::array<float, 9> gaussian_kernel_data_ = {0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f,
                                              0.125f, 0.0625f, 0.125f, 0.0625f};
  VulkanBuffer gaussian_kernel_;
#endif
  struct UniformData {
    int width;
    int height;
  } uniform_data_;
};

}  // namespace vulkan
}  // namespace core
