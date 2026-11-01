#pragma once

#include <iostream>

#include "Mat.h"
#include "VulkanBuffer.h"
#include "VulkanCompute.h"

namespace core {
namespace vulkan {

class ComputeSum : public VulkanCompute {
 public:
  ComputeSum(VulkanContext* context, VulkanBuffer& src, VulkanBuffer& dst, const int width,
             const int height);

  void Init() override;
  void Run(const VkCommandBuffer command_buffer);

  VulkanBuffer& src_buffer;
  VulkanBuffer& dst_buffer;

 protected:
  std::vector<BindingInfo> GetBindingInfo() const override;
  const std::vector<uint32_t>& LoadShaderCode() const override;

 private:
  static constexpr uint32_t kSmallInputLimit = 262144;
  VulkanBuffer uniform_buffer_;
  struct UniformData {
    int width;
    int height;
  } uniform_data_;
};
}  // namespace vulkan
}  // namespace core
