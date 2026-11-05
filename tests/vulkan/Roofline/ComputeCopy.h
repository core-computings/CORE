#pragma once

#include "VulkanBuffer.h"
#include "VulkanCompute.h"

namespace core {
namespace vulkan {

// Pure-streaming microbenchmark: coalesced vec4 copies between two large
// storage buffers with no arithmetic. Used to measure the GPU's peak memory
// bandwidth for the roofline model.
class ComputeCopy : public VulkanCompute {
 public:
  // Must match the shader's local_size_x.
  static constexpr uint32_t kLocalSizeX = 256;

  // count is the number of vec4 (16-byte) elements to copy.
  ComputeCopy(VulkanContext* context, VulkanBuffer& src, VulkanBuffer& dst, const int count);

  void Init() override;
  void Run(const VkCommandBuffer command_buffer);
  void BindBuffers(const VulkanBuffer& src, const VulkanBuffer& dst);

 protected:
  std::vector<BindingInfo> GetBindingInfo() const override;
  const std::vector<uint32_t>& LoadShaderCode() const override;

 private:
  VulkanBuffer& src_buffer_;
  VulkanBuffer& dst_buffer_;
  VulkanBuffer uniform_buffer_;
  struct UniformData {
    int count;
  } uniform_data_;
};

}  // namespace vulkan
}  // namespace core
