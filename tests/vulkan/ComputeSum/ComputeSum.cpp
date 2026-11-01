#include "ComputeSum.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace core {
namespace vulkan {

ComputeSum::ComputeSum(VulkanContext* context, VulkanBuffer& src, VulkanBuffer& dst,
                       const int width, const int height)
    : VulkanCompute(context),
      src_buffer(src),
      dst_buffer(dst),
      uniform_buffer_(context, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
      uniform_data_{.width = width, .height = height} {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("Sum dimensions must be positive");
  }
  const uint64_t count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(context_->physical_device, &properties);
  if (count > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
      count * sizeof(int32_t) > src_buffer.Size() || dst_buffer.Size() < sizeof(int32_t) ||
      src_buffer.Size() > properties.limits.maxStorageBufferRange ||
      dst_buffer.Size() > properties.limits.maxStorageBufferRange ||
      (count + 4095) / 4096 > properties.limits.maxComputeWorkGroupCount[0]) {
    throw std::invalid_argument("Sum buffers or dispatch range are invalid");
  }
}

void ComputeSum::Init() {
  VulkanCompute::Init();

  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  CreateStorageBufferDescriptorSet(1, src_buffer);
  CreateStorageBufferDescriptorSet(2, dst_buffer);

  vkUpdateDescriptorSets(context_->logical_device, writes_.size(), writes_.data(), 0, nullptr);
}

void ComputeSum::Run(const VkCommandBuffer command_buffer) {
  // Copy uniform data to the uniform buffer
  uniform_buffer_.MapData(
      [this](void* data) { memcpy(data, &uniform_data_, sizeof(UniformData)); });

  // Record commands to dispatch the compute shader
  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);

  const uint32_t count = static_cast<uint32_t>(uniform_data_.width) *
                         static_cast<uint32_t>(uniform_data_.height);
  if (count <= kSmallInputLimit) {
    const uint32_t group_x = (static_cast<uint32_t>(uniform_data_.width) + 15U) / 16U;
    const uint32_t group_y = (static_cast<uint32_t>(uniform_data_.height) + 15U) / 16U;
    vkCmdDispatch(command_buffer, group_x, group_y, 1);
  } else {
    vkCmdDispatch(command_buffer, (count + 4095U) / 4096U, 1, 1);
  }
}

std::vector<BindingInfo> ComputeSum::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}

const std::vector<uint32_t>& ComputeSum::LoadShaderCode() const {
  const uint32_t count = static_cast<uint32_t>(uniform_data_.width) *
                         static_cast<uint32_t>(uniform_data_.height);
  if (count > kSmallInputLimit) {
    static const std::vector<uint32_t> large_shader_code =
#include "ComputeSumLarge.comp.spv"
        ;
    return large_shader_code;
  }
  static const std::vector<uint32_t> shader_code =
#include "ComputeSum.comp.spv"
      ;
  return shader_code;
}

}  // namespace vulkan
}  // namespace core
