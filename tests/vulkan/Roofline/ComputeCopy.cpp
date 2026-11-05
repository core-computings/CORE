#include "ComputeCopy.h"

#include <cstring>
#include <stdexcept>

namespace core {
namespace vulkan {

ComputeCopy::ComputeCopy(VulkanContext* context, VulkanBuffer& src, VulkanBuffer& dst,
                         const int count)
    : VulkanCompute(context),
      src_buffer_(src),
      dst_buffer_(dst),
      uniform_buffer_(context, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
      uniform_data_{.count = count} {}

void ComputeCopy::Init() {
  VulkanCompute::Init();

  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  BindBuffers(src_buffer_, dst_buffer_);
}

void ComputeCopy::BindBuffers(const VulkanBuffer& src,
                              const VulkanBuffer& dst) {
  if (uniform_data_.count <= 0 ||
      src.Size() < static_cast<VkDeviceSize>(uniform_data_.count) * 16 ||
      dst.Size() < static_cast<VkDeviceSize>(uniform_data_.count) * 16 ||
      src.buffer == dst.buffer) {
    throw std::invalid_argument(
        "Compute copy requires distinct buffers with sufficient capacity");
  }
  CreateStorageBufferDescriptorSet(1, src);
  CreateStorageBufferDescriptorSet(2, dst);

  vkUpdateDescriptorSets(context_->logical_device,
                         static_cast<uint32_t>(writes_.size()), writes_.data(),
                         0, nullptr);
}

void ComputeCopy::Run(const VkCommandBuffer command_buffer) {
  uniform_buffer_.MapData(
      [this](void* data) { memcpy(data, &uniform_data_, sizeof(UniformData)); });

  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);

  const uint32_t group_count =
      (static_cast<uint32_t>(uniform_data_.count) + kLocalSizeX - 1) / kLocalSizeX;
  vkCmdDispatch(command_buffer, group_count, 1, 1);
}

std::vector<BindingInfo> ComputeCopy::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}

const std::vector<uint32_t>& ComputeCopy::LoadShaderCode() const {
  static const std::vector<uint32_t> shader_code =
#include "ComputeCopy.comp.spv"
      ;
  return shader_code;
}

}  // namespace vulkan
}  // namespace core
