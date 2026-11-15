#include "ComputeFma.h"

#include <cstring>
#include <stdexcept>

namespace core {
namespace vulkan {

ComputeFma::ComputeFma(VulkanContext* context, VulkanBuffer& dst, const int iterations,
                       const uint32_t group_count, const FmaPrecision precision)
    : VulkanCompute(context),
      dst_buffer_(dst),
      uniform_buffer_(context, sizeof(UniformData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
      uniform_data_{.iterations = iterations},
      group_count_(group_count),
      precision_(precision) {
  if (precision_ == FmaPrecision::kFloat16 && !context->float16_support().arithmetic) {
    throw std::invalid_argument("Float16 arithmetic is not supported");
  }
}

uint64_t ComputeFma::FlopPerIteration() const {
  return precision_ == FmaPrecision::kFloat16 ? kFloat16FlopPerIteration : kFlopPerIteration;
}

void ComputeFma::Init() {
  VulkanCompute::Init();

  CreateUniformBufferDescriptorSet(0, uniform_buffer_);
  CreateStorageBufferDescriptorSet(1, dst_buffer_);

  vkUpdateDescriptorSets(context_->logical_device, writes_.size(), writes_.data(), 0, nullptr);
}

void ComputeFma::Run(const VkCommandBuffer command_buffer) {
  uniform_buffer_.MapData(
      [this](void* data) { memcpy(data, &uniform_data_, sizeof(UniformData)); });

  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);
  vkCmdDispatch(command_buffer, group_count_, 1, 1);
}

std::vector<BindingInfo> ComputeFma::GetBindingInfo() const {
  return {{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT},
          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT}};
}

const std::vector<uint32_t>& ComputeFma::LoadShaderCode() const {
  static const std::vector<uint32_t> shader_code =
#include "ComputeFma.comp.spv"
      ;
  static const std::vector<uint32_t> float16_shader_code =
#include "ComputeFma16.comp.spv"
      ;
  return precision_ == FmaPrecision::kFloat16 ? float16_shader_code : shader_code;
}

}  // namespace vulkan
}  // namespace core
