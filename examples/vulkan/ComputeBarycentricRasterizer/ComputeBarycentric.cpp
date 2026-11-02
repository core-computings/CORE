#include "ComputeBarycentric.h"

#include <cstring>
#include <stdexcept>

namespace core {
namespace example {

ComputeBarycentric::ComputeBarycentric(core::vulkan::VulkanContext* context,
                                       const int width, const int height)
    : ComputeBarycentric(context, width, height, WorkgroupSize{}) {}

ComputeBarycentric::ComputeBarycentric(core::vulkan::VulkanContext* context,
                                       int width, int height,
                                       WorkgroupSize workgroup_size)
    : core::vulkan::VulkanCompute(context),
      width_(width),
      height_(height),
      workgroup_size_(workgroup_size) {
  if (width_ <= 0 || height_ <= 0 || width_ > 16384 || height_ > 16384) {
    throw std::invalid_argument(
        "Rasterizer dimensions must be between 1 and 16384");
  }
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(context->physical_device, &properties);
  const VkPhysicalDeviceLimits& limits = properties.limits;
  if (workgroup_size_.x == 0 || workgroup_size_.y == 0 ||
      workgroup_size_.x > limits.maxComputeWorkGroupSize[0] ||
      workgroup_size_.y > limits.maxComputeWorkGroupSize[1] ||
      static_cast<uint64_t>(workgroup_size_.x) * workgroup_size_.y >
          limits.maxComputeWorkGroupInvocations ||
      (static_cast<uint32_t>(width_) - 1) / workgroup_size_.x + 1 >
          limits.maxComputeWorkGroupCount[0] ||
      (static_cast<uint32_t>(height_) - 1) / workgroup_size_.y + 1 >
          limits.maxComputeWorkGroupCount[1]) {
    throw std::invalid_argument("Unsupported rasterizer workgroup size");
  }
  CreateBuffers();
}

void ComputeBarycentric::Init() {
  core::vulkan::VulkanCompute::Init();

  rasterized.MapData([this](void* data) {
    std::memset(data, 0, static_cast<size_t>(rasterized.Size()));
  });

  CreateStorageBufferDescriptorSet(0, rasterized);

  vkUpdateDescriptorSets(context_->logical_device,
                         static_cast<uint32_t>(writes_.size()), writes_.data(),
                         0, nullptr);
}

void ComputeBarycentric::Run(const VkCommandBuffer command_buffer) {
  Run(command_buffer, Triangle{.a = glm::ivec2(300, 800),
                               .b = glm::ivec2(500, 100),
                               .c = glm::ivec2(700, 800)});
}

void ComputeBarycentric::Run(VkCommandBuffer command_buffer,
                             const Triangle& triangle) {
  for (const glm::ivec2& vertex : {triangle.a, triangle.b, triangle.c}) {
    if (vertex.x < -16383 || vertex.x > 16383 || vertex.y < -16383 ||
        vertex.y > 16383) {
      throw std::invalid_argument(
          "Rasterizer coordinates must be between -16383 and 16383");
    }
  }
  const PushConstants parameters{.width = width_,
                                 .height = height_,
                                 .vertex_a = triangle.a,
                                 .vertex_b = triangle.b,
                                 .vertex_c = triangle.c};
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                       nullptr, 0, nullptr);

  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                          &descriptor_set_, 0, nullptr);
  vkCmdPushConstants(command_buffer, pipeline_layout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     static_cast<uint32_t>(sizeof(parameters)), &parameters);

  const uint32_t group_x =
      (static_cast<uint32_t>(width_) - 1) / workgroup_size_.x + 1;
  const uint32_t group_y =
      (static_cast<uint32_t>(height_) - 1) / workgroup_size_.y + 1;
  vkCmdDispatch(command_buffer, group_x, group_y, 1);
}

std::vector<core::vulkan::BindingInfo> ComputeBarycentric::GetBindingInfo() const {
  return {{.binding = 0,
           .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
           .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT}};
}

std::vector<VkPushConstantRange> ComputeBarycentric::GetPushConstantRanges()
    const {
  return {{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
           .offset = 0,
           .size = static_cast<uint32_t>(sizeof(PushConstants))}};
}

const std::vector<uint32_t>& ComputeBarycentric::LoadShaderCode() const {
  static const std::vector<uint32_t> shader_code =
#include "Barycentric.comp.spv"
      ;
  return shader_code;
}

core::vulkan::ComputeSpecialization ComputeBarycentric::GetSpecialization()
    const {
  const uint32_t values[] = {workgroup_size_.x, workgroup_size_.y};
  core::vulkan::ComputeSpecialization specialization{
      .entries = {{.constantID = 0, .offset = 0, .size = sizeof(uint32_t)},
                  {.constantID = 1,
                   .offset = sizeof(uint32_t),
                   .size = sizeof(uint32_t)}},
      .data = std::vector<std::byte>(sizeof(values))};
  std::memcpy(specialization.data.data(), values, sizeof(values));
  return specialization;
}

void ComputeBarycentric::CreateBuffers() {
  const VkDeviceSize size = static_cast<VkDeviceSize>(width_) *
                            static_cast<VkDeviceSize>(height_) * sizeof(int);
  rasterized = core::vulkan::VulkanBuffer(
      context_, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

}  // namespace example
}  // namespace core