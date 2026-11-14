#pragma once

#include <optional>
#include <vector>

#include "RenderCar.h"
#include "VulkanDynamicRendering.h"
#include "VulkanImage.h"
#include "VulkanRender.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "glm/glm.hpp"

namespace core {

class RenderCarShadow : public core::vulkan::VulkanRender {
 public:
  RenderCarShadow(core::vulkan::VulkanContext* context, VkFormat depth_format, uint32_t size);
  ~RenderCarShadow() override;
  RenderCarShadow(const RenderCarShadow&) = delete;
  RenderCarShadow& operator=(const RenderCarShadow&) = delete;

  void Init() override;
  void Record(VkCommandBuffer command_buffer, const RenderCar& car,
              const glm::mat4& light_model_view_projection) const;

  VkImageView shadow_map_view() const { return shadow_map_.image_view; }
  VkSampler shadow_sampler() const { return sampler_; }

 protected:
  VkCullModeFlags SetCullMode() const override { return VK_CULL_MODE_NONE; }
  VkBool32 SetDepthTesting() const override { return VK_TRUE; }
  VkBool32 SetDepthWriting() const override { return VK_TRUE; }
  std::optional<core::vulkan::DepthBiasInfo> SetDepthBias() const override {
    return core::vulkan::DepthBiasInfo{.constant_factor = 1.25f, .slope_factor = 1.75f};
  }

  std::vector<core::vulkan::BindingInfo> GetBindingInfo() const override { return {}; }
  std::vector<VkPushConstantRange> GetPushConstantRanges() const override;
  const std::vector<uint32_t> LoadVertexShader() const override;
  const std::vector<uint32_t> LoadFragmentShader() const override { return {}; }
  std::vector<VkVertexInputBindingDescription> GetVertexBindingDescriptions() const override;
  std::vector<VkVertexInputAttributeDescription> GetVertexAttributeDescriptions() const override;

 private:
  static core::vulkan::DynamicRenderingInfo DepthOnlyInfo(VkFormat depth_format);

  VkFormat depth_format_;
  uint32_t size_;
  core::vulkan::VulkanDynamicRendering dynamic_rendering_;
  core::vulkan::VulkanImage shadow_map_;
  VkSampler sampler_ = VK_NULL_HANDLE;
};

}
