#pragma once

#include <vulkan/vulkan.h>

#include "VulkanContext.h"

namespace core {
namespace vulkan {

struct SamplerOptions {
  VkFilter mag_filter = VK_FILTER_LINEAR;
  VkFilter min_filter = VK_FILTER_LINEAR;
  VkSamplerMipmapMode mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  VkSamplerAddressMode address_mode_u = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  VkSamplerAddressMode address_mode_v = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  VkSamplerAddressMode address_mode_w = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  VkBorderColor border_color = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
  float min_lod = 0.0f;
  float max_lod = VK_LOD_CLAMP_NONE;
};

class VulkanSampler {
 public:
  explicit VulkanSampler(VulkanContext* context);
  VulkanSampler(VulkanContext* context, const SamplerOptions& options);
  ~VulkanSampler();

 private:
  VulkanContext* context_ = nullptr;

 public:
  VkSampler sampler = VK_NULL_HANDLE;
};

}  // namespace vulkan
}  // namespace core