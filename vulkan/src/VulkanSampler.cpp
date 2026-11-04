#include "VulkanSampler.h"

#include <cmath>
#include <initializer_list>
#include <stdexcept>

namespace core {
namespace vulkan {

VulkanSampler::VulkanSampler(VulkanContext* context)
    : VulkanSampler(context, SamplerOptions{}) {}

VulkanSampler::VulkanSampler(VulkanContext* context,
                             const SamplerOptions& options)
    : context_(context) {
  if ((options.mag_filter != VK_FILTER_NEAREST &&
       options.mag_filter != VK_FILTER_LINEAR) ||
      (options.min_filter != VK_FILTER_NEAREST &&
       options.min_filter != VK_FILTER_LINEAR) ||
      (options.mipmap_mode != VK_SAMPLER_MIPMAP_MODE_NEAREST &&
       options.mipmap_mode != VK_SAMPLER_MIPMAP_MODE_LINEAR)) {
    throw std::invalid_argument(
        "Sampler requires core nearest or linear filters");
  }
  for (const VkSamplerAddressMode mode :
       {options.address_mode_u, options.address_mode_v,
        options.address_mode_w}) {
    if (mode != VK_SAMPLER_ADDRESS_MODE_REPEAT &&
        mode != VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT &&
        mode != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE &&
        mode != VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER) {
      throw std::invalid_argument("Sampler requires a core address mode");
    }
  }
  if (options.border_color != VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK &&
      options.border_color != VK_BORDER_COLOR_INT_TRANSPARENT_BLACK &&
      options.border_color != VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK &&
      options.border_color != VK_BORDER_COLOR_INT_OPAQUE_BLACK &&
      options.border_color != VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE &&
      options.border_color != VK_BORDER_COLOR_INT_OPAQUE_WHITE) {
    throw std::invalid_argument("Sampler requires a core border color");
  }
  if (!std::isfinite(options.min_lod) || !std::isfinite(options.max_lod) ||
      options.min_lod < 0.0f || options.max_lod < options.min_lod) {
    throw std::invalid_argument(
        "Sampler requires a finite nonnegative ordered LOD range");
  }
  VkSamplerCreateInfo sampler_info{};
  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.magFilter = options.mag_filter;
  sampler_info.minFilter = options.min_filter;
  sampler_info.addressModeU = options.address_mode_u;
  sampler_info.addressModeV = options.address_mode_v;
  sampler_info.addressModeW = options.address_mode_w;
  sampler_info.anisotropyEnable = VK_FALSE;  // Need to enable device feature first, if set VK_TRUE
  sampler_info.maxAnisotropy = 0;
  sampler_info.borderColor = options.border_color;
  sampler_info.unnormalizedCoordinates = VK_FALSE;
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
  sampler_info.mipmapMode = options.mipmap_mode;
  sampler_info.mipLodBias = 0.0f;
  sampler_info.minLod = options.min_lod;
  sampler_info.maxLod = options.max_lod;

  VK_CHECK(vkCreateSampler(context_->logical_device, &sampler_info, nullptr, &sampler));
}

VulkanSampler::~VulkanSampler() {
  if (sampler != VK_NULL_HANDLE) {
    vkDestroySampler(context_->logical_device, sampler, nullptr);
  }
}

}  // namespace vulkan
}  // namespace core