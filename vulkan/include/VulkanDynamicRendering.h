#pragma once

#include "VulkanContext.h"

namespace core {
namespace vulkan {

class VulkanDynamicRendering {
 public:
  explicit VulkanDynamicRendering(VulkanContext* context);

  void BeginDynamicRendering(VkCommandBuffer command_buffer, VkImageView target_image_view,
                             VkExtent2D extent, VkClearValue clear_value) const;
  void BeginDynamicRendering(VkCommandBuffer command_buffer, VkImageView target_image_view,
                             VkExtent2D extent, VkClearValue clear_value,
                             VkImageView depth_image_view, VkClearValue depth_clear_value) const;
  void BeginDynamicRendering(VkCommandBuffer command_buffer, VkImageView target_image_view,
                             VkImageView resolve_image_view, VkExtent2D extent,
                             VkClearValue clear_value,
                             VkImageView depth_image_view = VK_NULL_HANDLE,
                             VkClearValue depth_clear_value = {}) const;
  void BeginDepthRendering(VkCommandBuffer command_buffer, VkImageView depth_image_view,
                           VkExtent2D extent, VkClearValue depth_clear_value) const;
  void EndDynamicRendering(VkCommandBuffer command_buffer) const;

 private:
  void LoadDynamicRenderingCommands();

  VulkanContext* context_ = nullptr;
  PFN_vkCmdBeginRendering vk_cmd_begin_rendering_ = nullptr;
  PFN_vkCmdEndRendering vk_cmd_end_rendering_ = nullptr;
};

}  // namespace vulkan
}  // namespace core
