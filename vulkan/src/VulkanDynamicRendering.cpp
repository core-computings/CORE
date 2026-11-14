#include "VulkanDynamicRendering.h"

#include <stdexcept>

#include "VulkanUtils.h"

namespace core {
namespace vulkan {

VulkanDynamicRendering::VulkanDynamicRendering(VulkanContext* context) : context_(context) {
  LoadDynamicRenderingCommands();
}

void VulkanDynamicRendering::BeginDynamicRendering(VkCommandBuffer command_buffer,
                                                   VkImageView target_image_view, VkExtent2D extent,
                                                   VkClearValue clear_value) const {
  BeginDynamicRendering(command_buffer, target_image_view, VK_NULL_HANDLE, extent, clear_value);
}

void VulkanDynamicRendering::BeginDynamicRendering(VkCommandBuffer command_buffer,
                                                   VkImageView target_image_view, VkExtent2D extent,
                                                   VkClearValue clear_value,
                                                   VkImageView depth_image_view,
                                                   VkClearValue depth_clear_value) const {
  BeginDynamicRendering(command_buffer, target_image_view, VK_NULL_HANDLE, extent, clear_value,
                        depth_image_view, depth_clear_value);
}

void VulkanDynamicRendering::BeginDynamicRendering(VkCommandBuffer command_buffer,
                                                   VkImageView target_image_view,
                                                   VkImageView resolve_image_view,
                                                   VkExtent2D extent, VkClearValue clear_value,
                                                   VkImageView depth_image_view,
                                                   VkClearValue depth_clear_value) const {
  const bool has_resolve_attachment = resolve_image_view != VK_NULL_HANDLE;
  VkRenderingAttachmentInfo color_attachment{};
  color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
  color_attachment.imageView = target_image_view;
  color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color_attachment.resolveMode =
      has_resolve_attachment ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE;
  color_attachment.resolveImageView = resolve_image_view;
  color_attachment.resolveImageLayout =
      has_resolve_attachment ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp =
      has_resolve_attachment ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.clearValue = clear_value;

  VkRenderingAttachmentInfo depth_attachment{};
  depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
  depth_attachment.imageView = depth_image_view;
  depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth_attachment.clearValue = depth_clear_value;

  VkRenderingInfo rendering_info{};
  rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
  rendering_info.renderArea = {.offset = {0, 0}, .extent = extent};
  rendering_info.layerCount = 1;
  rendering_info.colorAttachmentCount = 1;
  rendering_info.pColorAttachments = &color_attachment;
  rendering_info.pDepthAttachment =
      depth_image_view == VK_NULL_HANDLE ? nullptr : &depth_attachment;

  vk_cmd_begin_rendering_(command_buffer, &rendering_info);
}

void VulkanDynamicRendering::BeginDepthRendering(VkCommandBuffer command_buffer,
                                                 VkImageView depth_image_view, VkExtent2D extent,
                                                 VkClearValue depth_clear_value) const {
  if (command_buffer == VK_NULL_HANDLE || depth_image_view == VK_NULL_HANDLE) {
    throw std::invalid_argument("Depth rendering requires a command buffer and depth image view");
  }
  if (extent.width == 0 || extent.height == 0) {
    throw std::invalid_argument("Depth rendering requires a non-empty extent");
  }

  const VkRenderingAttachmentInfo depth_attachment{
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .pNext = nullptr,
      .imageView = depth_image_view,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .resolveMode = VK_RESOLVE_MODE_NONE,
      .resolveImageView = VK_NULL_HANDLE,
      .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue = depth_clear_value,
  };
  const VkRenderingInfo rendering_info{
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .pNext = nullptr,
      .flags = 0,
      .renderArea = {.offset = {.x = 0, .y = 0}, .extent = extent},
      .layerCount = 1,
      .viewMask = 0,
      .colorAttachmentCount = 0,
      .pColorAttachments = nullptr,
      .pDepthAttachment = &depth_attachment,
      .pStencilAttachment = nullptr,
  };
  vk_cmd_begin_rendering_(command_buffer, &rendering_info);
}

void VulkanDynamicRendering::EndDynamicRendering(VkCommandBuffer command_buffer) const {
  vk_cmd_end_rendering_(command_buffer);
}

void VulkanDynamicRendering::LoadDynamicRenderingCommands() {
  const auto commands = core::vulkan::LoadDynamicRenderingCommands(context_->logical_device);
  vk_cmd_begin_rendering_ = commands.vkCmdBeginRendering;
  vk_cmd_end_rendering_ = commands.vkCmdEndRendering;

  if (!vk_cmd_begin_rendering_ || !vk_cmd_end_rendering_) {
    throw std::runtime_error("failed to load dynamic rendering commands");
  }
}

}  // namespace vulkan
}  // namespace core
