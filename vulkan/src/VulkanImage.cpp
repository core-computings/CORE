#include "VulkanImage.h"

#include <stdexcept>

#include "VulkanBuffer.h"

namespace core {
namespace vulkan {

VulkanImage::VulkanImage(VulkanContext* context, const uint32_t width, const uint32_t height,
                         const VkFormat format, const VkImageUsageFlags usage,
                         const VkImageAspectFlags aspect, const VkMemoryPropertyFlags properties,
                         const VkImageTiling tiling, const uint32_t mip_levels,
                         const VkSampleCountFlagBits samples, const VkImageCreateFlags flags,
                         const uint32_t layers)
    : context_(context),
      image_format_(format),
      mip_levels_(mip_levels),
      array_layers_(layers),
      samples_(samples),
      tiling_(tiling),
      usage_(usage),
      image_width(width),
      image_height(height) {
  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.extent.width = image_width;
  image_info.extent.height = image_height;
  image_info.extent.depth = 1;  // 2D image has depth 1
  image_info.mipLevels = mip_levels_;
  image_info.arrayLayers = layers;
  image_info.format = image_format_;
  image_info.tiling = tiling;
  image_info.initialLayout =
      VK_IMAGE_LAYOUT_UNDEFINED;  // only two states: UNDEFINED and PREINITIALIZED
  image_info.usage = usage;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.samples = samples_;
  image_info.flags = flags;  // Optional

  VK_CHECK(vkCreateImage(context_->logical_device, &image_info, nullptr, &image));

  VkMemoryRequirements mem_requirements;
  vkGetImageMemoryRequirements(context_->logical_device, image, &mem_requirements);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = mem_requirements.size;
  alloc_info.memoryTypeIndex =
      context_->FindMemoryType(mem_requirements.memoryTypeBits, properties);

  VK_CHECK(vkAllocateMemory(context_->logical_device, &alloc_info, nullptr, &image_memory));

  vkBindImageMemory(context_->logical_device, image, image_memory, 0);

  // Create image view
  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = image;
  if ((flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) != 0 && array_layers_ >= 6) {
    view_info.viewType =
        array_layers_ == 6 ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
  } else if (array_layers_ > 1) {
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  } else {
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  }
  view_info.format = image_format_;
  view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.subresourceRange.aspectMask = aspect;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = mip_levels_;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = layers;

  VK_CHECK(vkCreateImageView(context_->logical_device, &view_info, nullptr, &image_view));
}

VulkanImage::~VulkanImage() {
  if (context_ && image_view != VK_NULL_HANDLE) {
    vkDestroyImageView(context_->logical_device, image_view, nullptr);
  }
  if (context_ && image != VK_NULL_HANDLE) {
    vkDestroyImage(context_->logical_device, image, nullptr);
  }
  if (context_ && image_memory != VK_NULL_HANDLE) {
    vkFreeMemory(context_->logical_device, image_memory, nullptr);
  }
}

VulkanImage& VulkanImage::operator=(VulkanImage&& rhs) {
  if (this == &rhs) {
    return *this;
  }

  if (context_ && image_view != VK_NULL_HANDLE) {
    vkDestroyImageView(context_->logical_device, image_view, nullptr);
  }
  if (context_ && image != VK_NULL_HANDLE) {
    vkDestroyImage(context_->logical_device, image, nullptr);
  }
  if (context_ && image_memory != VK_NULL_HANDLE) {
    vkFreeMemory(context_->logical_device, image_memory, nullptr);
  }

  context_ = rhs.context_;
  image_format_ = rhs.image_format_;
  mip_levels_ = rhs.mip_levels_;
  array_layers_ = rhs.array_layers_;
  samples_ = rhs.samples_;
  tiling_ = rhs.tiling_;
  usage_ = rhs.usage_;

  image = rhs.image;
  rhs.image = VK_NULL_HANDLE;

  image_view = rhs.image_view;
  rhs.image_view = VK_NULL_HANDLE;

  image_memory = rhs.image_memory;
  rhs.image_memory = VK_NULL_HANDLE;

  image_width = rhs.image_width;
  image_height = rhs.image_height;

  return *this;
}

void VulkanImage::TransitionImageLayout(const VkImageLayout old_layout,
                                        const VkImageLayout new_layout,
                                        [[maybe_unused]] const VkFormat format,
                                        const uint32_t mip_levels, const uint32_t layers) {
  VulkanCommandBuffer command_buffer = VulkanCommandBuffer::BeginOneTimeCommands(context_);
  TransitionImageLayout(command_buffer.buffer(), old_layout, new_layout, format, mip_levels,
                        layers);
  command_buffer.EndOneTimeCommands();
}

void VulkanImage::TransitionImageLayout(VkCommandBuffer command_buffer, VkImageLayout old_layout,
                                       VkImageLayout new_layout, [[maybe_unused]] VkFormat format,
                                       uint32_t mip_levels, uint32_t layers) {
  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = mip_levels;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = layers;

  VkPipelineStageFlags source_stage;
  VkPipelineStageFlags destination_stage;

  /**
   * case 1: undefined -> transfer destination
   * case 2: transfer destination -> shader read
   */
  if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
      new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;  // Not a real stage within the graphics and
                                                         // compute pipelines. It is more of a
                                                         // pseudo-stage where transfers happen
  } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
             new_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    source_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    destination_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  } else if (old_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
             new_layout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = 0;

    source_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    destination_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  } else {
    throw std::invalid_argument("unsupported layout transition!");
  }

  vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);
}

void VulkanImage::TransitionDepthImageLayout(const VkImageLayout old_layout,
                                             const VkImageLayout new_layout,
                                             [[maybe_unused]] const VkFormat format) {
  VulkanCommandBuffer command_buffer = VulkanCommandBuffer::BeginOneTimeCommands(context_);

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags source_stage;
  VkPipelineStageFlags destination_stage;

  if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT) {
      barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }
  }

  if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
      new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
             new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  } else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
             new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask =
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destination_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  } else {
    throw std::invalid_argument("unsupported layout transition!");
  }

  vkCmdPipelineBarrier(command_buffer.buffer(), source_stage, destination_stage, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);

  command_buffer.EndOneTimeCommands();
}

void VulkanImage::GenerateMipmaps() {
  VulkanCommandBuffer command_buffer = VulkanCommandBuffer::BeginOneTimeCommands(context_);
  GenerateMipmaps(command_buffer.buffer());
  command_buffer.EndOneTimeCommands();
}

void VulkanImage::GenerateMipmaps(VkCommandBuffer command_buffer) {
  if (command_buffer == VK_NULL_HANDLE || image == VK_NULL_HANDLE ||
      samples_ != VK_SAMPLE_COUNT_1_BIT || (usage_ & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0) {
    throw std::invalid_argument("mipmap generation requires a valid single-sample transfer image");
  }
  if (mip_levels_ > 1) {
    if ((usage_ & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0) {
      throw std::invalid_argument("mipmap generation requires transfer source usage");
    }
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(context_->physical_device, image_format_, &properties);
    const VkFormatFeatureFlags features = tiling_ == VK_IMAGE_TILING_LINEAR
                                             ? properties.linearTilingFeatures
                                             : properties.optimalTilingFeatures;
    constexpr VkFormatFeatureFlags kRequired =
        VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    if ((features & kRequired) != kRequired) {
      throw std::invalid_argument("image format does not support linear mipmap blits");
    }
  }

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.image = image;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = array_layers_;
  barrier.subresourceRange.levelCount = 1;

  int32_t mip_width = static_cast<int32_t>(image_width);
  int32_t mip_height = static_cast<int32_t>(image_height);

  for (uint32_t i = 1; i < mip_levels_; i++) {
    barrier.subresourceRange.baseMipLevel = i - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    VkImageBlit blit{};
    // src
    blit.srcOffsets[0] = {.x = 0, .y = 0, .z = 0};
    blit.srcOffsets[1] = {.x = mip_width, .y = mip_height, .z = 1};
    blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.srcSubresource.mipLevel = i - 1;
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount = array_layers_;
    // dst
    blit.dstOffsets[0] = {.x = 0, .y = 0, .z = 0};
    blit.dstOffsets[1] = {.x = mip_width > 1 ? mip_width / 2 : 1,
                          .y = mip_height > 1 ? mip_height / 2 : 1,
                          .z = 1};
    blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blit.dstSubresource.mipLevel = i;
    blit.dstSubresource.baseArrayLayer = 0;
    blit.dstSubresource.layerCount = array_layers_;

    vkCmdBlitImage(command_buffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                         &barrier);

    if (mip_width > 1) {
      mip_width /= 2;
    }
    if (mip_height > 1) {
      mip_height /= 2;
    }
  }

  barrier.subresourceRange.baseMipLevel = mip_levels_ - 1;
  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                       &barrier);
}

}  // namespace vulkan
}  // namespace core
