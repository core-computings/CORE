#include "VulkanSwapChain.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

#include "VulkanCommandBuffer.h"

namespace core {
namespace vulkan {

VulkanSwapChain::VulkanSwapChain(VulkanContext* context, VkSurfaceKHR surface,
                                 const bool enable_depth_buffer,
                                 const VkExtent2D framebuffer_extent)
    : context_(context),
      surface_(surface),
      enable_depth_buffer_(enable_depth_buffer),
      retirement_fence_(context) {
  swapchain_support_details_ = QuerySwapChainSupport(context_->physical_device);
  surface_format_ = ChooseSwapSurfaceFormat(swapchain_support_details_.formats);
  present_mode_ = ChooseSwapPresentMode(swapchain_support_details_.present_modes);
  swapchain_extent =
      ChooseSwapExtent(swapchain_support_details_.capabilities, framebuffer_extent);
  if (swapchain_extent.width == 0 || swapchain_extent.height == 0) {
    throw std::runtime_error("swap chain surface has no drawable extent");
  }
  swapchain_image_format = surface_format_.format;

  swapchain = CreateSwapChain(swapchain_extent, VK_NULL_HANDLE);
  struct ConstructionGuard {
    VulkanSwapChain* owner;
    bool completed;
    ~ConstructionGuard() {
      if (!completed) {
        owner->UnInit();
      }
    }
  } guard{.owner = this, .completed = false};
  RetrieveSwapChainImages();
  CreateImageViews();
  CreatePresentSemaphores();
  guard.completed = true;
}

VulkanSwapChain::~VulkanSwapChain() {
  UnInit();
  if (context_ && surface_ != VK_NULL_HANDLE) {
    vkDestroySurfaceKHR(context_->instance, surface_, nullptr);
  }
}

void VulkanSwapChain::UnInit() {
  DestroyImageResources();
  if (swapchain != VK_NULL_HANDLE) {
    vkDestroySwapchainKHR(context_->logical_device, swapchain, nullptr);
    swapchain = VK_NULL_HANDLE;
  }
  swapchain_images.clear();
  swapchain_image_layouts_.clear();
  present_semaphores_.clear();
  if (retirement_fence_pending_) {
    static_cast<void>(vkWaitForFences(context_->logical_device, 1, &retirement_fence_.fence,
                                      VK_TRUE, UINT64_MAX));
    retirement_fence_pending_ = false;
  }
  DestroyRetiredSwapChains();
}

bool VulkanSwapChain::Recreate(VkExtent2D framebuffer_extent) {
  if (swapchain == VK_NULL_HANDLE) {
    throw std::logic_error("swap chain has been released");
  }
  if (framebuffer_extent.width == 0 || framebuffer_extent.height == 0) {
    return false;
  }

  VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
      context_->physical_device, surface_, &swapchain_support_details_.capabilities));
  const VkExtent2D extent =
      ChooseSwapExtent(swapchain_support_details_.capabilities, framebuffer_extent);
  if (extent.width == 0 || extent.height == 0) {
    return false;
  }

  VK_CHECK(vkDeviceWaitIdle(context_->logical_device));
  DestroyImageResources();
  swapchain_images.clear();
  swapchain_image_layouts_.clear();
  for (std::unique_ptr<VulkanSemaphore>& semaphore : present_semaphores_) {
    retired_semaphores_.push_back(std::move(semaphore));
  }
  present_semaphores_.clear();
  retired_swapchains_.push_back(swapchain);
  retire_on_signal_ = false;
  swapchain = VK_NULL_HANDLE;
  swapchain = CreateSwapChain(extent, retired_swapchains_.back());
  swapchain_extent = extent;

  RetrieveSwapChainImages();
  CreateImageViews();
  CreatePresentSemaphores();
  return true;
}

VkSwapchainKHR VulkanSwapChain::CreateSwapChain(VkExtent2D extent,
                                                VkSwapchainKHR old_swapchain) const {
  const VkSurfaceCapabilitiesKHR& capabilities = swapchain_support_details_.capabilities;
  uint32_t image_count = capabilities.minImageCount + 1;
  if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount) {
    image_count = capabilities.maxImageCount;
  }

  VkSwapchainCreateInfoKHR create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  create_info.surface = surface_;

  create_info.minImageCount = image_count;
  create_info.imageFormat = surface_format_.format;
  create_info.imageColorSpace = surface_format_.colorSpace;
  create_info.imageExtent = extent;
  create_info.imageArrayLayers = 1;
  create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

  QueueFamilyIndices indices = context_->GetQueueFamilyIndices();
  uint32_t queue_family_indices[] = {indices.graphics_family.value(),
                                     indices.present_family.value()};

  if (indices.graphics_family != indices.present_family) {
    create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    create_info.queueFamilyIndexCount = 2;
    create_info.pQueueFamilyIndices = queue_family_indices;
  } else {
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  create_info.preTransform = capabilities.currentTransform;
  create_info.compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
  for (const VkCompositeAlphaFlagBitsKHR composite_alpha :
       {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR}) {
    if ((capabilities.supportedCompositeAlpha & composite_alpha) != 0) {
      create_info.compositeAlpha = composite_alpha;
      break;
    }
  }
  create_info.presentMode = present_mode_;
  create_info.clipped = VK_TRUE;
  create_info.oldSwapchain = old_swapchain;

  VkSwapchainKHR new_swapchain = VK_NULL_HANDLE;
  const VkResult result =
      vkCreateSwapchainKHR(context_->logical_device, &create_info, nullptr, &new_swapchain);
  if (result != VK_SUCCESS) {
    throw std::runtime_error("create swap chain failed");
  }
  return new_swapchain;
}

void VulkanSwapChain::RetrieveSwapChainImages() {
  uint32_t image_count = 0;
  VK_CHECK(vkGetSwapchainImagesKHR(context_->logical_device, swapchain, &image_count, nullptr));
  swapchain_images.resize(image_count);
  VK_CHECK(vkGetSwapchainImagesKHR(context_->logical_device, swapchain, &image_count,
                                   swapchain_images.data()));
  swapchain_image_layouts_.assign(image_count, VK_IMAGE_LAYOUT_UNDEFINED);
  presented_images_.assign(image_count, false);
}

void VulkanSwapChain::CreatePresentSemaphores() {
  present_semaphores_.clear();
  for (size_t i = 0; i < swapchain_images.size(); ++i) {
    present_semaphores_.push_back(std::make_unique<VulkanSemaphore>(context_));
  }
}

VkSemaphore VulkanSwapChain::PresentSemaphore(uint32_t image_index) const {
  if (image_index >= present_semaphores_.size()) {
    throw std::out_of_range("swapchain image index out of range");
  }
  return present_semaphores_[image_index]->semaphore;
}

void VulkanSwapChain::DestroyImageResources() {
  for (VkFramebuffer framebuffer : swapchain_framebuffers) {
    vkDestroyFramebuffer(context_->logical_device, framebuffer, nullptr);
  }
  swapchain_framebuffers.clear();
  for (VkImageView image_view : swapchain_image_views) {
    vkDestroyImageView(context_->logical_device, image_view, nullptr);
  }
  swapchain_image_views.clear();
}

VkResult VulkanSwapChain::AcquireNextImage(VkSemaphore semaphore, uint64_t timeout,
                                           uint32_t* image_index) {
  const bool track = !retired_swapchains_.empty() && !retirement_fence_pending_;
  if (track) {
    retirement_fence_.Reset();
  }
  const VkResult result =
      vkAcquireNextImageKHR(context_->logical_device, swapchain, timeout, semaphore,
                            track ? retirement_fence_.fence : VK_NULL_HANDLE, image_index);
  if (track && (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR)) {
    retirement_fence_pending_ = true;
    retire_on_signal_ =
        *image_index < presented_images_.size() && presented_images_[*image_index];
  }
  ReleaseRetiredSwapChains();
  return result;
}

void VulkanSwapChain::ReleaseRetiredSwapChains() {
  if (!retirement_fence_pending_) {
    return;
  }
  const VkResult status = vkGetFenceStatus(context_->logical_device, retirement_fence_.fence);
  if (status == VK_NOT_READY) {
    return;
  }
  VK_CHECK(status);
  retirement_fence_pending_ = false;
  if (retire_on_signal_) {
    DestroyRetiredSwapChains();
  }
}

void VulkanSwapChain::DestroyRetiredSwapChains() {
  for (VkSwapchainKHR retired_swapchain : retired_swapchains_) {
    vkDestroySwapchainKHR(context_->logical_device, retired_swapchain, nullptr);
  }
  retired_swapchains_.clear();
  retired_semaphores_.clear();
}

VkResult VulkanSwapChain::Present(uint32_t image_index, VkSemaphore wait_semaphore) {
  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = &wait_semaphore;
  present_info.swapchainCount = 1;
  present_info.pSwapchains = &swapchain;
  present_info.pImageIndices = &image_index;
  const VkResult result = vkQueuePresentKHR(context_->present_queue(), &present_info);
  if ((result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR) &&
      image_index < presented_images_.size()) {
    presented_images_[image_index] = true;
  }
  return result;
}

SwapChainSupportDetails VulkanSwapChain::QuerySwapChainSupport(VkPhysicalDevice device) {
  SwapChainSupportDetails details;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &details.capabilities);

  uint32_t format_count;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &format_count, nullptr);
  if (format_count != 0) {
    details.formats.resize(format_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &format_count, details.formats.data());
  }

  uint32_t present_mode_count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &present_mode_count, nullptr);
  if (present_mode_count != 0) {
    details.present_modes.resize(present_mode_count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &present_mode_count,
                                              details.present_modes.data());
  }
  return details;
}

VkSurfaceFormatKHR VulkanSwapChain::ChooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR>& formats) {
  for (const auto& format : formats) {
    if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return format;
    }
  }
  return formats[0];
}

VkPresentModeKHR VulkanSwapChain::ChooseSwapPresentMode(
    const std::vector<VkPresentModeKHR>& present_modes) {
  for (const auto& mode : present_modes) {
    if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return mode;
    }
  }
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VulkanSwapChain::ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                                             VkExtent2D framebuffer_extent) const {
  if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  }
  if (framebuffer_extent.width == 0 || framebuffer_extent.height == 0) {
    return VkExtent2D{.width = 0, .height = 0};
  }
  return VkExtent2D{
      .width = std::clamp(framebuffer_extent.width, capabilities.minImageExtent.width,
                          capabilities.maxImageExtent.width),
      .height = std::clamp(framebuffer_extent.height, capabilities.minImageExtent.height,
                           capabilities.maxImageExtent.height),
  };
}

void VulkanSwapChain::CreateImageViews() {
  swapchain_image_views.resize(swapchain_images.size());
  for (size_t i = 0; i < swapchain_images.size(); ++i) {
    VkImageViewCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    create_info.image = swapchain_images[i];
    create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    create_info.format = swapchain_image_format;
    create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    create_info.subresourceRange.baseMipLevel = 0;
    create_info.subresourceRange.levelCount = 1;
    create_info.subresourceRange.baseArrayLayer = 0;
    create_info.subresourceRange.layerCount = 1;

    VK_CHECK(vkCreateImageView(context_->logical_device, &create_info, nullptr,
                               &swapchain_image_views[i]));
  }
}

void VulkanSwapChain::CreateDepthResources(VkFormat depth_format) {
  depth_image_ = core::vulkan::VulkanImage(
      context_, swapchain_extent.width, swapchain_extent.height, depth_format,
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_TILING_OPTIMAL);

  depth_image_.TransitionDepthImageLayout(
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, depth_format);
}

void VulkanSwapChain::CreateFrameBuffers(VulkanRenderPass& render_pass) {
  if (enable_depth_buffer_) {
    CreateDepthResources(render_pass.depth_format);
  }

  swapchain_framebuffers.resize(swapchain_image_views.size());

  for (size_t i = 0; i < swapchain_image_views.size(); ++i) {
    std::vector<VkImageView> attachments;
    attachments.push_back(swapchain_image_views[i]);
    if (enable_depth_buffer_) {
      attachments.push_back(depth_image_.image_view);
    }

    VkFramebufferCreateInfo framebuffer_info{};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = render_pass.GetRenderPass();
    framebuffer_info.attachmentCount = static_cast<uint32_t>(attachments.size());
    framebuffer_info.pAttachments = attachments.data();
    framebuffer_info.width = swapchain_extent.width;
    framebuffer_info.height = swapchain_extent.height;
    framebuffer_info.layers = 1;

    VK_CHECK(vkCreateFramebuffer(context_->logical_device, &framebuffer_info, nullptr,
                                 &swapchain_framebuffers[i]));
  }
}

void VulkanSwapChain::TransitionImageLayout(VkCommandBuffer command_buffer, uint32_t image_index,
                                            VkImageLayout new_layout) {
  if (image_index >= swapchain_images.size()) {
    throw std::out_of_range("swapchain image index out of range");
  }

  VkImageLayout old_layout = swapchain_image_layouts_[image_index];

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = swapchain_images[image_index];
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;

  VkPipelineStageFlags source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  VkPipelineStageFlags destination_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

  if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
      new_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destination_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  } else if (old_layout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR &&
             new_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
    barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    source_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    destination_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  } else if (old_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
             new_layout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) {
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = 0;
    source_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    destination_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  } else {
    throw std::invalid_argument("unsupported swapchain image layout transition!");
  }

  swapchain_image_layouts_[image_index] = new_layout;

  vkCmdPipelineBarrier(command_buffer, source_stage, destination_stage, 0, 0, nullptr, 0, nullptr,
                       1, &barrier);
}

}  // namespace vulkan
}  // namespace core
