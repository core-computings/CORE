#pragma once

#include <memory>

#include "VulkanContext.h"
#include "VulkanRenderPass.h"
#include "VulkanSync.h"

namespace core {
namespace vulkan {

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> present_modes;
};

class VulkanSwapChain {
 public:
  VulkanSwapChain() = delete;
  VulkanSwapChain(VulkanContext* context, VkSurfaceKHR surface,
                  const bool enable_depth_buffer = false,
                  const VkExtent2D framebuffer_extent = VkExtent2D{.width = 0, .height = 0});
  ~VulkanSwapChain();

  void UnInit();

  void CreateFrameBuffers(VulkanRenderPass& render_pass);

  [[nodiscard]] bool Recreate(VkExtent2D framebuffer_extent);

  void TransitionImageLayout(VkCommandBuffer command_buffer, uint32_t image_index,
                             VkImageLayout new_layout);

  [[nodiscard]] VkResult AcquireNextImage(VkSemaphore semaphore, uint64_t timeout,
                                          uint32_t* image_index);

  [[nodiscard]] VkResult Present(uint32_t image_index, VkSemaphore wait_semaphore);

  VkSemaphore PresentSemaphore(uint32_t image_index) const;

  size_t RetiredSwapChainCount() const { return retired_swapchains_.size(); }

  VkExtent2D swapchain_extent;
  std::vector<VkFramebuffer> swapchain_framebuffers;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  VkFormat swapchain_image_format;
  std::vector<VkImage> swapchain_images;
  std::vector<VkImageView> swapchain_image_views;

 private:
  SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device);
  VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats);
  VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& present_modes);
  VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                              VkExtent2D framebuffer_extent) const;

  VkSwapchainKHR CreateSwapChain(VkExtent2D extent, VkSwapchainKHR old_swapchain) const;
  void RetrieveSwapChainImages();
  void CreatePresentSemaphores();
  void DestroyImageResources();
  void ReleaseRetiredSwapChains();
  void DestroyRetiredSwapChains();
  void CreateImageViews();
  void CreateDepthResources(VkFormat depth_format);

  VulkanContext* context_ = nullptr;
  SwapChainSupportDetails swapchain_support_details_;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  VkSurfaceFormatKHR surface_format_;
  VkPresentModeKHR present_mode_;

  bool enable_depth_buffer_ = false;
  core::vulkan::VulkanImage depth_image_;

  std::vector<VkImageLayout> swapchain_image_layouts_;
  std::vector<std::unique_ptr<VulkanSemaphore>> present_semaphores_;
  std::vector<bool> presented_images_;
  std::vector<VkSwapchainKHR> retired_swapchains_;
  VulkanFence retirement_fence_;
  bool retirement_fence_pending_ = false;
  bool retire_on_signal_ = false;
  std::vector<std::unique_ptr<VulkanSemaphore>> retired_semaphores_;
};

}  // namespace vulkan
}  // namespace core
