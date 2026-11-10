#pragma once

#include <GLFW/glfw3.h>

#include <cstdint>

#include "VulkanSwapChain.h"
#include "VulkanUtils.h"

namespace core {

class GlfwSwapChainResizer {
 public:
  GlfwSwapChainResizer(GLFWwindow* window, vulkan::VulkanSwapChain* swap_chain)
      : window_(window), swap_chain_(swap_chain), framebuffer_extent_(FramebufferExtent()) {}

  VkResult AcquireNextImage(VkSemaphore semaphore, uint32_t* image_index) const {
    if (FramebufferResized()) {
      return VK_ERROR_OUT_OF_DATE_KHR;
    }
    const VkResult result = swap_chain_->AcquireNextImage(semaphore, UINT64_MAX, image_index);
    if (result != VK_ERROR_OUT_OF_DATE_KHR && result != VK_SUBOPTIMAL_KHR) {
      VK_CHECK(result);
    }
    return result;
  }

  bool NeedsRecreate(VkResult acquire_result, VkResult present_result) const {
    if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR ||
        acquire_result == VK_SUBOPTIMAL_KHR || FramebufferResized()) {
      return true;
    }
    VK_CHECK(present_result);
    return false;
  }

  bool Recreate() {
    framebuffer_extent_ = FramebufferExtent();
    while ((framebuffer_extent_.width == 0 || framebuffer_extent_.height == 0) &&
           glfwWindowShouldClose(window_) == GLFW_FALSE) {
      glfwWaitEvents();
      framebuffer_extent_ = FramebufferExtent();
    }
    return swap_chain_->Recreate(framebuffer_extent_);
  }

 private:
  VkExtent2D FramebufferExtent() const {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    return VkExtent2D{.width = static_cast<uint32_t>(width),
                      .height = static_cast<uint32_t>(height)};
  }

  bool FramebufferResized() const {
    const VkExtent2D extent = FramebufferExtent();
    return extent.width != framebuffer_extent_.width ||
           extent.height != framebuffer_extent_.height;
  }

  GLFWwindow* window_ = nullptr;
  vulkan::VulkanSwapChain* swap_chain_ = nullptr;
  VkExtent2D framebuffer_extent_{};
};

}
