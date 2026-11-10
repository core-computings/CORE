#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "GlfwSwapChainResizer.h"
#include "RenderGalaxy.h"
#include "VulkanCommandBuffer.h"
#include "VulkanDynamicRendering.h"
#include "VulkanSwapChain.h"
#include "VulkanSync.h"
#include "VulkanUtils.h"

// Renders a procedural spiral galaxy with ~50k stars rotating with differential angular velocity.
const uint32_t kWidth = 1280;
const uint32_t kHeight = 800;
const uint32_t kStarCount = 60000;

int main() {
  VkSurfaceKHR window_surface = VK_NULL_HANDLE;
  GLFWwindow* window;
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window = glfwCreateWindow(kWidth, kHeight, "Galaxy", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    throw std::runtime_error("failed to create window");
  }

  core::vulkan::QueueFamilyType queue_family_type = core::vulkan::QueueFamilyType::Graphics;
  core::vulkan::VulkanContext context(true, queue_family_type, nullptr);
  std::unique_ptr<core::vulkan::VulkanSwapChain> swap_chain;

  if (glfwCreateWindowSurface(context.instance, window, nullptr, &window_surface) != VK_SUCCESS) {
    throw std::runtime_error("failed to create window surface");
  }

  context.Init(window_surface);
  if (window_surface != VK_NULL_HANDLE) {
    swap_chain = std::make_unique<core::vulkan::VulkanSwapChain>(&context, window_surface);
  }

  core::vulkan::VulkanCommandBuffer command_buffer(&context);
  core::vulkan::VulkanDynamicRendering dynamic_rendering(&context);
  core::vulkan::VulkanSemaphore image_available_semaphore(&context);
  core::vulkan::VulkanFence in_flight_fence(&context);
  core::vulkan::DynamicRenderingInfo dynamic_rendering_info{
      .color_formats = {swap_chain->swapchain_image_format}};
  auto galaxy = std::make_unique<core::RenderGalaxy>(&context, dynamic_rendering_info, kStarCount);
  const VkClearValue clear_value{
      .color =
          {
              .float32 = {0.01f, 0.005f, 0.02f, 1.0f},
          },
  };

  galaxy->Init();

  core::GlfwSwapChainResizer resizer(window, swap_chain.get());

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    vkWaitForFences(context.logical_device, 1, &(in_flight_fence.fence), VK_TRUE, UINT64_MAX);

    uint32_t image_index = 0;
    const VkResult acquire_result =
        resizer.AcquireNextImage(image_available_semaphore.semaphore, &image_index);
    if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
      static_cast<void>(resizer.Recreate());
      continue;
    }
    in_flight_fence.Reset();

    galaxy->UpdateUniformBuffer(swap_chain->swapchain_extent.width,
                                swap_chain->swapchain_extent.height);

    command_buffer.BeginCommandBufferWithReset();

    swap_chain->TransitionImageLayout(command_buffer.buffer(), image_index,
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    dynamic_rendering.BeginDynamicRendering(command_buffer.buffer(),
                                            swap_chain->swapchain_image_views[image_index],
                                            swap_chain->swapchain_extent, clear_value);
    galaxy->Render(command_buffer.buffer(), swap_chain->swapchain_extent);
    dynamic_rendering.EndDynamicRendering(command_buffer.buffer());

    swap_chain->TransitionImageLayout(command_buffer.buffer(), image_index,
                                      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    command_buffer.Submit(
        in_flight_fence.fence,
        core::vulkan::SubmitSyncInfo{
            .wait_semaphores = {image_available_semaphore.semaphore},
            .wait_stage_masks = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
            .signal_semaphores = {swap_chain->PresentSemaphore(image_index)},
        });

    const VkResult present_result =
        swap_chain->Present(image_index, swap_chain->PresentSemaphore(image_index));
    if (resizer.NeedsRecreate(acquire_result, present_result)) {
      static_cast<void>(resizer.Recreate());
    }
  }
  vkDeviceWaitIdle(context.logical_device);

  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
