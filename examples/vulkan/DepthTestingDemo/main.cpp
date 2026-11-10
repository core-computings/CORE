#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <array>
#include <memory>
#include <vector>

#include "GlfwSwapChainResizer.h"
#include "RenderDepth.h"
#include "VulkanCommandBuffer.h"
#include "VulkanSwapChain.h"
#include "VulkanSync.h"
#include "VulkanUtils.h"

const uint32_t kWidth = 1000;
const uint32_t kHeight = 1000;
const bool kEnableDepthBuffer = true;

int main() {
  VkSurfaceKHR window_surface = VK_NULL_HANDLE;
  GLFWwindow* window;
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window = glfwCreateWindow(kWidth, kHeight, "Vulkan", nullptr, nullptr);
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
    swap_chain = std::make_unique<core::vulkan::VulkanSwapChain>(&context, window_surface,
                                                                 kEnableDepthBuffer);
  }

  core::vulkan::VulkanCommandBuffer command_buffer(&context);
  core::vulkan::VulkanFence fence(&context);
  core::vulkan::VulkanSemaphore image_available_semaphore(&context);
  core::vulkan::VulkanFence in_flight_fence(&context);
  core::vulkan::VulkanRenderPass render_pass(&context, swap_chain->swapchain_image_format,
                                             kEnableDepthBuffer);  // enable depth buffer
  std::unique_ptr<core::RenderDepth> texture =
      std::make_unique<core::RenderDepth>(&context, &render_pass);
  const std::array<VkClearValue, 2> clear_values{
      VkClearValue{.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}},
      VkClearValue{.depthStencil = {.depth = 1.0f, .stencil = 0}},
  };

  texture->Init("examples/data/core.png");
  swap_chain->CreateFrameBuffers(render_pass);

  core::GlfwSwapChainResizer resizer(window, swap_chain.get());
  const auto recreate_swap_chain = [&]() {
    if (resizer.Recreate()) {
      swap_chain->CreateFrameBuffers(render_pass);
    }
  };

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    vkWaitForFences(context.logical_device, 1, &(in_flight_fence.fence), VK_TRUE, UINT64_MAX);

    uint32_t image_index = 0;
    const VkResult acquire_result =
        resizer.AcquireNextImage(image_available_semaphore.semaphore, &image_index);
    if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
      recreate_swap_chain();
      continue;
    }
    in_flight_fence.Reset();
    texture->UpdateUniformBuffer(swap_chain->swapchain_extent.width,
                                 swap_chain->swapchain_extent.height);
    command_buffer.BeginCommandBufferWithReset();
    VkRenderPassBeginInfo renderpass_info{};
    renderpass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderpass_info.renderPass = render_pass.GetRenderPass();
    renderpass_info.framebuffer = swap_chain->swapchain_framebuffers[image_index];
    renderpass_info.renderArea.offset = {.x = 0, .y = 0};
    renderpass_info.renderArea.extent = swap_chain->swapchain_extent;
    renderpass_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
    renderpass_info.pClearValues = clear_values.data();
    vkCmdBeginRenderPass(command_buffer.buffer(), &renderpass_info, VK_SUBPASS_CONTENTS_INLINE);
    texture->Render(command_buffer.buffer(), swap_chain->swapchain_extent);
    vkCmdEndRenderPass(command_buffer.buffer());

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
      recreate_swap_chain();
    }
  }
  vkDeviceWaitIdle(context.logical_device);

  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}
