#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdio>
#include <memory>
#include <string_view>
#include <vector>

#include "RenderTexture.h"
#include "VulkanCommandBuffer.h"
#include "VulkanSwapChain.h"
#include "VulkanSync.h"
#include "VulkanUtils.h"

// This demo draws a texture in a glfw window using vulkan graphic pipeline
const uint32_t kWidth = 800;
const uint32_t kHeight = 600;

int main(int argc, char* argv[]) {
  core::vulkan::SamplerOptions sampler_options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--nearest") {
      sampler_options.mag_filter = VK_FILTER_NEAREST;
      sampler_options.min_filter = VK_FILTER_NEAREST;
      sampler_options.mipmap_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    } else if (argument == "--clamp") {
      sampler_options.address_mode_u = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      sampler_options.address_mode_v = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      sampler_options.address_mode_w = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    } else {
      std::fprintf(argument == "--help" ? stdout : stderr,
                   "Usage: vk_texture_demo [--nearest] [--clamp]\n");
      return argument == "--help" ? 0 : 1;
    }
  }
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
    swap_chain = std::make_unique<core::vulkan::VulkanSwapChain>(&context, window_surface);
  }

  core::vulkan::VulkanCommandBuffer command_buffer(&context);
  core::vulkan::VulkanFence fence(&context);
  core::vulkan::VulkanSemaphore image_available_semaphore(&context);
  std::vector<std::unique_ptr<core::vulkan::VulkanSemaphore>> render_finished_semaphores;
  core::vulkan::VulkanFence in_flight_fence(&context);
  core::vulkan::VulkanRenderPass render_pass(&context, swap_chain->swapchain_image_format);
  std::unique_ptr<core::RenderTexture> texture =
      std::make_unique<core::RenderTexture>(&context, &render_pass,
                                            sampler_options);
  const VkClearValue clear_color{
      .color =
          {
              .float32 = {0.0f, 0.0f, 0.0f, 1.0f},
          },
  };
  for (size_t i = 0; i < swap_chain->swapchain_images.size(); ++i) {
    render_finished_semaphores.push_back(std::make_unique<core::vulkan::VulkanSemaphore>(&context));
  }

  texture->Init("examples/data/core.png");
  swap_chain->CreateFrameBuffers(render_pass);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    vkWaitForFences(context.logical_device, 1, &(in_flight_fence.fence), VK_TRUE, UINT64_MAX);
    in_flight_fence.Reset();

    uint32_t image_index;
    vkAcquireNextImageKHR(context.logical_device, swap_chain->swapchain, UINT64_MAX,
                          image_available_semaphore.semaphore, VK_NULL_HANDLE, &image_index);
    texture->UpdateUniformBuffer(swap_chain->swapchain_extent.width,
                                 swap_chain->swapchain_extent.height);
    command_buffer.BeginCommandBufferWithReset();
    VkRenderPassBeginInfo renderpass_info{};
    renderpass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderpass_info.renderPass = render_pass.GetRenderPass();
    renderpass_info.framebuffer = swap_chain->swapchain_framebuffers[image_index];
    renderpass_info.renderArea.offset = {.x = 0, .y = 0};
    renderpass_info.renderArea.extent = swap_chain->swapchain_extent;
    renderpass_info.clearValueCount = 1;
    renderpass_info.pClearValues = &clear_color;
    vkCmdBeginRenderPass(command_buffer.buffer(), &renderpass_info, VK_SUBPASS_CONTENTS_INLINE);
    texture->Render(command_buffer.buffer(), swap_chain->swapchain_extent);
    vkCmdEndRenderPass(command_buffer.buffer());

    command_buffer.Submit(
        in_flight_fence.fence,
        core::vulkan::SubmitSyncInfo{
            .wait_semaphores = {image_available_semaphore.semaphore},
            .wait_stage_masks = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
            .signal_semaphores = {render_finished_semaphores[image_index]->semaphore},
        });

    const VkResult present_result =
        swap_chain->Present(image_index, render_finished_semaphores[image_index]->semaphore);
    if (present_result != VK_SUCCESS && present_result != VK_SUBOPTIMAL_KHR) {
      VK_CHECK(present_result);
    }
  }
  vkDeviceWaitIdle(context.logical_device);

  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}
