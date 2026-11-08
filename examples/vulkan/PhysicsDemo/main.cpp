#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <vector>

#include "Physics.h"
#include "RenderCube.h"
#include "RenderFloor.h"
#include "VulkanCamera.h"
#include "VulkanCommandBuffer.h"
#include "VulkanDynamicRendering.h"
#include "VulkanImage.h"
#include "VulkanPipelineCache.h"
#include "VulkanSwapChain.h"
#include "VulkanSync.h"
#include "VulkanUtils.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

// Demo: a cube spawns mid-air, falls under gravity onto a checkerboard floor,
// bounces with restitution and friction, tumbles, and auto-resets once it
// settles. Dynamic rendering with a manually-managed depth attachment.
// WASD moves the camera; mouse-look rotates the view.
const uint32_t kWidth = 1280;
const uint32_t kHeight = 800;
const VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

// Camera setup — positioned to view the spawn region with the floor below.
const glm::vec3 kCameraPos = glm::vec3(6.0f, 4.5f, 8.0f);
const glm::vec3 kCameraFront = glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f) - kCameraPos);
const glm::vec3 kCameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
const float kCameraSpeed = 0.05f;
const float kMouseSensitivity = 0.1f;

std::unique_ptr<core::vulkan::VulkanCamera> camera =
    std::make_unique<core::vulkan::VulkanCamera>(kCameraPos, kCameraFront, kCameraUp, kCameraSpeed);

float last_x = kWidth / 2.0f;
float last_y = kHeight / 2.0f;
bool first_mouse = true;

void process_inputs(GLFWwindow* window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);

int main() {
  VkSurfaceKHR window_surface = VK_NULL_HANDLE;
  GLFWwindow* window;
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window = glfwCreateWindow(kWidth, kHeight, "Physics Demo", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    throw std::runtime_error("failed to create window");
  }

  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwSetCursorPosCallback(window, mouse_callback);

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

  // Manually-managed depth image — the swapchain's internal depth buffer is
  // only wired up for the traditional render-pass path, so for dynamic
  // rendering we own the depth attachment here.
  core::vulkan::VulkanImage depth_image(
      &context, swap_chain->swapchain_extent.width, swap_chain->swapchain_extent.height,
      kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_TILING_OPTIMAL);
  depth_image.TransitionDepthImageLayout(
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, kDepthFormat);

  core::vulkan::VulkanCommandBuffer command_buffer(&context);
  core::vulkan::VulkanDynamicRendering dynamic_rendering(&context);
  core::vulkan::VulkanSemaphore image_available_semaphore(&context);
  std::vector<std::unique_ptr<core::vulkan::VulkanSemaphore>> render_finished_semaphores;
  core::vulkan::VulkanFence in_flight_fence(&context);

  core::vulkan::DynamicRenderingInfo dynamic_rendering_info{};
  dynamic_rendering_info.color_formats = {swap_chain->swapchain_image_format};
  dynamic_rendering_info.depth_format = kDepthFormat;

  auto floor = std::make_unique<core::RenderFloor>(&context, dynamic_rendering_info);
  auto cube = std::make_unique<core::RenderCube>(&context, dynamic_rendering_info);
  const VkClearValue color_clear_value{
      .color =
          {
              .float32 = {0.08f, 0.10f, 0.14f, 1.0f},
          },
  };
  const VkClearValue depth_clear_value{
      .depthStencil = {.depth = 1.0f, .stencil = 0},
  };

  core::PhysicsBody body;
  for (size_t i = 0; i < swap_chain->swapchain_images.size(); ++i) {
    render_finished_semaphores.push_back(std::make_unique<core::vulkan::VulkanSemaphore>(&context));
  }

  core::vulkan::VulkanPipelineCache pipeline_cache(
      &context, std::filesystem::path(PIPELINE_CACHE_DIR) / "physics_demo.cache");
  floor->SetPipelineCache(&pipeline_cache);
  cube->SetPipelineCache(&pipeline_cache);
  floor->Init();
  cube->Init();
  pipeline_cache.Save();

  auto last_time = std::chrono::high_resolution_clock::now();

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    process_inputs(window);

    const auto now = std::chrono::high_resolution_clock::now();
    const float dt =
        std::chrono::duration<float, std::chrono::seconds::period>(now - last_time).count();
    last_time = now;

    // Fixed-step the physics so visual behaviour is independent of FPS.
    constexpr float kFixedDt = 1.0f / 240.0f;
    static float accumulator = 0.0f;
    accumulator = std::min(accumulator + dt, 0.25f);
    while (accumulator >= kFixedDt) {
      body.Step(kFixedDt);
      accumulator -= kFixedDt;
    }

    vkWaitForFences(context.logical_device, 1, &(in_flight_fence.fence), VK_TRUE, UINT64_MAX);
    in_flight_fence.Reset();

    uint32_t image_index;
    vkAcquireNextImageKHR(context.logical_device, swap_chain->swapchain, UINT64_MAX,
                          image_available_semaphore.semaphore, VK_NULL_HANDLE, &image_index);

    const glm::mat4 view = camera->GetViewMatrix();
    const glm::mat4 project =
        glm::perspective(glm::radians(45.0f),
                         static_cast<float>(swap_chain->swapchain_extent.width) /
                             static_cast<float>(swap_chain->swapchain_extent.height),
                         0.1f, 100.0f);

    floor->UpdateUniformBuffer(view, project);
    cube->UpdateUniformBuffer(body.ModelMatrix(), view, project);

    command_buffer.BeginCommandBufferWithReset();

    swap_chain->TransitionImageLayout(command_buffer.buffer(), image_index,
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    dynamic_rendering.BeginDynamicRendering(
        command_buffer.buffer(), swap_chain->swapchain_image_views[image_index],
        swap_chain->swapchain_extent, color_clear_value, depth_image.image_view, depth_clear_value);
    floor->Render(command_buffer.buffer(), swap_chain->swapchain_extent);
    cube->Render(command_buffer.buffer(), swap_chain->swapchain_extent);
    dynamic_rendering.EndDynamicRendering(command_buffer.buffer());

    swap_chain->TransitionImageLayout(command_buffer.buffer(), image_index,
                                      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

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

void process_inputs(GLFWwindow* window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera->ProcessKeyboard(core::vulkan::CameraMovement::FORWARD);
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera->ProcessKeyboard(core::vulkan::CameraMovement::BACKWARD);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera->ProcessKeyboard(core::vulkan::CameraMovement::LEFT);
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera->ProcessKeyboard(core::vulkan::CameraMovement::RIGHT);
}

void mouse_callback([[maybe_unused]] GLFWwindow* window, double xpos, double ypos) {
  if (first_mouse) {
    last_x = static_cast<float>(xpos);
    last_y = static_cast<float>(ypos);
    first_mouse = false;
  }
  float xoffset = static_cast<float>(xpos) - last_x;
  float yoffset = last_y - static_cast<float>(ypos);  // Y inverted (screen y grows downward)
  last_x = static_cast<float>(xpos);
  last_y = static_cast<float>(ypos);

  xoffset *= kMouseSensitivity;
  yoffset *= kMouseSensitivity;

  camera->yaw += xoffset;
  camera->pitch += yoffset;

  if (camera->pitch > 89.0f) camera->pitch = 89.0f;
  if (camera->pitch < -89.0f) camera->pitch = -89.0f;

  glm::vec3 front;
  front.x = std::cos(glm::radians(camera->yaw)) * std::cos(glm::radians(camera->pitch));
  front.y = std::sin(glm::radians(camera->pitch));
  front.z = std::sin(glm::radians(camera->yaw)) * std::cos(glm::radians(camera->pitch));
  camera->camera_front = glm::normalize(front);
}
