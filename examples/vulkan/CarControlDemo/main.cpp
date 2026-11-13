#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "RenderCar.h"
#include "RenderGround.h"
#include "RenderSkybox.h"
#include "VulkanCommandBuffer.h"
#include "VulkanDynamicRendering.h"
#include "VulkanImage.h"
#include "VulkanSwapChain.h"
#include "VulkanSync.h"
#include "VulkanUtils.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

// Demo: third-person car controller. W/S drive the car along its heading;
// A/D only steer while the car is being driven (W or S held), matching how a
// real car cannot turn while stationary. The mouse orbits the camera around
// the car (yaw/pitch).
const uint32_t kWidth = 1280;
const uint32_t kHeight = 800;
const VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
const std::string kGroundTexturePath = "./examples/data/asphalt.jpg";
// Skybox face order is Vulkan cubemap layer order: +X, -X, +Y, -Y, +Z, -Z.
const std::array<std::string, 6> kSkyboxFacePaths = {
    "./examples/data/skybox/right.jpg", "./examples/data/skybox/left.jpg",
    "./examples/data/skybox/top.jpg",   "./examples/data/skybox/bottom.jpg",
    "./examples/data/skybox/front.jpg", "./examples/data/skybox/back.jpg",
};

constexpr float kDriveSpeed = 8.0f;                // world units per second
constexpr float kTurnRate = glm::radians(110.0f);  // radians per second while driving
constexpr float kMouseSensitivity = 0.15f;

// Camera orbit parameters.
constexpr float kCameraDistance = 12.0f;
constexpr float kCameraHeight = 1.8f;  // height of the orbit target above the floor
constexpr float kPitchMin = -5.0f;
constexpr float kPitchMax = 70.0f;

// Mouse state.
float last_x = kWidth / 2.0f;
float last_y = kHeight / 2.0f;
bool first_mouse = true;

// Camera state (orbit around the car). The car's hood points along +Z (its
// body-frame forward), so the camera starts at -Z to look at the car's rear.
float camera_yaw = -90.0f;
float camera_pitch = 18.0f;

void mouse_callback(GLFWwindow* window, double xpos, double ypos);

int main() {
  VkSurfaceKHR window_surface = VK_NULL_HANDLE;
  GLFWwindow* window;
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window = glfwCreateWindow(kWidth, kHeight, "Car Control Demo", nullptr, nullptr);
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

  // Manually-managed depth image for dynamic rendering.
  core::vulkan::VulkanImage depth_image(
      &context, swap_chain->swapchain_extent.width, swap_chain->swapchain_extent.height,
      kDepthFormat,
      VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      VK_IMAGE_ASPECT_DEPTH_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_TILING_OPTIMAL);
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

  auto skybox = std::make_unique<core::RenderSkybox>(&context, dynamic_rendering_info);
  auto ground = std::make_unique<core::RenderGround>(&context, dynamic_rendering_info);
  auto car = std::make_unique<core::RenderCar>(&context, dynamic_rendering_info);
  const VkClearValue color_clear_value{
      .color =
          {
              .float32 = {0.08f, 0.10f, 0.14f, 1.0f},
          },
  };
  const VkClearValue depth_clear_value{
      .depthStencil = {.depth = 1.0f, .stencil = 0},
  };

  // Mutable car state.
  glm::vec3 car_pos(0.0f, 0.0f, 0.0f);
  float car_yaw = 0.0f;  // radians; car's local +Z is its forward direction.
  for (size_t i = 0; i < swap_chain->swapchain_images.size(); ++i) {
    render_finished_semaphores.push_back(std::make_unique<core::vulkan::VulkanSemaphore>(&context));
  }

  skybox->Init(kSkyboxFacePaths);
  ground->Init(kGroundTexturePath);
  car->Init();

  auto last_time = std::chrono::high_resolution_clock::now();

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

    const auto now = std::chrono::high_resolution_clock::now();
    const float dt =
        std::chrono::duration<float, std::chrono::seconds::period>(now - last_time).count();
    last_time = now;

    // W/S → throttle (forward / reverse); A/D → steering. The car can only
    // steer while it is being driven, so A or D alone do nothing.
    const bool key_w = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
    const bool key_s = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
    const bool key_a = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
    const bool key_d = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;

    const float throttle = (key_w ? 1.0f : 0.0f) - (key_s ? 1.0f : 0.0f);
    const float steer = (key_d ? 1.0f : 0.0f) - (key_a ? 1.0f : 0.0f);

    if (throttle != 0.0f) {
      // Subtract: a positive rotation about +Y is CCW (from above), which
      // tilts the heading toward +X = the player's screen left. Pressing D
      // must therefore *decrease* yaw to swing the car toward screen right.
      car_yaw -= steer * kTurnRate * dt;
      const glm::vec3 car_forward(std::sin(car_yaw), 0.0f, std::cos(car_yaw));
      car_pos += car_forward * (throttle * kDriveSpeed * dt);
    }

    // Build the camera transform: orbit at fixed distance around a target
    // above the car's center, controlled by the mouse.
    const float cy = std::cos(glm::radians(camera_yaw));
    const float sy = std::sin(glm::radians(camera_yaw));
    const float cp = std::cos(glm::radians(camera_pitch));
    const float sp = std::sin(glm::radians(camera_pitch));
    const glm::vec3 cam_dir_from_target(cy * cp, sp, sy * cp);
    const glm::vec3 target = car_pos + glm::vec3(0.0f, kCameraHeight, 0.0f);
    const glm::vec3 camera_pos = target + cam_dir_from_target * kCameraDistance;
    const glm::mat4 view = glm::lookAt(camera_pos, target, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::mat4 project =
        glm::perspective(glm::radians(45.0f),
                         static_cast<float>(swap_chain->swapchain_extent.width) /
                             static_cast<float>(swap_chain->swapchain_extent.height),
                         0.1f, 300.0f);

    // Car model matrix: translate to world position, then yaw around Y.
    glm::mat4 model = glm::translate(glm::mat4(1.0f), car_pos);
    model = glm::rotate(model, car_yaw, glm::vec3(0.0f, 1.0f, 0.0f));

    vkWaitForFences(context.logical_device, 1, &(in_flight_fence.fence), VK_TRUE, UINT64_MAX);
    in_flight_fence.Reset();

    uint32_t image_index;
    vkAcquireNextImageKHR(context.logical_device, swap_chain->swapchain, UINT64_MAX,
                          image_available_semaphore.semaphore, VK_NULL_HANDLE, &image_index);

    skybox->UpdateUniformBuffer(view, project);
    ground->UpdateUniformBuffer(view, project);
    car->UpdateUniformBuffer(model, view, project);

    command_buffer.BeginCommandBufferWithReset();

    swap_chain->TransitionImageLayout(command_buffer.buffer(), image_index,
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    dynamic_rendering.BeginDynamicRendering(
        command_buffer.buffer(), swap_chain->swapchain_image_views[image_index],
        swap_chain->swapchain_extent, color_clear_value, depth_image.image_view, depth_clear_value);
    // Skybox first (depth disabled) so subsequent depth-tested geometry can
    // overdraw it normally.
    skybox->Render(command_buffer.buffer(), swap_chain->swapchain_extent);
    ground->Render(command_buffer.buffer(), swap_chain->swapchain_extent);
    car->Render(command_buffer.buffer(), swap_chain->swapchain_extent);
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

void mouse_callback([[maybe_unused]] GLFWwindow* window, double xpos, double ypos) {
  if (first_mouse) {
    last_x = static_cast<float>(xpos);
    last_y = static_cast<float>(ypos);
    first_mouse = false;
  }
  float xoffset = static_cast<float>(xpos) - last_x;
  float yoffset = static_cast<float>(ypos) - last_y;  // moving mouse down tilts camera down
  last_x = static_cast<float>(xpos);
  last_y = static_cast<float>(ypos);

  xoffset *= kMouseSensitivity;
  yoffset *= kMouseSensitivity;

  camera_yaw += xoffset;
  camera_pitch += yoffset;
  if (camera_pitch < kPitchMin) camera_pitch = kPitchMin;
  if (camera_pitch > kPitchMax) camera_pitch = kPitchMax;
}
