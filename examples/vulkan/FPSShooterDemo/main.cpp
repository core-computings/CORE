#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "Game.h"
#include "RenderCubes.h"
#include "RenderFloor.h"
#include "RenderGunModel.h"
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

// Simple first-person shooter demo built on the Vulkan helper library.
//  - WASD moves the player along the ground; the mouse looks around.
//  - Left mouse button fires bullets from the gun muzzle.
//  - Bullets collide with the cube targets, which get knocked back and tumble
//    before respawning.
const uint32_t kWidth = 1280;
const uint32_t kHeight = 800;
const VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

const float kEyeHeight = 1.6f;
const float kMoveSpeed = 6.0f;        // units per second
const float kJumpSpeed = 5.6f;        // jump apex ~1.1, enough to climb one cube
const float kPlayerGravity = -14.0f;  // gravity applied while airborne
const float kMouseSensitivity = 0.08f;
const float kArenaHalf = 22.0f;     // keep the player inside the arena
const float kFireCooldown = 0.12f;  // seconds between shots

// Real weapon model + texture ("Cerberus" by Andrew Maximov, see
// examples/data/assets/ATTRIBUTION.txt). FPS_ASSET_DIR is baked in at compile time (see
// examples/CMakeLists.txt) so the demo runs from any working directory; the
// fallback keeps it working if launched from repo root.
#ifndef FPS_ASSET_DIR
#define FPS_ASSET_DIR "./examples/data/assets"
#endif
const std::string kGunModelPath = std::string(FPS_ASSET_DIR) + "/cerberus.obj";
const std::string kGunTexturePath = std::string(FPS_ASSET_DIR) + "/cerberus_albedo.png";

const glm::vec3 kWorldUp = glm::vec3(0.0f, 1.0f, 0.0f);

std::unique_ptr<core::vulkan::VulkanCamera> camera = std::make_unique<core::vulkan::VulkanCamera>(
    glm::vec3(0.0f, kEyeHeight, 2.0f), glm::vec3(0.0f, 0.0f, -1.0f), kWorldUp, kMoveSpeed);

float last_x = kWidth / 2.0f;
float last_y = kHeight / 2.0f;
bool first_mouse = true;

void process_inputs(GLFWwindow* window, float dt, const std::vector<core::Target>& targets);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);

int main() {
  VkSurfaceKHR window_surface = VK_NULL_HANDLE;
  GLFWwindow* window;
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window = glfwCreateWindow(kWidth, kHeight, "FPS Shooter Demo", nullptr, nullptr);
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

  // Manually-managed depth image for the dynamic rendering path.
  core::vulkan::VulkanImage depth_image(
      &context, swap_chain->swapchain_extent.width, swap_chain->swapchain_extent.height,
      kDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_TILING_OPTIMAL);
  depth_image.TransitionDepthImageLayout(
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, kDepthFormat);

  core::vulkan::VulkanCommandBuffer command_buffer(&context);
  core::vulkan::VulkanDynamicRendering dynamic_rendering(&context);
  core::vulkan::VulkanSemaphore image_available_semaphore(&context);
  // One render-finished (present) semaphore per swapchain image. A binary
  // semaphore used for presentation cannot be reused until its image is
  // re-acquired, so indexing by image avoids the swapchain semaphore-reuse
  // validation error.
  std::vector<std::unique_ptr<core::vulkan::VulkanSemaphore>> render_finished_semaphores;
  core::vulkan::VulkanFence in_flight_fence(&context);

  core::vulkan::DynamicRenderingInfo dynamic_rendering_info{};
  dynamic_rendering_info.color_formats = {swap_chain->swapchain_image_format};
  dynamic_rendering_info.depth_format = kDepthFormat;

  auto floor = std::make_unique<core::RenderFloor>(&context, dynamic_rendering_info);
  auto targets = std::make_unique<core::RenderCubes>(&context, dynamic_rendering_info, 16);
  auto bullets = std::make_unique<core::RenderCubes>(&context, dynamic_rendering_info, 64);
  auto gun = std::make_unique<core::RenderGunModel>(&context, dynamic_rendering_info);
  const VkClearValue color_clear_value{
      .color =
          {
              .float32 = {0.55f, 0.70f, 0.90f, 1.0f},
          },
  };
  const VkClearValue depth_clear_value{
      .depthStencil = {.depth = 1.0f, .stencil = 0},
  };

  core::Game game;

  float recoil = 0.0f;
  float flash = 0.0f;
  float fire_timer = 0.0f;
  for (size_t i = 0; i < swap_chain->swapchain_image_views.size(); ++i) {
    render_finished_semaphores.push_back(std::make_unique<core::vulkan::VulkanSemaphore>(&context));
  }

  core::vulkan::VulkanPipelineCache pipeline_cache(
      &context, std::filesystem::path(PIPELINE_CACHE_DIR) / "fps_shooter_demo.cache");
  floor->SetPipelineCache(&pipeline_cache);
  targets->SetPipelineCache(&pipeline_cache);
  bullets->SetPipelineCache(&pipeline_cache);
  gun->SetPipelineCache(&pipeline_cache);
  floor->Init();
  targets->Init();
  bullets->Init();
  gun->Init(kGunTexturePath, kGunModelPath);
  pipeline_cache.Save();

  auto last_time = std::chrono::high_resolution_clock::now();

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    const auto now = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float, std::chrono::seconds::period>(now - last_time).count();
    last_time = now;
    dt = std::min(dt, 0.05f);  // clamp on hitches

    process_inputs(window, dt, game.targets());

    // Shooting: fire from the muzzle along the view direction on a cooldown.
    fire_timer = std::max(0.0f, fire_timer - dt);
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && fire_timer <= 0.0f) {
      const glm::vec3 right = glm::normalize(glm::cross(camera->camera_front, kWorldUp));
      const glm::vec3 muzzle =
          camera->camera_position + camera->camera_front * 1.0f + right * 0.15f - kWorldUp * 0.12f;
      game.Fire(muzzle, camera->camera_front);
      fire_timer = kFireCooldown;
      recoil = 1.0f;
      flash = 1.0f;
    }
    recoil = std::max(0.0f, recoil - dt * 6.0f);
    flash = std::max(0.0f, flash - dt * 14.0f);

    game.Update(dt);

    // Turn game state into instance transforms.
    std::vector<core::CubeInstance> target_instances;
    target_instances.reserve(game.targets().size());
    for (const auto& t : game.targets()) {
      glm::mat4 model = glm::translate(glm::mat4(1.0f), t.position) * glm::mat4_cast(t.orientation);
      const glm::vec3 color = glm::mix(t.color, glm::vec3(1.0f, 1.0f, 1.0f), t.hit_flash);
      target_instances.push_back({model, color});
    }
    targets->SetInstances(target_instances);

    std::vector<core::CubeInstance> bullet_instances;
    for (const auto& b : game.bullets()) {
      if (!b.active) continue;
      // Orient the bullet as a streak along its velocity so the ballistic arc
      // is visible as the round drops over distance.
      const glm::vec3 dir = glm::length(b.velocity) > 1e-4f ? glm::normalize(b.velocity)
                                                            : glm::vec3(0.0f, 0.0f, -1.0f);
      const glm::vec3 up =
          std::abs(dir.y) > 0.99f ? glm::vec3(1.0f, 0.0f, 0.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
      const glm::vec3 right = glm::normalize(glm::cross(up, dir));
      const glm::vec3 up2 = glm::cross(dir, right);
      glm::mat4 rot(1.0f);
      rot[0] = glm::vec4(right, 0.0f);
      rot[1] = glm::vec4(up2, 0.0f);
      rot[2] = glm::vec4(dir, 0.0f);
      glm::mat4 model = glm::translate(glm::mat4(1.0f), b.position) * rot *
                        glm::scale(glm::mat4(1.0f), glm::vec3(0.05f, 0.05f, 0.32f));
      bullet_instances.push_back({model, glm::vec3(1.0f, 0.85f, 0.30f)});
    }
    bullets->SetInstances(bullet_instances);

    vkWaitForFences(context.logical_device, 1, &(in_flight_fence.fence), VK_TRUE, UINT64_MAX);
    in_flight_fence.Reset();

    uint32_t image_index;
    vkAcquireNextImageKHR(context.logical_device, swap_chain->swapchain, UINT64_MAX,
                          image_available_semaphore.semaphore, VK_NULL_HANDLE, &image_index);

    const glm::mat4 view = camera->GetViewMatrix();
    const glm::mat4 project =
        glm::perspective(glm::radians(60.0f),
                         static_cast<float>(swap_chain->swapchain_extent.width) /
                             static_cast<float>(swap_chain->swapchain_extent.height),
                         0.1f, 200.0f);

    floor->UpdateUniformBuffer(view, project);
    targets->UpdateUniformBuffer(view, project);
    bullets->UpdateUniformBuffer(view, project);

    // Place the weapon model in view space (lower-right), rotated so its barrel
    // points forward; recoil pulls it toward the camera and dips it slightly.
    // Cerberus is ~3 units long with its bore axis along +z at model y~0.4, so
    // scale it down and drop it so the bore sits just below the crosshair.
    const glm::mat4 viewmodel =
        glm::translate(glm::mat4(1.0f),
                       glm::vec3(0.14f, -0.23f - 0.03f * recoil, -0.55f + 0.06f * recoil)) *
        glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f)) *
        glm::scale(glm::mat4(1.0f), glm::vec3(0.2f));
    gun->UpdateUniformBuffer(project, viewmodel, flash);

    command_buffer.BeginCommandBufferWithReset();

    swap_chain->TransitionImageLayout(command_buffer.buffer(), image_index,
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    dynamic_rendering.BeginDynamicRendering(
        command_buffer.buffer(), swap_chain->swapchain_image_views[image_index],
        swap_chain->swapchain_extent, color_clear_value, depth_image.image_view, depth_clear_value);
    floor->Render(command_buffer.buffer(), swap_chain->swapchain_extent);
    targets->Render(command_buffer.buffer(), swap_chain->swapchain_extent);
    bullets->Render(command_buffer.buffer(), swap_chain->swapchain_extent);
    // Gun last; its close view-space depth keeps it in front while preserving
    // self-occlusion within the weapon mesh.
    gun->Render(command_buffer.buffer(), swap_chain->swapchain_extent);
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

void process_inputs(GLFWwindow* window, float dt, const std::vector<core::Target>& targets) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(window, true);

  // Horizontal movement only: project the view direction onto the ground plane.
  const glm::vec3 flat_front =
      glm::normalize(glm::vec3(camera->camera_front.x, 0.0f, camera->camera_front.z));
  const glm::vec3 right = glm::normalize(glm::cross(flat_front, kWorldUp));
  const float step = kMoveSpeed * dt;

  glm::vec3 pos = camera->camera_position;
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) pos += flat_front * step;
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) pos -= flat_front * step;
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) pos -= right * step;
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) pos += right * step;

  pos.x = std::clamp(pos.x, -kArenaHalf, kArenaHalf);
  pos.z = std::clamp(pos.z, -kArenaHalf, kArenaHalf);

  // Resting (not hit) cubes are solid: their sides block the player and their
  // tops can be stood on. The player is treated as a vertical capsule of
  // radius kPlayerRadius with feet kEyeHeight below the camera.
  const float kPlayerRadius = 0.3f;
  const float kStepTolerance = 0.1f;  // height the player can step up without jumping
  const float feet = pos.y - kEyeHeight;

  // Push the player out of cube sides (smallest-penetration axis).
  for (const auto& t : targets) {
    if (t.hit) continue;
    const float top = t.position.y + core::Game::kTargetHalf;
    const float bottom = t.position.y - core::Game::kTargetHalf;
    if (feet >= top - kStepTolerance || feet + kEyeHeight + 0.1f <= bottom) continue;
    const float half = core::Game::kTargetHalf + kPlayerRadius;
    const float dx = pos.x - t.position.x;
    const float dz = pos.z - t.position.z;
    if (std::abs(dx) >= half || std::abs(dz) >= half) continue;
    const float push_x = half - std::abs(dx);
    const float push_z = half - std::abs(dz);
    if (push_x < push_z) {
      pos.x += (dx >= 0.0f) ? push_x : -push_x;
    } else {
      pos.z += (dz >= 0.0f) ? push_z : -push_z;
    }
  }

  // Ground height under the player: the floor, or the highest cube top at or
  // below the feet whose footprint the player overlaps.
  float ground = 0.0f;
  for (const auto& t : targets) {
    if (t.hit) continue;
    const float top = t.position.y + core::Game::kTargetHalf;
    if (top > feet + kStepTolerance) continue;
    if (std::abs(pos.x - t.position.x) < core::Game::kTargetHalf + kPlayerRadius &&
        std::abs(pos.z - t.position.z) < core::Game::kTargetHalf + kPlayerRadius) {
      ground = std::max(ground, top);
    }
  }
  const float ground_eye = ground + kEyeHeight;

  // Jump: space launches the player when grounded; gravity brings them back
  // down onto the floor or whichever cube is underneath.
  static float vertical_velocity = 0.0f;
  const bool grounded = pos.y <= ground_eye + 1e-4f;
  if (grounded && glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
    vertical_velocity = kJumpSpeed;
  }
  vertical_velocity += kPlayerGravity * dt;
  pos.y += vertical_velocity * dt;
  if (pos.y <= ground_eye) {
    pos.y = ground_eye;
    vertical_velocity = 0.0f;
  }
  camera->camera_position = pos;
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
