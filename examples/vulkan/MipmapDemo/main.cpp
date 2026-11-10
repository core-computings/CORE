#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>

#include <memory>
#include <vector>

#include "GlfwSwapChainResizer.h"
#include "RenderModel.h"
#include "VulkanCamera.h"
#include "VulkanCommandBuffer.h"
#include "VulkanDynamicRendering.h"
#include "VulkanSwapChain.h"
#include "VulkanSync.h"
#include "VulkanUtils.h"

void process_inputs(GLFWwindow* window);

const uint32_t kWidth = 1000;
const uint32_t kHeight = 1000;
const bool kEnableDepthBuffer = true;
const std::string kModelPath = "./examples/data/viking_room.obj";
const std::string kTexturePath = "./examples/data/viking_room.png";
const glm::vec3 kCameraPos = glm::vec3(0.0f, 1.5f, 3.0f);
const glm::vec3 kCameraFront = glm::vec3(0.0f, -1.0f, -3.0f);  // look toward -Z
const glm::vec3 kCameraUp = glm::vec3(0.0f, 1.0f, 0.0f);       // Y-up
const float kCameraSpeed = 0.03f;
float model_rotation = -90.0f;

std::unique_ptr<core::vulkan::VulkanCamera> camera =
    std::make_unique<core::vulkan::VulkanCamera>(kCameraPos, kCameraFront, kCameraUp, kCameraSpeed);

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

  // imgui setup
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();

  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForVulkan(window, true);
  // Field order must match ImGui_ImplVulkan_InitInfo's declaration order and
  // avoid nested designators (e.g. `.PipelineInfoMain.Field = ...`), which is
  // a C99 extension Clang accepts leniently on some toolchains but rejects
  // under -Werror on others.
  ImGui_ImplVulkan_InitInfo init_info = {
      .ApiVersion = VK_API_VERSION_1_3,
      .Instance = context.instance,
      .PhysicalDevice = context.physical_device,
      .Device = context.logical_device,
      .QueueFamily = context.GetQueueFamilyIndices().graphics_family.value(),
      .Queue = context.graphics_queue(),
      .DescriptorPool = VK_NULL_HANDLE,
      .DescriptorPoolSize = 128,  // >= IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE
      .MinImageCount = 2,
      .ImageCount = static_cast<uint32_t>(swap_chain->swapchain_images.size()),
      .PipelineCache = VK_NULL_HANDLE,
      .PipelineInfoMain =
          ImGui_ImplVulkan_PipelineInfo{
              .PipelineRenderingCreateInfo =
                  VkPipelineRenderingCreateInfoKHR{
                      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
                      .colorAttachmentCount = 1,
                      .pColorAttachmentFormats = &(swap_chain->swapchain_image_format)}},
      .UseDynamicRendering = true,
      .Allocator = VK_NULL_HANDLE};
  ImGui_ImplVulkan_Init(&init_info);

  core::vulkan::VulkanCommandBuffer command_buffer(&context);
  core::vulkan::VulkanDynamicRendering dynamic_rendering(&context);
  core::vulkan::VulkanFence fence(&context);
  core::vulkan::VulkanSemaphore image_available_semaphore(&context);
  core::vulkan::VulkanFence in_flight_fence(&context);
  // Dynamic rendering
  core::vulkan::DynamicRenderingInfo dynamic_rendering_info{};
  dynamic_rendering_info.color_formats = {swap_chain->swapchain_image_format};
  std::unique_ptr<core::RenderModel> model =
      std::make_unique<core::RenderModel>(&context, dynamic_rendering_info);
  const VkClearValue clear_value{
      .color =
          {
              .float32 = {0.0f, 0.0f, 0.0f, 1.0f},
          },
  };

  model->Init(kTexturePath, kModelPath);

  core::GlfwSwapChainResizer resizer(window, swap_chain.get());

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    process_inputs(window);

    // imgui
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // UI to control model rotation
    ImGui::Begin("Model Rotation Control");
    ImGui::SliderFloat("rotation", &model_rotation, -180.0f, 180.0f);
    ImGui::End();

    ImGui::Render();

    vkWaitForFences(context.logical_device, 1, &(in_flight_fence.fence), VK_TRUE, UINT64_MAX);

    uint32_t image_index = 0;
    const VkResult acquire_result =
        resizer.AcquireNextImage(image_available_semaphore.semaphore, &image_index);
    if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
      static_cast<void>(resizer.Recreate());
      continue;
    }
    in_flight_fence.Reset();
    const auto camera_view = camera->GetViewMatrix();
    model->UpdateUniformBuffer(swap_chain->swapchain_extent.width,
                               swap_chain->swapchain_extent.height, camera_view, model_rotation);
    command_buffer.BeginCommandBufferWithReset();

    swap_chain->TransitionImageLayout(command_buffer.buffer(), image_index,
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    dynamic_rendering.BeginDynamicRendering(command_buffer.buffer(),
                                            swap_chain->swapchain_image_views[image_index],
                                            swap_chain->swapchain_extent, clear_value);
    model->Render(command_buffer.buffer(), swap_chain->swapchain_extent);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffer.buffer());
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

  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

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
