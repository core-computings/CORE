#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "RenderModel.h"
#include "VulkanCommandBuffer.h"
#include "VulkanDynamicRendering.h"
#include "VulkanSync.h"

namespace core {
namespace test {
namespace {

struct ModelFixture {
  std::filesystem::path directory =
      std::filesystem::path(PIPELINE_CACHE_DIR) /
      ("model-upload-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));

  ModelFixture() {
    std::filesystem::create_directories(directory);
    std::ofstream model(directory / "triangle.obj");
    model << "v -0.75 -0.75 0.5\nv 0.75 -0.75 0.5\nv 0 0.75 0.5\n"
             "vt 0 0\nvt 1 0\nvt 0.5 1\nf 1/1 2/2 3/3\nf 3/3 2/2 1/1\n";
    std::ofstream texture(directory / "red.ppm", std::ios::binary);
    texture << "P6\n2 2\n255\n";
    const std::array<uint8_t, 12> pixels = {255, 0, 0, 255, 0, 0,
                                            255, 0, 0, 255, 0, 0};
    texture.write(reinterpret_cast<const char*>(pixels.data()),
                  static_cast<std::streamsize>(pixels.size()));
  }

  ~ModelFixture() {
    std::error_code error;
    std::filesystem::remove_all(directory, error);
  }
};

TEST(VulkanModelUploadTest, RendersAfterInitializationResourcesLeaveScope) {
  vulkan::VulkanContext context(true, vulkan::QueueFamilyType::Compute,
                                VK_NULL_HANDLE);
  context.Init();
  uint32_t count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device, &count,
                                           nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device, &count,
                                           families.data());
  ASSERT_NE(families.at(context.GetQueueFamilyIndices().compute_family.value())
                    .queueFlags &
                VK_QUEUE_GRAPHICS_BIT,
            0U);
  constexpr VkExtent2D kExtent{.width = 64, .height = 64};
  constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
  constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
  const vulkan::DynamicRenderingInfo rendering_info{
      .color_formats = {kColorFormat},
      .depth_format = kDepthFormat,
      .stencil_format = VK_FORMAT_UNDEFINED};
  vulkan::VulkanDynamicRendering rendering(&context);
  vulkan::VulkanCommandBuffer command(&context);
  vulkan::VulkanFence fence(&context);
  vulkan::VulkanBuffer readback(&context, kExtent.width * kExtent.height * 4,
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  for (uint32_t instance = 0; instance < 3; ++instance) {
    RenderModel model(&context, rendering_info);
    {
      ModelFixture fixture;
      std::printf("MODEL_INIT_BEGIN %u\n", instance);
      model.Init((fixture.directory / "red.ppm").string(),
                 (fixture.directory / "triangle.obj").string());
      std::printf("MODEL_INIT_END %u\n", instance);
    }
    vulkan::VulkanImage color(
        &context, kExtent.width, kExtent.height, kColorFormat,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vulkan::VulkanImage depth(
        &context, kExtent.width, kExtent.height, kDepthFormat,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    for (uint32_t frame = 0; frame < 2; ++frame) {
      fence.Reset();
      command.BeginCommandBufferWithReset();
      VkImageMemoryBarrier color_barrier{};
      color_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      color_barrier.srcAccessMask =
          frame == 0 ? 0 : VK_ACCESS_TRANSFER_READ_BIT;
      color_barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      color_barrier.oldLayout = frame == 0
                                    ? VK_IMAGE_LAYOUT_UNDEFINED
                                    : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      color_barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      color_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      color_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      color_barrier.image = color.image;
      color_barrier.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                        .baseMipLevel = 0,
                                        .levelCount = 1,
                                        .baseArrayLayer = 0,
                                        .layerCount = 1};
      vkCmdPipelineBarrier(command.buffer(),
                           frame == 0 ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                      : VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0,
                           nullptr, 0, nullptr, 1, &color_barrier);
      VkImageMemoryBarrier depth_barrier = color_barrier;
      depth_barrier.srcAccessMask =
          frame == 0 ? 0 : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      depth_barrier.dstAccessMask =
          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      depth_barrier.oldLayout =
          frame == 0 ? VK_IMAGE_LAYOUT_UNDEFINED
                     : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      depth_barrier.newLayout =
          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      depth_barrier.image = depth.image;
      depth_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      vkCmdPipelineBarrier(command.buffer(),
                           frame == 0
                               ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                               : VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                               VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                           0, 0, nullptr, 0, nullptr, 1, &depth_barrier);
      const VkClearValue clear_color{
          .color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};
      const VkClearValue clear_depth{
          .depthStencil = {.depth = 1.0f, .stencil = 0}};
      rendering.BeginDynamicRendering(command.buffer(), color.image_view,
                                      kExtent, clear_color, depth.image_view,
                                      clear_depth);
      model.Render(command.buffer(), kExtent);
      rendering.EndDynamicRendering(command.buffer());
      color_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      color_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      color_barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      color_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      vkCmdPipelineBarrier(command.buffer(),
                           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                           nullptr, 1, &color_barrier);
      VkBufferImageCopy copy{};
      copy.imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                               .mipLevel = 0,
                               .baseArrayLayer = 0,
                               .layerCount = 1};
      copy.imageExtent = {
          .width = kExtent.width, .height = kExtent.height, .depth = 1};
      vkCmdCopyImageToBuffer(command.buffer(), color.image,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             readback.buffer, 1, &copy);
      const VkMemoryBarrier host_barrier{
          .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
          .pNext = nullptr,
          .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
      vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_barrier, 0,
                           nullptr, 0, nullptr);
      command.Submit(fence.fence);
      ASSERT_EQ(vkWaitForFences(context.logical_device, 1, &fence.fence,
                                VK_TRUE, UINT64_MAX),
                VK_SUCCESS);
      readback.MapData([](void* data) {
        const uint8_t* pixels = static_cast<const uint8_t*>(data);
        size_t checked = 0;
        for (int y = 0; y < 64; ++y) {
          for (int x = 0; x < 64; ++x) {
            const int edge0 = 48 * (2 * y + 1 - 16);
            const int edge1 = -24 * (2 * y + 1 - 16) - 48 * (2 * x + 1 - 112);
            const int edge2 = -24 * (2 * y + 1 - 112) + 48 * (2 * x + 1 - 64);
            if (std::abs(edge0) < 96 || std::abs(edge1) < 96 ||
                std::abs(edge2) < 96) {
              continue;
            }
            const bool inside = edge0 > 0 && edge1 > 0 && edge2 > 0;
            const size_t offset = static_cast<size_t>(y * 64 + x) * 4;
            ASSERT_EQ(pixels[offset], inside ? 255 : 0) << x << "," << y;
            ASSERT_EQ(pixels[offset + 1], 0);
            ASSERT_EQ(pixels[offset + 2], 0);
            ASSERT_EQ(pixels[offset + 3], 255);
            ++checked;
          }
        }
        EXPECT_GT(checked, 3500U);
      });
    }
  }
}

}
}
}
