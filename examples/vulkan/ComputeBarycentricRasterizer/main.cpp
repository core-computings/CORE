#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <charconv>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "ComputeBarycentric.h"
#include "Mat.h"
#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "VulkanImage.h"
#include "VulkanQueryPool.h"
#include "VulkanSync.h"
#include "VulkanUtils.h"

const int kWidth = 1000;
const int kHeight = 1000;

int main(int argc, char* argv[]) {
  core::example::ComputeBarycentric::WorkgroupSize workgroup{};
  if (argc != 1 && argc != 3) {
    throw std::invalid_argument(
        "Usage: vk_rasterize_demo [workgroup_x workgroup_y]");
  }
  if (argc == 3) {
    uint32_t* dimensions[] = {&workgroup.x, &workgroup.y};
    for (int index = 0; index < 2; ++index) {
      const char* begin = argv[index + 1];
      const char* end = begin + std::strlen(begin);
      const std::from_chars_result result =
          std::from_chars(begin, end, *dimensions[index]);
      if (result.ec != std::errc{} || result.ptr != end) {
        throw std::invalid_argument(
            "Workgroup dimensions must be unsigned integers");
      }
    }
  }
  // Setup Vulkan
  core::vulkan::QueueFamilyType queue_family_type = core::vulkan::QueueFamilyType::Compute;
  core::vulkan::VulkanContext context(true, queue_family_type, nullptr);
  context.Init();
  core::vulkan::VulkanCommandBuffer command_buffer(&context);
  core::vulkan::VulkanFence fence(&context);
  core::vulkan::VulkanQueryPool query_pool(&context, VK_QUERY_TYPE_TIMESTAMP);

  std::unique_ptr<core::example::ComputeBarycentric> barycentric =
      std::make_unique<core::example::ComputeBarycentric>(&context, kWidth,
                                                          kHeight, workgroup);
  barycentric->Init();

  fence.Reset();

  command_buffer.BeginCommandBufferWithReset();
  query_pool.Reset(command_buffer.buffer());
  query_pool.Query(command_buffer.buffer(), 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  barycentric->Run(command_buffer.buffer());
  barycentric->Run(
      command_buffer.buffer(),
      core::example::ComputeBarycentric::Triangle{.a = glm::ivec2(50, 450),
                                                  .b = glm::ivec2(150, 50),
                                                  .c = glm::ivec2(250, 450)});
  query_pool.Query(command_buffer.buffer(), 1, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  VkMemoryBarrier host_barrier{};
  host_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  host_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(
      command_buffer.buffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_barrier, 0, nullptr, 0, nullptr);
  command_buffer.Submit(fence.fence, core::vulkan::SubmitSyncInfo{});

  vkWaitForFences(context.logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX);

  query_pool.GetQueryResults();
  const auto timestamps = query_pool.GetTimeStamps();
  const auto runtime_ms = (timestamps[1] - timestamps[0]) * (context.timestamp_period / 1000000.0);
  printf("GPU time: %fms\n", runtime_ms);

  core::Mat<int, 1> result(kHeight, kWidth);
  barycentric->rasterized.MapData(
      [&result](void* data) { memcpy(result.data(), data, result.total() * sizeof(int)); });

  printf("rasterization at (1, 1): %d\n", *result(1, 1));
  printf("rasterization at (500, 600): %d\n", *result(500, 600));

  std::vector<unsigned char> pixels(kWidth * kHeight);
  for (int y = 0; y < kHeight; ++y) {
    for (int x = 0; x < kWidth; ++x) {
      pixels[y * kWidth + x] = static_cast<unsigned char>(*result(y, x) ? 255 : 0);
    }
  }

  const bool write_status =
      stbi_write_png("./rasterized.png", kWidth, kHeight, 1, pixels.data(), kWidth);
  if (write_status) {
    printf("saved rasterization results successfully");
  } else {
    throw std::runtime_error("could not save rasterized results");
  }

  return 0;
}
