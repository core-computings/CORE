#include <gtest/gtest.h>

#include <array>
#include <span>

#include "ComputeSum.h"
#include "Mat.h"
#include "Timer.h"
#include "VulkanBuffer.h"
#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "VulkanImage.h"
#include "VulkanQueryPool.h"
#include "VulkanSync.h"
#include "VulkanUtils.h"

namespace core {
namespace test {

TEST(ComputeSum, test) {
  // Setup Vulkan
  core::vulkan::QueueFamilyType queue_family_type = core::vulkan::QueueFamilyType::Compute;
  core::vulkan::VulkanContext context(true, queue_family_type, nullptr);
  context.Init();
  core::vulkan::VulkanCommandBuffer command_buffer(&context);
  core::vulkan::VulkanFence fence(&context);
  core::vulkan::VulkanQueryPool query_pool(&context, VK_QUERY_TYPE_TIMESTAMP, 4);
  core::Mat<int, 1> mat(6000, 6000);
  mat.Fill(3);
  core::Timer t;
  const VkDeviceSize buffer_size = mat.rows() * mat.cols() * sizeof(int);

  // Create buffers
  core::vulkan::VulkanBuffer input_buffer(
      &context, buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  core::vulkan::VulkanBuffer device_input(
      &context, buffer_size,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  core::vulkan::VulkanBuffer sum_buffer(
      &context, sizeof(int),
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  core::vulkan::VulkanBuffer result_buffer(
      &context, sizeof(int), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Fill buffers
  input_buffer.MapData([&mat](void* data) { memcpy(data, mat.data(), mat.total() * sizeof(int)); });
  result_buffer.MapData([](void* data) {
    int zero = 0;
    memcpy(data, &zero, sizeof(int));
  });

  // Create and run compute sum pipeline
  std::unique_ptr<core::vulkan::ComputeSum> compute_sum =
      std::make_unique<core::vulkan::ComputeSum>(&context, device_input, sum_buffer, mat.cols(),
                                               mat.rows());
  compute_sum->Init();

  fence.Reset();

  command_buffer.BeginCommandBufferWithReset();
  query_pool.Reset(command_buffer.buffer());
  query_pool.Query(command_buffer.buffer(), 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
  const std::array<core::vulkan::BatchBufferCopy, 2> uploads = {
      core::vulkan::BatchBufferCopy{.source = &input_buffer,
                                  .destination = &device_input,
                                  .source_offset = 0,
                                  .destination_offset = 0,
                                  .size = VK_WHOLE_SIZE},
      core::vulkan::BatchBufferCopy{.source = &result_buffer,
                                  .destination = &sum_buffer,
                                  .source_offset = 0,
                                  .destination_offset = 0,
                                  .size = VK_WHOLE_SIZE}};
  core::vulkan::VulkanBuffer::CopyBuffers(command_buffer.buffer(), uploads);
  query_pool.Query(command_buffer.buffer(), 1);
  compute_sum->Run(command_buffer.buffer());
  query_pool.Query(command_buffer.buffer(), 2);
  const core::vulkan::BatchBufferCopy download{
      .source = &sum_buffer,
      .destination = &result_buffer,
      .source_offset = 0,
      .destination_offset = 0,
      .size = VK_WHOLE_SIZE};
  core::vulkan::VulkanBuffer::CopyBuffers(command_buffer.buffer(), std::span(&download, 1));
  query_pool.Query(command_buffer.buffer(), 3);
  VkSubmitInfo submit_info{};
  command_buffer.Submit(fence.fence, submit_info);

  ASSERT_EQ(vkWaitForFences(context.logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX),
            VK_SUCCESS);

  // Check data
  int result = 0;
  result_buffer.MapData([&result](void* data) { memcpy(&result, data, sizeof(int)); });
  printf("GPU Result: %d\n", result);

  query_pool.GetQueryResults();
  const double upload_ms = query_pool.GetElapsedMilliseconds(0, 1);
  const double compute_ms = query_pool.GetElapsedMilliseconds(1, 2);
  const double download_ms = query_pool.GetElapsedMilliseconds(2, 3);
  const double total_ms = query_pool.GetElapsedMilliseconds(0, 3);
  printf("GPU upload/compute/download/total: %f/%f/%f/%fms\n", upload_ms, compute_ms,
         download_ms, total_ms);
  EXPECT_GE(upload_ms, 0.0);
  EXPECT_GT(compute_ms, 0.0);
  EXPECT_GE(download_ms, 0.0);
  EXPECT_NEAR(upload_ms + compute_ms + download_ms, total_ms, 1e-6);

  int cpu_sum = 0;
  t.start();
  for (int row = 0; row < mat.rows(); ++row) {
    for (int col = 0; col < mat.cols(); ++col) {
      cpu_sum += *mat(row, col);
    }
  }
  t.end();
  printf("CPU Result: %d, run time: %fms\n", cpu_sum, t.time());

  EXPECT_EQ(result, mat.rows() * mat.cols() * 3);
}

}  // namespace test
}  // namespace core
