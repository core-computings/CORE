#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "ComputeBarycentric.h"
#include "ComputeCopy.h"
#include "ComputeFma.h"
#include "ComputeGaussianBlur.h"
#include "ComputeSum.h"
#include "VulkanCommandBuffer.h"
#include "VulkanPipelineCache.h"
#include "VulkanSync.h"

namespace {

using core::vulkan::VulkanBuffer;
using core::vulkan::VulkanContext;
using core::vulkan::VulkanPipelineCache;

constexpr int kImageSize = 256;
constexpr int kCopyCount = 1024;
constexpr size_t kCopyWords = static_cast<size_t>(kCopyCount) * 4;
constexpr size_t kPipelineCount = 5;
constexpr std::array<const char*, kPipelineCount> kPipelineNames = {"gaussian_blur", "sum", "copy",
                                                                    "fma", "barycentric"};
constexpr VkMemoryPropertyFlags kHostMemory =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr VkDeviceSize kImageBytes = static_cast<VkDeviceSize>(kImageSize) * kImageSize * 4;
constexpr VkDeviceSize kCopyBytes = static_cast<VkDeviceSize>(kCopyWords) * sizeof(uint32_t);

struct Sample {
  double total_us = 0.0;
  std::array<double, kPipelineCount> pipeline_us{};
};

double Elapsed(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
      .count();
}

uint32_t Pattern(size_t index) { return static_cast<uint32_t>(index) * 2654435761U; }

class PipelineSet {
 public:
  explicit PipelineSet(VulkanContext* context)
      : context_(context),
        image_source_(context, kImageBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostMemory),
        image_destination_(context, kImageBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostMemory),
        copy_source_(context, kCopyBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostMemory),
        copy_destination_(context, kCopyBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostMemory),
        fma_output_(context, 64 * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, kHostMemory),
        blur_(context, image_source_, image_destination_, kImageSize, kImageSize),
        sum_(context, image_source_, image_destination_, kImageSize, kImageSize),
        copy_(context, copy_source_, copy_destination_, kCopyCount),
        fma_(context, fma_output_, 1, 1),
        barycentric_(context, kImageSize, kImageSize) {}

  Sample Create(const VulkanPipelineCache& cache) {
    const std::array<core::vulkan::VulkanCompute*, kPipelineCount> pipelines = {
        &blur_, &sum_, &copy_, &fma_, &barycentric_};
    Sample sample{};
    for (size_t index = 0; index < kPipelineCount; ++index) {
      const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
      pipelines[index]->SetPipelineCache(&cache);
      pipelines[index]->Init();
      sample.pipeline_us[index] = Elapsed(start);
      if (pipelines[index]->pipeline == VK_NULL_HANDLE) {
        throw std::runtime_error("Pipeline cache benchmark created a null pipeline");
      }
    }
    return sample;
  }

  void VerifyCopy() {
    copy_source_.MapData([](void* data) {
      uint32_t* values = static_cast<uint32_t*>(data);
      for (size_t index = 0; index < kCopyWords; ++index) {
        values[index] = Pattern(index);
      }
    });
    copy_destination_.MapData(
        [](void* data) { std::fill_n(static_cast<uint32_t*>(data), kCopyWords, 0U); });
    core::vulkan::VulkanCommandBuffer command(context_);
    command.BeginCommandBufferWithReset();
    copy_.Run(command.buffer());
    const VkMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    core::vulkan::VulkanFence fence(context_);
    fence.Reset();
    command.Submit(fence.fence);
    VK_CHECK(vkWaitForFences(context_->logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX));
    copy_destination_.MapData([](void* data) {
      const uint32_t* values = static_cast<const uint32_t*>(data);
      for (size_t index = 0; index < kCopyWords; ++index) {
        if (values[index] != Pattern(index)) {
          throw std::runtime_error("Pipeline cache benchmark copy mismatch");
        }
      }
    });
  }

 private:
  VulkanContext* context_;
  VulkanBuffer image_source_;
  VulkanBuffer image_destination_;
  VulkanBuffer copy_source_;
  VulkanBuffer copy_destination_;
  VulkanBuffer fma_output_;
  core::vulkan::ComputeGaussianBlur blur_;
  core::vulkan::ComputeSum sum_;
  core::vulkan::ComputeCopy copy_;
  core::vulkan::ComputeFma fma_;
  core::example::ComputeBarycentric barycentric_;
};

Sample Run(const std::filesystem::path& cache_path, bool warm) {
  VulkanContext context(false);
  context.Init();
  PipelineSet pipelines(&context);
  const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  const VulkanPipelineCache cache(&context, warm ? cache_path : std::filesystem::path());
  Sample sample = pipelines.Create(cache);
  sample.total_us = Elapsed(start);
  if (cache.restored() != warm) {
    throw std::runtime_error("Unexpected pipeline cache restore state");
  }
  pipelines.VerifyCopy();
  return sample;
}

void Prime(const std::filesystem::path& cache_path) {
  std::filesystem::remove(cache_path);
  VulkanContext context(false);
  context.Init();
  PipelineSet pipelines(&context);
  const VulkanPipelineCache cache(&context, cache_path);
  pipelines.Create(cache);
  if (!cache.Save()) {
    throw std::runtime_error("Failed to save pipeline cache");
  }
}

double Percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const size_t index = static_cast<size_t>(fraction * static_cast<double>(values.size() - 1));
  return values[index];
}

std::vector<double> Totals(const std::vector<Sample>& samples) {
  std::vector<double> totals;
  for (const Sample& sample : samples) {
    totals.push_back(sample.total_us);
  }
  return totals;
}

void Report(const char* label, const std::vector<Sample>& samples) {
  const std::vector<double> totals = Totals(samples);
  std::printf("%-5s total_us n=%zu median=%.1f p10=%.1f p90=%.1f min=%.1f max=%.1f\n", label,
              totals.size(), Percentile(totals, 0.5), Percentile(totals, 0.1),
              Percentile(totals, 0.9), Percentile(totals, 0.0), Percentile(totals, 1.0));
  for (size_t index = 0; index < kPipelineCount; ++index) {
    std::vector<double> values;
    for (const Sample& sample : samples) {
      values.push_back(sample.pipeline_us[index]);
    }
    std::printf("%-5s %-13s median_us=%.1f\n", label, kPipelineNames[index],
                Percentile(values, 0.5));
  }
}

}

int main(int argc, char** argv) {
  const std::string cache_dir = PIPELINE_CACHE_DIR;
  const std::filesystem::path default_dir =
      cache_dir.empty() ? std::filesystem::temp_directory_path() : std::filesystem::path(cache_dir);
  const std::filesystem::path cache_path =
      argc > 1 ? std::filesystem::path(argv[1]) : default_dir / "pipeline_cache_benchmark.cache";
  const int samples = argc > 2 ? std::stoi(argv[2]) : 15;
  if (samples <= 0) {
    throw std::invalid_argument("Sample count must be positive");
  }
  constexpr int kWarmups = 2;

  Prime(cache_path);
  std::vector<Sample> cold;
  std::vector<Sample> warm;
  for (int sample = -kWarmups; sample < samples; ++sample) {
    for (int turn = 0; turn < 2; ++turn) {
      const bool use_warm = ((sample + kWarmups + turn) % 2) == 0;
      const Sample result = Run(cache_path, use_warm);
      if (sample >= 0) {
        (use_warm ? warm : cold).push_back(result);
      }
    }
  }
  std::printf("cache=%s bytes=%ju\n", cache_path.string().c_str(),
              static_cast<std::uintmax_t>(std::filesystem::file_size(cache_path)));
  Report("cold", cold);
  Report("warm", warm);
  std::printf("median_speedup=%.2fx\n",
              Percentile(Totals(cold), 0.5) / Percentile(Totals(warm), 0.5));
  std::filesystem::remove(cache_path);
  return 0;
}
