#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#if defined(__ANDROID__)
#include <unistd.h>
#endif

#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "VulkanDynamicRendering.h"
#include "VulkanImage.h"
#include "VulkanSync.h"

namespace {

constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr VkSampleCountFlagBits kSamples = VK_SAMPLE_COUNT_4_BIT;
constexpr int kSets = 4;
constexpr int kRounds = 7;

struct Sample {
  int64_t allocated = 0;
  int64_t usage = 0;
};

bool SupportsMemoryBudget(VkPhysicalDevice device) {
  uint32_t count = 0;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> extensions(count);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data());
  for (const VkExtensionProperties& extension : extensions) {
    if (std::strcmp(extension.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
      return true;
    }
  }
  return false;
}

int64_t ProcessGpuMemory() {
#if defined(__ANDROID__)
  FILE* pipe = popen("dumpsys gpu --gpumem", "r");
  if (pipe == nullptr) {
    return -1;
  }
  const std::string prefix = "Proc " + std::to_string(getpid()) + " total: ";
  int64_t total = -1;
  char line[256];
  while (std::fgets(line, sizeof(line), pipe) != nullptr) {
    if (std::strncmp(line, prefix.c_str(), prefix.size()) == 0) {
      char* end = nullptr;
      const long long value = std::strtoll(line + prefix.size(), &end, 10);
      if (end != line + prefix.size() && value >= 0) {
        total = static_cast<int64_t>(value);
      }
    }
  }
  if (pclose(pipe) != 0) {
    return -1;
  }
  return total;
#else
  return -1;
#endif
}

int64_t DeviceMemoryUsage(VkPhysicalDevice device, bool budget_supported) {
  if (!budget_supported) {
    return ProcessGpuMemory();
  }
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
  budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
  VkPhysicalDeviceMemoryProperties2 properties{};
  properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
  properties.pNext = &budget;
  vkGetPhysicalDeviceMemoryProperties2(device, &properties);
  int64_t usage = 0;
  for (uint32_t i = 0; i < properties.memoryProperties.memoryHeapCount; ++i) {
    if ((properties.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) !=
        0U) {
      usage += static_cast<int64_t>(budget.heapUsage[i]);
    }
  }
  return usage;
}

Sample Measure(core::vulkan::VulkanContext& context,
               const core::vulkan::VulkanDynamicRendering& rendering,
               core::vulkan::VulkanImage& resolve, uint32_t width, uint32_t height,
               bool transient, bool budget_supported) {
  const VkImageUsageFlags transient_bit = transient ? VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT : 0U;
  const int64_t usage_before = DeviceMemoryUsage(context.physical_device, budget_supported);
  std::vector<std::unique_ptr<core::vulkan::VulkanImage>> images;
  Sample sample{};
  for (int set = 0; set < kSets; ++set) {
    images.push_back(std::make_unique<core::vulkan::VulkanImage>(
        &context, width, height, kColorFormat,
        transient_bit | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_TILING_OPTIMAL, 1, kSamples));
    core::vulkan::VulkanImage& color = *images.back();
    images.push_back(std::make_unique<core::vulkan::VulkanImage>(
        &context, width, height, kDepthFormat,
        transient_bit | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_IMAGE_TILING_OPTIMAL, 1, kSamples));
    core::vulkan::VulkanImage& depth = *images.back();
    for (const core::vulkan::VulkanImage* image : {&color, &depth}) {
      VkMemoryRequirements requirements{};
      vkGetImageMemoryRequirements(context.logical_device, image->image, &requirements);
      sample.allocated += static_cast<int64_t>(requirements.size);
    }

    const core::vulkan::VulkanCommandBuffer command =
        core::vulkan::VulkanCommandBuffer::BeginOneTimeCommands(&context);
    color.TransitionImageLayout(command.buffer(), VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, kColorFormat);
    resolve.TransitionImageLayout(command.buffer(), VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, kColorFormat);
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = depth.image;
    barrier.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 1};
    vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, nullptr, 0, nullptr,
                         1, &barrier);
    const VkClearValue color_clear{.color = {.float32 = {0.0f, 0.0f, 0.0f, 1.0f}}};
    const VkClearValue depth_clear{.depthStencil = {.depth = 1.0f, .stencil = 0}};
    rendering.BeginDynamicRendering(command.buffer(), color.image_view, resolve.image_view,
                                    VkExtent2D{.width = width, .height = height}, color_clear,
                                    depth.image_view, depth_clear);
    rendering.EndDynamicRendering(command.buffer());
    core::vulkan::VulkanFence fence(&context);
    fence.Reset();
    command.Submit(fence.fence);
    VK_CHECK(vkWaitForFences(context.logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX));
  }
  const int64_t usage_after = DeviceMemoryUsage(context.physical_device, budget_supported);
  sample.usage = usage_before < 0 || usage_after < 0 ? -1 : usage_after - usage_before;
  if (transient) {
    for (const std::unique_ptr<core::vulkan::VulkanImage>& image : images) {
      if ((image->memory_properties() & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) == 0U) {
        std::printf("warning: transient image did not receive lazily allocated memory\n");
      }
    }
  }
  return sample;
}

int64_t Median(std::vector<int64_t> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

double MiB(int64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

}

int main() {
  core::vulkan::VulkanContext context(false);
  context.Init();
  uint32_t family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device, &family_count, nullptr);
  std::vector<VkQueueFamilyProperties> families(family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(context.physical_device, &family_count,
                                           families.data());
  const uint32_t family = context.GetQueueFamilyIndices().compute_family.value();
  if ((families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0U) {
    std::printf("unsupported: selected queue family does not support graphics\n");
    return 1;
  }
  for (const VkImageUsageFlags usage :
       {static_cast<VkImageUsageFlags>(VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT),
        static_cast<VkImageUsageFlags>(VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)}) {
    VkImageFormatProperties format_properties{};
    if (vkGetPhysicalDeviceImageFormatProperties(context.physical_device, kDepthFormat,
                                                 VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, usage,
                                                 0, &format_properties) != VK_SUCCESS ||
        (format_properties.sampleCounts & kSamples) == 0U) {
      std::printf("unsupported: 4x D32 depth attachments are not available\n");
      return 1;
    }
  }
  const bool budget_supported = SupportsMemoryBudget(context.physical_device);
  const core::vulkan::VulkanDynamicRendering rendering(&context);
  std::printf("sets=%d rounds=%d samples=4 color=RGBA8 depth=D32 usage_source=%s\n", kSets,
              kRounds, budget_supported ? "VK_EXT_memory_budget" : "dumpsys_gpu_gpumem");

  const VkExtent2D extents[3] = {{.width = 1280, .height = 720},
                                 {.width = 1920, .height = 1080},
                                 {.width = 2560, .height = 1440}};
  for (const VkExtent2D& extent : extents) {
    core::vulkan::VulkanImage resolve(
        &context, extent.width, extent.height, kColorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Measure(context, rendering, resolve, extent.width, extent.height, false, budget_supported);
    Measure(context, rendering, resolve, extent.width, extent.height, true, budget_supported);

    std::vector<int64_t> usage[2];
    int64_t allocated[2] = {0, 0};
    for (int round = 0; round < kRounds; ++round) {
      for (int order = 0; order < 2; ++order) {
        const int mode = (round + order) % 2;
        const Sample sample = Measure(context, rendering, resolve, extent.width, extent.height,
                                      mode == 1, budget_supported);
        usage[mode].push_back(sample.usage);
        allocated[mode] = sample.allocated;
      }
    }
    for (int mode = 0; mode < 2; ++mode) {
      if (std::find(usage[mode].begin(), usage[mode].end(), -1) != usage[mode].end()) {
        std::printf("%ux%u %s allocated=%.2fMiB usage unavailable\n", extent.width,
                    extent.height, mode == 1 ? "transient " : "persistent", MiB(allocated[mode]));
        continue;
      }
      std::printf(
          "%ux%u %s allocated=%.2fMiB usage median=%.2fMiB min=%.2f max=%.2f\n", extent.width,
          extent.height, mode == 1 ? "transient " : "persistent", MiB(allocated[mode]),
          MiB(Median(usage[mode])), MiB(*std::min_element(usage[mode].begin(), usage[mode].end())),
          MiB(*std::max_element(usage[mode].begin(), usage[mode].end())));
    }
  }
  return 0;
}
