#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "VulkanCommandBuffer.h"
#include "VulkanCompute.h"
#include "VulkanPipelineCache.h"
#include "VulkanRender.h"
#include "VulkanRenderPass.h"
#include "VulkanSync.h"

namespace {

VkPipelineCache last_graphics_pipeline_cache = VK_NULL_HANDLE;
int graphics_pipeline_calls = 0;

}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateGraphicsPipelines(
    VkDevice device, VkPipelineCache pipeline_cache, uint32_t create_info_count,
    const VkGraphicsPipelineCreateInfo* create_infos, const VkAllocationCallbacks* allocator,
    VkPipeline* pipelines) {
  last_graphics_pipeline_cache = pipeline_cache;
  ++graphics_pipeline_calls;
  const PFN_vkCreateGraphicsPipelines create = reinterpret_cast<PFN_vkCreateGraphicsPipelines>(
      vkGetDeviceProcAddr(device, "vkCreateGraphicsPipelines"));
  return create(device, pipeline_cache, create_info_count, create_infos, allocator, pipelines);
}

namespace core {
namespace test {
namespace {

class CacheComputeProbe : public vulkan::VulkanCompute {
 public:
  explicit CacheComputeProbe(vulkan::VulkanContext* context, std::string legacy_path = "")
      : VulkanCompute(context), legacy_path_(std::move(legacy_path)) {}

  void Bind(const vulkan::VulkanBuffer& output) {
    CreateStorageBufferDescriptorSet(0, output);
    vkUpdateDescriptorSets(context_->logical_device, static_cast<uint32_t>(writes_.size()),
                           writes_.data(), 0, nullptr);
  }

  void Run(VkCommandBuffer command) {
    vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
                            &descriptor_set_, 0, nullptr);
    vkCmdDispatch(command, 1, 1, 1);
  }

  void SaveLegacy(const std::string& path) const { SavePipelineCache(path); }
  VkPipelineCache used_cache() const { return pipeline_cache_; }

  mutable bool legacy_path_queried = false;

 private:
  std::vector<vulkan::BindingInfo> GetBindingInfo() const override {
    return {{.binding = 0,
             .descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             .shader_stage = VK_SHADER_STAGE_COMPUTE_BIT}};
  }

  const std::string GetPipelineCache() const override {
    legacy_path_queried = true;
    return legacy_path_;
  }

  const std::vector<uint32_t>& LoadShaderCode() const override {
    static const std::vector<uint32_t> code =
#include "ComputeSpecialization.comp.spv"
        ;
    return code;
  }

  std::string legacy_path_;
};

class CacheRenderProbe : public vulkan::VulkanRender {
 public:
  CacheRenderProbe(vulkan::VulkanContext* context, vulkan::VulkanRenderPass* render_pass)
      : VulkanRender(context, render_pass) {}

 private:
  std::vector<vulkan::BindingInfo> GetBindingInfo() const override {
    return {{.binding = 0,
             .descriptor_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
             .shader_stage = VK_SHADER_STAGE_VERTEX_BIT}};
  }

  const std::vector<uint32_t> LoadVertexShader() const override {
    return
#include "PipelineCacheProbe.vert.spv"
        ;
  }

  const std::vector<uint32_t> LoadFragmentShader() const override {
    return
#include "PipelineCacheProbe.frag.spv"
        ;
  }

  std::vector<VkVertexInputBindingDescription> GetVertexBindingDescriptions() const override {
    return {};
  }

  std::vector<VkVertexInputAttributeDescription> GetVertexAttributeDescriptions() const override {
    return {};
  }
};

std::vector<std::byte> ReadBytes(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  const std::vector<char> chars((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
  std::vector<std::byte> bytes(chars.size());
  std::memcpy(bytes.data(), chars.data(), chars.size());
  return bytes;
}

void WriteBytes(const std::filesystem::path& path, const std::vector<std::byte>& bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
}

VkPhysicalDeviceProperties DeviceProperties(const vulkan::VulkanContext& context) {
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(context.physical_device, &properties);
  return properties;
}

void ExpectDefaultDispatch(vulkan::VulkanContext& context, CacheComputeProbe& probe) {
  vulkan::VulkanBuffer output(
      &context, 5 * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  output.MapData([](void* bytes) { std::memset(bytes, 0xff, 5 * sizeof(uint32_t)); });
  probe.Bind(output);
  vulkan::VulkanCommandBuffer command(&context);
  command.BeginCommandBufferWithReset();
  probe.Run(command.buffer());
  const VkMemoryBarrier barrier{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .pNext = nullptr,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
  };
  vkCmdPipelineBarrier(command.buffer(), VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  vulkan::VulkanFence fence(&context);
  fence.Reset();
  command.Submit(fence.fence, vulkan::SubmitSyncInfo{});
  VK_CHECK(vkWaitForFences(context.logical_device, 1, &fence.fence, VK_TRUE, UINT64_MAX));
  const uint32_t expected[] = {std::bit_cast<uint32_t>(int32_t{-3}), std::bit_cast<uint32_t>(1.5F),
                               0U, 1U, 1U};
  output.MapData([&expected](void* bytes) {
    const uint32_t* actual = static_cast<const uint32_t*>(bytes);
    for (size_t index = 0; index < 5; ++index) {
      EXPECT_EQ(actual[index], expected[index]) << index;
    }
  });
}

class VulkanPipelineCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string cache_dir = PIPELINE_CACHE_DIR;
    const std::filesystem::path root = cache_dir.empty() ? std::filesystem::temp_directory_path()
                                                         : std::filesystem::path(cache_dir);
    directory_ = root / "vulkan_pipeline_cache_test";
    std::filesystem::remove_all(directory_);
    std::filesystem::create_directories(directory_);
  }

  void TearDown() override {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }

  size_t FileCount() const {
    size_t count = 0;
    for (const std::filesystem::directory_entry& entry :
         std::filesystem::directory_iterator(directory_)) {
      if (entry.is_regular_file()) {
        ++count;
      }
    }
    return count;
  }

  std::filesystem::path directory_;
};

TEST_F(VulkanPipelineCacheTest, SharesCacheAcrossComputeAndGraphicsAndRestoresOnNewDevice) {
  const std::filesystem::path path = directory_ / "shared.cache";
  {
    vulkan::VulkanContext context(true);
    context.Init();
    vulkan::VulkanPipelineCache cache(&context, path);
    EXPECT_FALSE(cache.restored());
    ASSERT_NE(cache.cache(), VK_NULL_HANDLE);

    CacheComputeProbe compute(&context, (directory_ / "legacy.cache").string());
    compute.SetPipelineCache(&cache);
    compute.Init();
    EXPECT_FALSE(compute.legacy_path_queried);
    EXPECT_EQ(compute.used_cache(), cache.cache());

    vulkan::VulkanRenderPass render_pass(&context, VK_FORMAT_R8G8B8A8_UNORM);
    CacheRenderProbe render(&context, &render_pass);
    render.SetPipelineCache(&cache);
    render.Init();
    ASSERT_NE(render.pipeline, VK_NULL_HANDLE);

    ASSERT_TRUE(cache.Save());
    EXPECT_EQ(FileCount(), 1U);
    EXPECT_TRUE(
        vulkan::VulkanPipelineCache::IsCompatible(ReadBytes(path), DeviceProperties(context)));
    ExpectDefaultDispatch(context, compute);
  }

  vulkan::VulkanContext context(true);
  context.Init();
  vulkan::VulkanPipelineCache restored(&context, path);
  EXPECT_TRUE(restored.restored());
  CacheComputeProbe compute(&context);
  compute.SetPipelineCache(&restored);
  compute.Init();
  vulkan::VulkanRenderPass render_pass(&context, VK_FORMAT_R8G8B8A8_UNORM);
  CacheRenderProbe render(&context, &render_pass);
  render.SetPipelineCache(&restored);
  render.Init();
  ASSERT_NE(render.pipeline, VK_NULL_HANDLE);
  ExpectDefaultDispatch(context, compute);
  EXPECT_TRUE(restored.Save());
  EXPECT_EQ(FileCount(), 1U);
}

TEST_F(VulkanPipelineCacheTest, RejectsIncompatibleDataAndReplacesItOnSave) {
  vulkan::VulkanContext context(true);
  context.Init();
  const VkPhysicalDeviceProperties properties = DeviceProperties(context);
  const std::filesystem::path path = directory_ / "pipeline.cache";
  {
    vulkan::VulkanPipelineCache cache(&context, path);
    CacheComputeProbe compute(&context);
    compute.SetPipelineCache(&cache);
    compute.Init();
    ASSERT_TRUE(cache.Save());
  }
  const std::vector<std::byte> valid = ReadBytes(path);
  ASSERT_TRUE(vulkan::VulkanPipelineCache::IsCompatible(valid, properties));

  const auto flip = [&valid](size_t offset) {
    std::vector<std::byte> bytes = valid;
    bytes[offset] ^= std::byte{0x01};
    return bytes;
  };
  std::vector<std::byte> zero_header_size = valid;
  std::fill_n(zero_header_size.begin(), 4, std::byte{0x00});
  const std::vector<std::vector<std::byte>> invalid = {
      {},
      std::vector<std::byte>(valid.begin(), valid.begin() + 31),
      std::vector<std::byte>(valid.size(), std::byte{0x5a}),
      zero_header_size,
      flip(0),
      flip(1),
      flip(4),
      flip(8),
      flip(12),
      flip(16),
      flip(31),
  };
  for (size_t index = 0; index < invalid.size(); ++index) {
    SCOPED_TRACE(index);
    EXPECT_FALSE(vulkan::VulkanPipelineCache::IsCompatible(invalid[index], properties));
    WriteBytes(path, invalid[index]);
    vulkan::VulkanPipelineCache cache(&context, path);
    EXPECT_FALSE(cache.restored());
    CacheComputeProbe compute(&context);
    compute.SetPipelineCache(&cache);
    compute.Init();
    ExpectDefaultDispatch(context, compute);
    ASSERT_TRUE(cache.Save());
    EXPECT_TRUE(vulkan::VulkanPipelineCache::IsCompatible(ReadBytes(path), properties));
    EXPECT_EQ(FileCount(), 1U);
  }

  for (const std::vector<std::byte>& header :
       {std::vector<std::byte>(valid.begin(), valid.begin() + 32), flip(8)}) {
    WriteBytes(path, header);
    std::filesystem::resize_file(path, vulkan::VulkanPipelineCache::kMaxDataSize + 1);
    vulkan::VulkanPipelineCache oversized(&context, path);
    EXPECT_FALSE(oversized.restored());
  }
  std::filesystem::remove(path);
}

TEST_F(VulkanPipelineCacheTest, GraphicsPipelinesReceiveSharedCache) {
  vulkan::VulkanContext context(true);
  context.Init();
  vulkan::VulkanPipelineCache cache(&context, directory_ / "graphics.cache");
  vulkan::VulkanRenderPass render_pass(&context, VK_FORMAT_R8G8B8A8_UNORM);

  CacheRenderProbe uncached(&context, &render_pass);
  const int calls_before = graphics_pipeline_calls;
  uncached.Init();
  ASSERT_EQ(graphics_pipeline_calls, calls_before + 1);
  EXPECT_EQ(last_graphics_pipeline_cache, VK_NULL_HANDLE);
  ASSERT_NE(uncached.pipeline, VK_NULL_HANDLE);

  CacheRenderProbe cached(&context, &render_pass);
  cached.SetPipelineCache(&cache);
  cached.Init();
  ASSERT_EQ(graphics_pipeline_calls, calls_before + 2);
  EXPECT_EQ(last_graphics_pipeline_cache, cache.cache());
  ASSERT_NE(cached.pipeline, VK_NULL_HANDLE);
}

TEST_F(VulkanPipelineCacheTest, LegacyComputePathValidatesRestoredData) {
  vulkan::VulkanContext context(true);
  context.Init();
  const VkPhysicalDeviceProperties properties = DeviceProperties(context);
  const std::filesystem::path path = directory_ / "legacy.cache";
  WriteBytes(path, std::vector<std::byte>(256, std::byte{0xa5}));

  CacheComputeProbe compute(&context, path.string());
  compute.Init();
  EXPECT_TRUE(compute.legacy_path_queried);
  EXPECT_NE(compute.used_cache(), VK_NULL_HANDLE);
  ExpectDefaultDispatch(context, compute);
  compute.SaveLegacy(path.string());
  EXPECT_TRUE(vulkan::VulkanPipelineCache::IsCompatible(ReadBytes(path), properties));

  vulkan::VulkanPipelineCache restored(&context, path);
  EXPECT_TRUE(restored.restored());
  EXPECT_EQ(FileCount(), 1U);
}

TEST_F(VulkanPipelineCacheTest, SaveFailsWithoutLeavingFiles) {
  vulkan::VulkanContext context(true);
  context.Init();

  vulkan::VulkanPipelineCache in_memory(&context, std::filesystem::path());
  EXPECT_FALSE(in_memory.restored());
  CacheComputeProbe compute(&context);
  compute.SetPipelineCache(&in_memory);
  compute.Init();
  EXPECT_FALSE(in_memory.Save());

  const std::filesystem::path missing = directory_ / "missing" / "pipeline.cache";
  vulkan::VulkanPipelineCache unwritable(&context, missing);
  EXPECT_FALSE(unwritable.restored());
  CacheComputeProbe second(&context);
  second.SetPipelineCache(&unwritable);
  second.Init();
  EXPECT_FALSE(unwritable.Save());
  EXPECT_FALSE(std::filesystem::exists(missing.parent_path()));
  EXPECT_EQ(FileCount(), 0U);
  EXPECT_FALSE(vulkan::VulkanPipelineCache::Write(context.logical_device, VK_NULL_HANDLE,
                                                  directory_ / "null.cache"));
  EXPECT_EQ(FileCount(), 0U);
}

}
}
}
