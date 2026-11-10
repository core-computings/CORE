#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "VulkanCommandBuffer.h"
#include "VulkanContext.h"
#include "VulkanDynamicRendering.h"
#include "VulkanRenderPass.h"
#include "VulkanSwapChain.h"
#include "VulkanSync.h"

#if defined(__ANDROID__)
#include <android/native_window.h>
#include <media/NdkImageReader.h>
#include <vulkan/vulkan_android.h>
#endif

struct SwapChainCalls {
  bool fail_swapchain_creation;
  bool fail_semaphore_creation;
  bool hold_acquire_fence;
  bool acquire_fence_eligible;
  bool eligible_fence_held;
  int created_swapchains;
  int destroyed_swapchains;
  int destroyed_with_unsignaled_acquire;
  VkFence acquire_fence;
  VkSwapchainKHR acquire_swapchain;
  uint64_t acquired_images;
};

SwapChainCalls g_calls{};

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateSwapchainKHR(
    VkDevice device, const VkSwapchainCreateInfoKHR* create_info,
    const VkAllocationCallbacks* allocator, VkSwapchainKHR* swapchain) {
  if (g_calls.fail_swapchain_creation) {
    return VK_ERROR_INITIALIZATION_FAILED;
  }
  const VkResult result = reinterpret_cast<PFN_vkCreateSwapchainKHR>(vkGetDeviceProcAddr(
      device, "vkCreateSwapchainKHR"))(device, create_info, allocator, swapchain);
  g_calls.created_swapchains += (result == VK_SUCCESS) ? 1 : 0;
  return result;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkDestroySwapchainKHR(
    VkDevice device, VkSwapchainKHR swapchain, const VkAllocationCallbacks* allocator) {
  if (g_calls.acquire_fence != VK_NULL_HANDLE &&
      vkGetFenceStatus(device, g_calls.acquire_fence) != VK_SUCCESS) {
    ++g_calls.destroyed_with_unsignaled_acquire;
  }
  ++g_calls.destroyed_swapchains;
  reinterpret_cast<PFN_vkDestroySwapchainKHR>(vkGetDeviceProcAddr(
      device, "vkDestroySwapchainKHR"))(device, swapchain, allocator);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkAcquireNextImageKHR(
    VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout, VkSemaphore semaphore,
    VkFence fence, uint32_t* image_index) {
  const VkResult result = reinterpret_cast<PFN_vkAcquireNextImageKHR>(vkGetDeviceProcAddr(
      device, "vkAcquireNextImageKHR"))(device, swapchain, timeout, semaphore, fence,
                                        image_index);
  if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
    return result;
  }
  if (swapchain != g_calls.acquire_swapchain) {
    g_calls.acquire_swapchain = swapchain;
    g_calls.acquired_images = 0;
  }
  if (*image_index >= 64u) {
    ADD_FAILURE() << "Swapchain image index " << *image_index << " exceeds the tracking mask";
    return result;
  }
  const uint64_t image_bit = uint64_t{1} << *image_index;
  if (fence != VK_NULL_HANDLE) {
    g_calls.acquire_fence = fence;
    g_calls.acquire_fence_eligible = (g_calls.acquired_images & image_bit) != 0;
  }
  g_calls.acquired_images |= image_bit;
  return result;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkGetFenceStatus(VkDevice device, VkFence fence) {
  if (g_calls.hold_acquire_fence && g_calls.acquire_fence_eligible &&
      fence == g_calls.acquire_fence) {
    g_calls.eligible_fence_held = true;
    return VK_NOT_READY;
  }
  return reinterpret_cast<PFN_vkGetFenceStatus>(vkGetDeviceProcAddr(device, "vkGetFenceStatus"))(
      device, fence);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateSemaphore(
    VkDevice device, const VkSemaphoreCreateInfo* create_info,
    const VkAllocationCallbacks* allocator, VkSemaphore* semaphore) {
  if (g_calls.fail_semaphore_creation) {
    return VK_ERROR_OUT_OF_DEVICE_MEMORY;
  }
  return reinterpret_cast<PFN_vkCreateSemaphore>(vkGetDeviceProcAddr(
      device, "vkCreateSemaphore"))(device, create_info, allocator, semaphore);
}

namespace core {
namespace test {
namespace {

constexpr uint64_t kAcquireTimeoutNs = 2000000000ull;

class PresentationWindow {
 public:
  PresentationWindow(uint32_t width, uint32_t height) {
#if defined(__APPLE__)
    if (glfwInit() != GLFW_TRUE) {
      throw std::runtime_error("glfwInit failed");
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window_ = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), "swapchain",
                               nullptr, nullptr);
    if (window_ == nullptr) {
      glfwTerminate();
      throw std::runtime_error("glfwCreateWindow failed");
    }
#elif defined(__ANDROID__)
    if (AImageReader_newWithUsage(static_cast<int32_t>(width), static_cast<int32_t>(height),
                                  AIMAGE_FORMAT_RGBA_8888, AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT,
                                  8, &reader_) != AMEDIA_OK) {
      throw std::runtime_error("AImageReader_newWithUsage failed");
    }
    if (AImageReader_getWindow(reader_, &window_) != AMEDIA_OK) {
      AImageReader_delete(reader_);
      throw std::runtime_error("AImageReader_getWindow failed");
    }
    extent_ = VkExtent2D{.width = width, .height = height};
#endif
  }

  PresentationWindow(const PresentationWindow&) = delete;
  PresentationWindow& operator=(const PresentationWindow&) = delete;

  ~PresentationWindow() {
#if defined(__APPLE__)
    glfwDestroyWindow(window_);
    glfwTerminate();
#elif defined(__ANDROID__)
    AImageReader_delete(reader_);
#endif
  }

  VkSurfaceKHR CreateSurface(VkInstance instance) const {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
#if defined(__APPLE__)
    VK_CHECK(glfwCreateWindowSurface(instance, window_, nullptr, &surface));
#elif defined(__ANDROID__)
    const VkAndroidSurfaceCreateInfoKHR create_info{
        .sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = 0,
        .window = window_,
    };
    VK_CHECK(vkCreateAndroidSurfaceKHR(instance, &create_info, nullptr, &surface));
#endif
    return surface;
  }

  void Resize(uint32_t width, uint32_t height) {
#if defined(__APPLE__)
    glfwSetWindowSize(window_, static_cast<int>(width), static_cast<int>(height));
    glfwPollEvents();
#elif defined(__ANDROID__)
    if (ANativeWindow_setBuffersGeometry(window_, static_cast<int32_t>(width),
                                         static_cast<int32_t>(height), 0) != 0) {
      throw std::runtime_error("ANativeWindow_setBuffersGeometry failed");
    }
    extent_ = VkExtent2D{.width = width, .height = height};
#endif
  }

  VkExtent2D FramebufferExtent() const {
#if defined(__APPLE__)
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    return VkExtent2D{.width = static_cast<uint32_t>(width),
                      .height = static_cast<uint32_t>(height)};
#elif defined(__ANDROID__)
    return extent_;
#endif
  }

  void ReleasePresentedImages() const {
#if defined(__ANDROID__)
    AImage* image = nullptr;
    while (AImageReader_acquireNextImage(reader_, &image) == AMEDIA_OK) {
      AImage_delete(image);
      image = nullptr;
    }
#endif
  }

 private:
#if defined(__APPLE__)
  GLFWwindow* window_ = nullptr;
#elif defined(__ANDROID__)
  AImageReader* reader_ = nullptr;
  ANativeWindow* window_ = nullptr;
  VkExtent2D extent_{};
#endif
};

class FramePresenter {
 public:
  explicit FramePresenter(vulkan::VulkanContext* context)
      : context_(context),
        command_buffer_(context),
        dynamic_rendering_(context),
        image_available_(context),
        in_flight_(context) {}

  VkResult Present(vulkan::VulkanSwapChain& swap_chain) {
    uint32_t image_index = 0;
    const VkResult acquire_result =
        swap_chain.AcquireNextImage(image_available_.semaphore, kAcquireTimeoutNs, &image_index);
    if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR) {
      return acquire_result;
    }

    in_flight_.Reset();
    command_buffer_.BeginCommandBufferWithReset();
    const VkClearValue clear_value{.color = {.float32 = {1.0f, 0.5f, 0.0f, 1.0f}}};
    swap_chain.TransitionImageLayout(command_buffer_.buffer(), image_index,
                                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    dynamic_rendering_.BeginDynamicRendering(command_buffer_.buffer(),
                                             swap_chain.swapchain_image_views.at(image_index),
                                             swap_chain.swapchain_extent, clear_value);
    dynamic_rendering_.EndDynamicRendering(command_buffer_.buffer());
    swap_chain.TransitionImageLayout(command_buffer_.buffer(), image_index,
                                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    const VkSemaphore render_finished = swap_chain.PresentSemaphore(image_index);
    command_buffer_.Submit(in_flight_.fence,
                           vulkan::SubmitSyncInfo{
                               .wait_semaphores = {image_available_.semaphore},
                               .wait_stage_masks = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
                               .signal_semaphores = {render_finished},
                           });
    const VkResult present_result = swap_chain.Present(image_index, render_finished);
    VK_CHECK(vkWaitForFences(context_->logical_device, 1, &in_flight_.fence, VK_TRUE,
                             kAcquireTimeoutNs));
    return present_result;
  }

 private:
  vulkan::VulkanContext* context_ = nullptr;
  vulkan::VulkanCommandBuffer command_buffer_;
  vulkan::VulkanDynamicRendering dynamic_rendering_;
  vulkan::VulkanSemaphore image_available_;
  vulkan::VulkanFence in_flight_;
};

void ExpectExtent(VkExtent2D actual, VkExtent2D expected) {
  EXPECT_EQ(actual.width, expected.width);
  EXPECT_EQ(actual.height, expected.height);
}

void ExpectLiveImageResources(const vulkan::VulkanSwapChain& swap_chain) {
  ASSERT_FALSE(swap_chain.swapchain_images.empty());
  ASSERT_EQ(swap_chain.swapchain_image_views.size(), swap_chain.swapchain_images.size());
  for (uint32_t i = 0; i < swap_chain.swapchain_images.size(); ++i) {
    EXPECT_NE(swap_chain.swapchain_image_views[i], VK_NULL_HANDLE);
    EXPECT_NE(swap_chain.PresentSemaphore(i), VK_NULL_HANDLE);
  }
  EXPECT_THROW(static_cast<void>(swap_chain.PresentSemaphore(
                   static_cast<uint32_t>(swap_chain.swapchain_images.size()))),
               std::out_of_range);
}

TEST(VulkanSwapChainTest, RecreateFollowsResizedSurfaceAndPresents) {
  g_calls = SwapChainCalls{};
  PresentationWindow window(96, 64);
  vulkan::VulkanContext context(true, vulkan::QueueFamilyType::Graphics, VK_NULL_HANDLE);
  const VkSurfaceKHR surface = window.CreateSurface(context.instance);
  context.Init(surface);
  vulkan::VulkanSwapChain swap_chain(&context, surface, true);
  vulkan::VulkanRenderPass render_pass(&context, swap_chain.swapchain_image_format, true);
  swap_chain.CreateFrameBuffers(render_pass);
  FramePresenter presenter(&context);

  ExpectExtent(swap_chain.swapchain_extent, window.FramebufferExtent());
  EXPECT_EQ(presenter.Present(swap_chain), VK_SUCCESS);
  window.ReleasePresentedImages();

  window.Resize(160, 120);
  const VkSwapchainKHR old_swapchain = swap_chain.swapchain;
  ASSERT_TRUE(swap_chain.Recreate(window.FramebufferExtent()));

  EXPECT_NE(swap_chain.swapchain, old_swapchain);
  EXPECT_EQ(swap_chain.RetiredSwapChainCount(), 1u);
  const size_t max_frames = 8 * swap_chain.swapchain_images.size();
  g_calls.hold_acquire_fence = true;
  for (size_t frame = 0; frame < max_frames; ++frame) {
    ASSERT_EQ(presenter.Present(swap_chain), VK_SUCCESS);
    window.ReleasePresentedImages();
  }
  ASSERT_TRUE(g_calls.eligible_fence_held);
  EXPECT_EQ(swap_chain.RetiredSwapChainCount(), 1u);
  EXPECT_EQ(g_calls.destroyed_swapchains, 0);
  g_calls.hold_acquire_fence = false;
  ExpectExtent(swap_chain.swapchain_extent, window.FramebufferExtent());
  ExpectLiveImageResources(swap_chain);
  EXPECT_TRUE(swap_chain.swapchain_framebuffers.empty());
  swap_chain.CreateFrameBuffers(render_pass);
  ASSERT_EQ(swap_chain.swapchain_framebuffers.size(), swap_chain.swapchain_images.size());
  for (const VkFramebuffer framebuffer : swap_chain.swapchain_framebuffers) {
    EXPECT_NE(framebuffer, VK_NULL_HANDLE);
  }
  for (size_t frame = 0; frame < max_frames && swap_chain.RetiredSwapChainCount() != 0; ++frame) {
    ASSERT_EQ(presenter.Present(swap_chain), VK_SUCCESS);
    window.ReleasePresentedImages();
  }
  EXPECT_EQ(swap_chain.RetiredSwapChainCount(), 0u);
  EXPECT_NE(g_calls.acquire_fence, VK_NULL_HANDLE);
  EXPECT_EQ(g_calls.destroyed_swapchains, 1);
  EXPECT_EQ(g_calls.destroyed_with_unsignaled_acquire, 0);
  VK_CHECK(vkDeviceWaitIdle(context.logical_device));
}

TEST(VulkanSwapChainTest, RecreateKeepsSwapchainForZeroExtentAndReleasesOnFailure) {
  g_calls = SwapChainCalls{};
  PresentationWindow window(96, 64);
  vulkan::VulkanContext context(true, vulkan::QueueFamilyType::Graphics, VK_NULL_HANDLE);
  const VkSurfaceKHR surface = window.CreateSurface(context.instance);
  context.Init(surface);
  g_calls.fail_semaphore_creation = true;
  EXPECT_THROW(static_cast<void>(vulkan::VulkanSwapChain(&context, surface)), std::runtime_error);
  g_calls.fail_semaphore_creation = false;
  EXPECT_EQ(g_calls.created_swapchains, 1);
  EXPECT_EQ(g_calls.destroyed_swapchains, 1);
  vulkan::VulkanSwapChain swap_chain(&context, surface);
  FramePresenter presenter(&context);

  const VkSwapchainKHR original_swapchain = swap_chain.swapchain;
  const VkExtent2D original_extent = swap_chain.swapchain_extent;
  const std::vector<VkImageView> original_views = swap_chain.swapchain_image_views;

  EXPECT_FALSE(swap_chain.Recreate(VkExtent2D{.width = 0, .height = 64}));
  EXPECT_FALSE(swap_chain.Recreate(VkExtent2D{.width = 96, .height = 0}));

  EXPECT_EQ(swap_chain.swapchain, original_swapchain);
  ExpectExtent(swap_chain.swapchain_extent, original_extent);
  EXPECT_EQ(swap_chain.swapchain_image_views, original_views);
  EXPECT_EQ(presenter.Present(swap_chain), VK_SUCCESS);
  window.ReleasePresentedImages();
  VK_CHECK(vkDeviceWaitIdle(context.logical_device));

  window.Resize(128, 80);
  g_calls.fail_swapchain_creation = true;
  EXPECT_THROW(static_cast<void>(swap_chain.Recreate(window.FramebufferExtent())),
               std::runtime_error);
  g_calls.fail_swapchain_creation = false;
  EXPECT_EQ(swap_chain.swapchain, VK_NULL_HANDLE);
  EXPECT_EQ(swap_chain.RetiredSwapChainCount(), 1u);
  EXPECT_TRUE(swap_chain.swapchain_images.empty());
  EXPECT_TRUE(swap_chain.swapchain_image_views.empty());
  EXPECT_THROW(static_cast<void>(swap_chain.Recreate(window.FramebufferExtent())),
               std::logic_error);
  swap_chain.UnInit();
  swap_chain.UnInit();
  EXPECT_EQ(swap_chain.RetiredSwapChainCount(), 0u);
}

}
}
}
