#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "VulkanContext.h"

namespace core::test {
namespace {

constexpr char kExpectedValidationErrors[] = "expected_validation_errors";

uint64_t ExpectedValidationErrors(const testing::TestInfo& test_info) {
  const testing::TestResult* result = test_info.result();
  for (int index = 0; index < result->test_property_count(); ++index) {
    const testing::TestProperty& property = result->GetTestProperty(index);
    if (std::strcmp(property.key(), kExpectedValidationErrors) == 0) {
      return std::stoull(property.value());
    }
  }
  return 0;
}

class ValidationErrorChecker : public testing::EmptyTestEventListener {
 public:
  void OnTestStart(const testing::TestInfo&) override {
    initial_count_ = vulkan::VulkanContext::TotalValidationErrorCount();
  }

  void OnTestEnd(const testing::TestInfo& test_info) override {
    const uint64_t observed = vulkan::VulkanContext::TotalValidationErrorCount() - initial_count_;
    EXPECT_EQ(observed, ExpectedValidationErrors(test_info))
        << "Vulkan validation errors reported during " << test_info.test_suite_name() << "."
        << test_info.name();
  }

 private:
  uint64_t initial_count_ = 0;
};

const bool kValidationErrorCheckerInstalled = [] {
  testing::UnitTest::GetInstance()->listeners().Append(new ValidationErrorChecker());
  return true;
}();

bool ValidationLayerAvailable() {
  uint32_t count = 0;
  VK_CHECK(vkEnumerateInstanceLayerProperties(&count, nullptr));
  std::vector<VkLayerProperties> layers(count);
  VK_CHECK(vkEnumerateInstanceLayerProperties(&count, layers.data()));
  for (const VkLayerProperties& layer : layers) {
    if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
      return true;
    }
  }
  return false;
}

void SubmitMessage(VkInstance instance, VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                   VkDebugUtilsMessageTypeFlagsEXT type) {
  const auto submit = reinterpret_cast<PFN_vkSubmitDebugUtilsMessageEXT>(
      vkGetInstanceProcAddr(instance, "vkSubmitDebugUtilsMessageEXT"));
  ASSERT_NE(submit, nullptr);
  const VkDebugUtilsMessengerCallbackDataEXT callback_data{
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CALLBACK_DATA_EXT,
      .pNext = nullptr,
      .flags = 0,
      .pMessageIdName = "CORE-validation-test",
      .messageIdNumber = 0,
      .pMessage = "Injected message from VulkanValidationTest",
      .queueLabelCount = 0,
      .pQueueLabels = nullptr,
      .cmdBufLabelCount = 0,
      .pCmdBufLabels = nullptr,
      .objectCount = 0,
      .pObjects = nullptr,
  };
  submit(instance, severity, type, &callback_data);
}

TEST(VulkanValidationTest, ReportsLayerAvailability) {
  vulkan::VulkanContext enabled(true, vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
  EXPECT_EQ(enabled.validation_layers_enabled(), ValidationLayerAvailable());
  enabled.Init();
  EXPECT_EQ(enabled.validation_error_count(), 0u);

  vulkan::VulkanContext disabled(false, vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
  EXPECT_FALSE(disabled.validation_layers_enabled());
  EXPECT_EQ(disabled.logical_device, VK_NULL_HANDLE);
  EXPECT_EQ(disabled.physical_device, VK_NULL_HANDLE);
  disabled.Init();
  EXPECT_NE(disabled.logical_device, VK_NULL_HANDLE);
  EXPECT_EQ(disabled.validation_error_count(), 0u);
}

TEST(VulkanValidationTest, CountsErrorsBeforeAndAfterDeviceCreation) {
  vulkan::VulkanContext context(true, vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
  vulkan::VulkanContext observer(true, vulkan::QueueFamilyType::Compute, VK_NULL_HANDLE);
  observer.Init();
  const uint64_t initial_total = vulkan::VulkanContext::TotalValidationErrorCount();

  SubmitMessage(context.instance, VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT);
  EXPECT_EQ(context.validation_error_count(), 1u);

  context.Init();
  EXPECT_EQ(context.validation_error_count(), 1u);
  SubmitMessage(context.instance, VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT);
  SubmitMessage(context.instance, VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
                VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT);
  SubmitMessage(context.instance, VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT);
  EXPECT_EQ(context.validation_error_count(), 2u);

  EXPECT_EQ(observer.validation_error_count(), 0u);
  EXPECT_EQ(vulkan::VulkanContext::TotalValidationErrorCount() - initial_total, 2u);
  RecordProperty(kExpectedValidationErrors, 2);
}

}
}
