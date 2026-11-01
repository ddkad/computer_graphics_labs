#include "application.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <imgui.h>

#include "math3d.hpp"
#include "octahedron.hpp"

#ifndef SHADER_DIR
#define SHADER_DIR "shaders/"
#endif

namespace application {

namespace {

using math::Mat4;
using math::Vec3;

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* mapped = nullptr;
	VkDeviceSize size = 0;
};

struct SceneData {
	float view_proj[16];
	float camera_position[4];
	float ambient[4];
	float dir_direction[4];
	float dir_color[4];
	uint32_t counts[4];
};
static_assert(sizeof(SceneData) == 144, "Unexpected scene uniform layout");

struct ModelData {
	float model[16];
	float normal_matrix[16];
	float albedo[4];
	float specular[4];
};
static_assert(sizeof(ModelData) == 160, "Unexpected model uniform layout");

struct GpuPointLight {
	float position[4];
	float color[4];
};
static_assert(sizeof(GpuPointLight) == 32, "Unexpected point light layout");

struct GpuSpotLight {
	float position[4];
	float direction[4];
	float color[4];
	float cone[4];
};
static_assert(sizeof(GpuSpotLight) == 64, "Unexpected spot light layout");

struct Mesh {
	uint32_t first_index = 0;
	uint32_t index_count = 0;
	int32_t vertex_offset = 0;
};

enum MeshId : int { MeshOctahedron = 0, MeshPlane = 1, MeshCount = 2 };

struct Material {
	float albedo[3];
	float specular[3];
	float shininess;
};

struct Object {
	const char* name;
	int mesh;
	float position[3];
	float rotation_deg[3];
	float scale[3];
	Material material;
	Buffer uniform;
	VkDescriptorSet set = VK_NULL_HANDLE;
};

enum Projection : int { Perspective = 0, Orthographic = 1 };
enum Trajectory : int { Lissajous = 0, Trefoil = 1, Spiral = 2 };

struct Camera {
	int projection = Perspective;
	float position[3] = { 0.0f, 2.5f, 11.0f };
	float yaw_deg = 0.0f;
	float pitch_deg = -8.0f;
	float roll_deg = 0.0f;
	float fov_deg = 60.0f;
	float ortho_size = 6.0f;
	float near_plane = 0.1f;
	float far_plane = 100.0f;
	float move_speed = 6.0f;
	float mouse_sensitivity = 0.2f;
};

struct DirectionalLight {
	bool enabled = true;
	float yaw_deg = -40.0f;
	float pitch_deg = -45.0f;
	float color[3] = { 1.0f, 1.0f, 1.0f };
	float intensity = 0.35f;
};

struct PointLight {
	float position[3] = { 0.0f, 2.0f, 3.0f };
	float color[3] = { 1.0f, 1.0f, 1.0f };
	float intensity = 40.0f;
};

struct SpotLight {
	float position[3] = { 0.0f, 6.0f, 2.0f };
	float target[3] = { 0.0f, -2.5f, 0.0f };
	float color[3] = { 1.0f, 1.0f, 0.9f };
	float intensity = 80.0f;
	float inner_deg = 12.0f;
	float outer_deg = 25.0f;
};

struct Animation {
	bool playing = true;
	float speed = 1.0f;
	int trajectory = Lissajous;
	float radius = 2.5f;
	float height = 0.7f;
	int freq[3] = { 3, 2, 1 };
	float phase_deg = 90.0f;
	float spin_speed_deg = 60.0f;
	float spin_axis[3] = { 0.4f, 1.0f, 0.2f };
	float clock = 0.0f;
	float spin_angle = 0.0f;
};

struct State {
	Camera camera;
	DirectionalLight directional;
	float ambient_color[3] = { 0.6f, 0.65f, 1.0f };
	float ambient_intensity = 0.08f;
	std::vector<PointLight> point_lights;
	std::vector<SpotLight> spot_lights;
	Animation animation;
	int selected_object = 0;
	double last_time = -1.0;
};

State state;
std::vector<Object> objects;
Mesh meshes[MeshCount];

Buffer vertex_buffer;
Buffer index_buffer;
Buffer scene_buffer;
Buffer point_buffer;
Buffer spot_buffer;

constexpr uint32_t initial_light_capacity = 8;

VkShaderModule shader_vert = VK_NULL_HANDLE;
VkShaderModule shader_frag = VK_NULL_HANDLE;
VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_set = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

Vec3 toVec3(const float v[3]) {
	return { v[0], v[1], v[2] };
}

Vec3 directionFromAngles(float yaw_deg, float pitch_deg) {
	const float yaw = math::radians(yaw_deg);
	const float pitch = math::radians(pitch_deg);
	return {
		-std::sin(yaw) * std::cos(pitch),
		std::sin(pitch),
		-std::cos(yaw) * std::cos(pitch),
	};
}

Vec3 trajectoryOffset(const Animation& a, float t) {
	switch (a.trajectory) {
	case Lissajous: {
		const float phase = math::radians(a.phase_deg);
		return {
			a.radius * std::sin(float(a.freq[0]) * t + phase),
			a.height * std::sin(float(a.freq[1]) * t),
			a.radius * std::sin(float(a.freq[2]) * t),
		};
	}
	case Trefoil: {
		const float k = a.radius / 3.0f;
		return {
			k * (std::sin(t) + 2.0f * std::sin(2.0f * t)),
			-a.height * std::sin(3.0f * t),
			k * (std::cos(t) - 2.0f * std::cos(2.0f * t)),
		};
	}
	case Spiral:
	default: {
		const float r = a.radius * (0.65f + 0.35f * std::sin(0.3f * t));
		return {
			r * std::cos(t),
			a.height * std::sin(float(a.freq[1]) * 0.5f * t),
			r * std::sin(t),
		};
	}
	}
}

Mat4 modelMatrix(int index) {
	const Object& o = objects[size_t(index)];

	Vec3 offset = {};
	Mat4 spin = math::identity();

	if (index == 0) {
		offset = trajectoryOffset(state.animation, state.animation.clock);
		spin = math::rotateAxis(toVec3(state.animation.spin_axis), state.animation.spin_angle);
	}

	const Mat4 T = math::translate(toVec3(o.position) + offset);
	const Mat4 R = math::rotateZ(math::radians(o.rotation_deg[2])) *
	               math::rotateY(math::radians(o.rotation_deg[1])) *
	               math::rotateX(math::radians(o.rotation_deg[0]));
	const Mat4 S = math::scale(toVec3(o.scale));

	return T * R * spin * S;
}

Mat4 buildView(const Camera& c) {
	return math::rotateZ(-math::radians(c.roll_deg)) *
	       math::rotateX(-math::radians(c.pitch_deg)) *
	       math::rotateY(-math::radians(c.yaw_deg)) *
	       math::translate(toVec3(c.position) * -1.0f);
}

Mat4 buildProjection(const Camera& c, float aspect) {
	if (c.projection == Perspective) {
		return math::perspective(math::radians(c.fov_deg), aspect, c.near_plane, c.far_plane);
	}
	return math::orthographic(c.ortho_size, aspect, c.near_plane, c.far_plane);
}

float currentAspect() {
	const auto& extent = graphics::internal::context.swapchain_extent;
	return extent.height > 0 ? float(extent.width) / float(extent.height) : 1.0f;
}

void showMatrix(const char* label, const Mat4& m) {
	ImGui::TextUnformatted(label);
	for (int row = 0; row < 4; ++row) {
		ImGui::Text("%8.3f %8.3f %8.3f %8.3f", m.m[0][row], m.m[1][row], m.m[2][row], m.m[3][row]);
	}
}

void setupInitialState() {
	State& s = state;

	s.point_lights.clear();
	s.spot_lights.clear();

	PointLight warm;
	warm.position[0] = -4.5f; warm.position[1] = 1.5f; warm.position[2] = 3.0f;
	warm.color[0] = 1.0f; warm.color[1] = 0.65f; warm.color[2] = 0.35f;
	warm.intensity = 40.0f;
	s.point_lights.push_back(warm);

	PointLight cold;
	cold.position[0] = 4.5f; cold.position[1] = 0.5f; cold.position[2] = 3.0f;
	cold.color[0] = 0.35f; cold.color[1] = 0.55f; cold.color[2] = 1.0f;
	cold.intensity = 40.0f;
	s.point_lights.push_back(cold);

	s.spot_lights.push_back(SpotLight{});

	objects.clear();
	objects.push_back({ "Octahedron (animated)", MeshOctahedron,
	                    { 0.0f, 0.0f, 0.0f }, { 20.0f, 30.0f, 0.0f }, { 1.2f, 1.2f, 1.2f },
	                    { { 0.85f, 0.30f, 0.25f }, { 1.0f, 1.0f, 1.0f }, 64.0f } });
	objects.push_back({ "Floor", MeshPlane,
	                    { 0.0f, -2.5f, 0.0f }, { 0.0f, 0.0f, 0.0f }, { 10.0f, 1.0f, 10.0f },
	                    { { 0.55f, 0.55f, 0.60f }, { 0.25f, 0.25f, 0.25f }, 16.0f } });
	objects.push_back({ "Octahedron A", MeshOctahedron,
	                    { -4.0f, -1.7f, -1.0f }, { 0.0f, 20.0f, 0.0f }, { 0.8f, 0.8f, 0.8f },
	                    { { 0.25f, 0.70f, 0.35f }, { 0.6f, 0.6f, 0.6f }, 32.0f } });
	objects.push_back({ "Octahedron B", MeshOctahedron,
	                    { 4.0f, -1.2f, -2.0f }, { 0.0f, -35.0f, 0.0f }, { 1.3f, 1.3f, 1.3f },
	                    { { 0.30f, 0.45f, 0.85f }, { 1.0f, 1.0f, 1.0f }, 128.0f } });
}

bool createShaderModule(const std::string& path, VkShaderModule& module) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader file: " << path
		          << "\n(compile shaders with glslc first, see CMakeLists.txt)\n";
		return false;
	}

	const size_t size = size_t(file.tellg());
	if (size == 0 || size % 4 != 0) {
		std::cerr << "Invalid SPIR-V file: " << path << '\n';
		return false;
	}

	std::vector<uint32_t> code(size / 4);
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), std::streamsize(size));

	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = code.data(),
	};

	if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module: " << path << '\n';
		return false;
	}

	return true;
}

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, const void* data, Buffer& out) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo result{};
	auto& allocator = graphics::internal::context.allocator;

	Buffer created;
	if (vmaCreateBuffer(allocator, &buffer_info, &allocation_info,
	                    &created.buffer, &created.allocation, &result) != VK_SUCCESS ||
	    result.pMappedData == nullptr) {
		std::cerr << "Failed to create Vulkan buffer\n";
		if (created.buffer != VK_NULL_HANDLE) {
			vmaDestroyBuffer(allocator, created.buffer, created.allocation);
		}
		return false;
	}

	created.mapped = result.pMappedData;
	created.size = size;

	if (data != nullptr) {
		std::memcpy(created.mapped, data, size_t(size));
	} else {
		std::memset(created.mapped, 0, size_t(size));
	}
	vmaFlushAllocation(allocator, created.allocation, 0, VK_WHOLE_SIZE);

	out = created;
	return true;
}

void writeBuffer(Buffer& buffer, const void* data, VkDeviceSize size) {
	if (buffer.mapped == nullptr || size == 0 || size > buffer.size) {
		return;
	}
	std::memcpy(buffer.mapped, data, size_t(size));
	vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, 0, size);
}

void destroyBuffer(Buffer& buffer) {
	if (buffer.buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(graphics::internal::context.allocator, buffer.buffer, buffer.allocation);
	}
	buffer = {};
}

void writeDescriptor(VkDescriptorSet set, uint32_t binding, VkDescriptorType type, const Buffer& buffer) {
	const VkDescriptorBufferInfo buffer_info = {
		.buffer = buffer.buffer,
		.offset = 0,
		.range = VK_WHOLE_SIZE,
	};

	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = binding,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType = type,
		.pBufferInfo = &buffer_info,
	};

	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

bool growStorageBuffer(Buffer& buffer, uint32_t binding, VkDeviceSize required) {
	if (buffer.size >= required) {
		return true;
	}

	Buffer grown;
	const VkDeviceSize new_size = std::max<VkDeviceSize>(buffer.size * 2, required);
	if (!createBuffer(new_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, nullptr, grown)) {
		return false;
	}

	destroyBuffer(buffer);
	buffer = grown;
	writeDescriptor(scene_set, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffer);
	return true;
}

bool createDescriptors() {
	auto& context = graphics::internal::context;

	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(scene_bindings) / sizeof(scene_bindings[0]),
		.pBindings = scene_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr,
	                                &scene_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create scene descriptor set layout\n";
		return false;
	}

	const VkDescriptorSetLayoutBinding model_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo model_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr,
	                                &model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create model descriptor set layout\n";
		return false;
	}

	const uint32_t object_count = uint32_t(objects.size());

	const VkDescriptorPoolSize pool_sizes[] = {
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 + object_count },
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + object_count,
		.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo scene_alloc = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &scene_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &scene_alloc, &scene_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate scene descriptor set\n";
		return false;
	}

	writeDescriptor(scene_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, scene_buffer);
	writeDescriptor(scene_set, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, point_buffer);
	writeDescriptor(scene_set, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, spot_buffer);

	for (Object& o : objects) {
		const VkDescriptorSetAllocateInfo model_alloc = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &model_set_layout,
		};

		if (vkAllocateDescriptorSets(context.device, &model_alloc, &o.set) != VK_SUCCESS) {
			std::cerr << "Failed to allocate model descriptor set\n";
			return false;
		}

		writeDescriptor(o.set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, o.uniform);
	}

	return true;
}

bool createPipeline() {
	auto& context = graphics::internal::context;

	if (!createShaderModule(std::string(SHADER_DIR) + "octahedron.vert.spv", shader_vert) ||
	    !createShaderModule(std::string(SHADER_DIR) + "octahedron.frag.spv", shader_frag)) {
		return false;
	}

	const VkDescriptorSetLayout set_layouts[] = { scene_set_layout, model_set_layout };

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = sizeof(set_layouts) / sizeof(set_layouts[0]),
		.pSetLayouts = set_layouts,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = shader_vert,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = shader_frag,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = uint32_t(offsetof(Vertex, position)) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = uint32_t(offsetof(Vertex, normal)) },
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = 2,
		.pVertexAttributeDescriptions = attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE,
	};

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	const VkPipelineColorBlendAttachmentState blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};

	const VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info,
	                              nullptr, &pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}

	return true;
}

void destroyResources() {
	auto& context = graphics::internal::context;

	if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(context.device, pipeline, nullptr);
	if (pipeline_layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	if (descriptor_pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	if (scene_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);
	if (model_set_layout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(context.device, model_set_layout, nullptr);
	if (shader_vert != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, shader_vert, nullptr);
	if (shader_frag != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, shader_frag, nullptr);

	for (Object& o : objects) {
		destroyBuffer(o.uniform);
		o.set = VK_NULL_HANDLE;
	}

	destroyBuffer(scene_buffer);
	destroyBuffer(point_buffer);
	destroyBuffer(spot_buffer);
	destroyBuffer(vertex_buffer);
	destroyBuffer(index_buffer);

	pipeline = VK_NULL_HANDLE;
	pipeline_layout = VK_NULL_HANDLE;
	descriptor_pool = VK_NULL_HANDLE;
	scene_set = VK_NULL_HANDLE;
	scene_set_layout = VK_NULL_HANDLE;
	model_set_layout = VK_NULL_HANDLE;
	shader_vert = shader_frag = VK_NULL_HANDLE;
}

void updateCamera(Camera& c, float dt) {
	ImGuiIO& io = ImGui::GetIO();

	if (!io.WantCaptureMouse && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
		c.yaw_deg -= io.MouseDelta.x * c.mouse_sensitivity;
		c.pitch_deg = std::clamp(c.pitch_deg - io.MouseDelta.y * c.mouse_sensitivity, -89.0f, 89.0f);

		if (c.yaw_deg > 180.0f) c.yaw_deg -= 360.0f;
		if (c.yaw_deg < -180.0f) c.yaw_deg += 360.0f;
	}

	if (io.WantTextInput) {
		return;
	}

	const Vec3 forward = directionFromAngles(c.yaw_deg, c.pitch_deg);
	const float yaw = math::radians(c.yaw_deg);
	const Vec3 right = { std::cos(yaw), 0.0f, -std::sin(yaw) };
	const Vec3 up = { 0.0f, 1.0f, 0.0f };

	Vec3 move = {};
	if (ImGui::IsKeyDown(ImGuiKey_W)) move = move + forward;
	if (ImGui::IsKeyDown(ImGuiKey_S)) move = move - forward;
	if (ImGui::IsKeyDown(ImGuiKey_D)) move = move + right;
	if (ImGui::IsKeyDown(ImGuiKey_A)) move = move - right;
	if (ImGui::IsKeyDown(ImGuiKey_E)) move = move + up;
	if (ImGui::IsKeyDown(ImGuiKey_Q)) move = move - up;

	if (math::length(move) > 0.0f) {
		const float boost = ImGui::IsKeyDown(ImGuiKey_LeftShift) ? 3.0f : 1.0f;
		const Vec3 delta = math::normalize(move) * (c.move_speed * boost * dt);
		c.position[0] += delta.x;
		c.position[1] += delta.y;
		c.position[2] += delta.z;
	}
}

void drawCameraUi(Camera& c) {
	ImGui::RadioButton("Perspective", &c.projection, Perspective);
	ImGui::SameLine();
	ImGui::RadioButton("Orthographic", &c.projection, Orthographic);

	ImGui::DragFloat3("Position", c.position, 0.05f);
	ImGui::SliderFloat("Yaw (deg)", &c.yaw_deg, -180.0f, 180.0f);
	ImGui::SliderFloat("Pitch (deg)", &c.pitch_deg, -89.0f, 89.0f);
	ImGui::SliderFloat("Roll (deg)", &c.roll_deg, -180.0f, 180.0f);

	if (c.projection == Perspective) {
		ImGui::SliderFloat("FOV (deg)", &c.fov_deg, 20.0f, 120.0f);
	} else {
		ImGui::SliderFloat("Ortho size (half height)", &c.ortho_size, 1.0f, 20.0f);
	}

	ImGui::SliderFloat("Near plane", &c.near_plane, 0.01f, 5.0f);
	ImGui::SliderFloat("Far plane", &c.far_plane, 10.0f, 300.0f);
	ImGui::SliderFloat("Move speed", &c.move_speed, 0.5f, 30.0f);
	ImGui::SliderFloat("Mouse sensitivity", &c.mouse_sensitivity, 0.05f, 1.0f);

	if (ImGui::Button("Reset camera")) {
		c = Camera{};
	}

	ImGui::TextDisabled("Hold RMB + move mouse: look around");
	ImGui::TextDisabled("W/A/S/D: move, Q/E: down/up, Shift: faster");
}

void drawLightingUi(State& s) {
	ImGui::ColorEdit3("Ambient color", s.ambient_color);
	ImGui::SliderFloat("Ambient intensity", &s.ambient_intensity, 0.0f, 1.0f);

	ImGui::Separator();
	ImGui::TextUnformatted("Directional light");
	DirectionalLight& d = s.directional;
	ImGui::Checkbox("Enabled##directional", &d.enabled);
	ImGui::SliderFloat("Yaw (deg)##directional", &d.yaw_deg, -180.0f, 180.0f);
	ImGui::SliderFloat("Pitch (deg)##directional", &d.pitch_deg, -89.0f, 89.0f);
	ImGui::ColorEdit3("Color##directional", d.color);
	ImGui::SliderFloat("Intensity##directional", &d.intensity, 0.0f, 5.0f);
}

void drawPointLightsUi(State& s) {
	if (ImGui::Button("Add point light")) {
		s.point_lights.push_back(PointLight{});
	}

	int remove_index = -1;
	for (int i = 0; i < int(s.point_lights.size()); ++i) {
		PointLight& l = s.point_lights[size_t(i)];

		char label[48];
		std::snprintf(label, sizeof(label), "Point light %d", i + 1);

		ImGui::PushID(i);
		if (ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::DragFloat3("Position", l.position, 0.05f);
			ImGui::ColorEdit3("Color", l.color);
			ImGui::DragFloat("Intensity", &l.intensity, 0.5f, 0.0f, 100000.0f);

			if (ImGui::Button("Remove")) {
				remove_index = i;
			}
			ImGui::TreePop();
		}
		ImGui::PopID();
	}

	if (remove_index >= 0) {
		s.point_lights.erase(s.point_lights.begin() + remove_index);
	}
}

void drawSpotLightsUi(State& s) {
	if (ImGui::Button("Add spot light")) {
		s.spot_lights.push_back(SpotLight{});
	}

	int remove_index = -1;
	for (int i = 0; i < int(s.spot_lights.size()); ++i) {
		SpotLight& l = s.spot_lights[size_t(i)];

		char label[48];
		std::snprintf(label, sizeof(label), "Spot light %d", i + 1);

		ImGui::PushID(i);
		if (ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::DragFloat3("Position", l.position, 0.05f);
			ImGui::DragFloat3("Target", l.target, 0.05f);
			ImGui::ColorEdit3("Color", l.color);
			ImGui::DragFloat("Intensity", &l.intensity, 0.5f, 0.0f, 100000.0f);

			const bool inner_changed = ImGui::SliderFloat("Inner angle (deg)", &l.inner_deg, 0.0f, 89.0f);
			const bool outer_changed = ImGui::SliderFloat("Outer angle (deg)", &l.outer_deg, 0.0f, 90.0f);
			if (inner_changed && l.inner_deg > l.outer_deg) {
				l.outer_deg = l.inner_deg;
			}
			if (outer_changed && l.outer_deg < l.inner_deg) {
				l.inner_deg = l.outer_deg;
			}

			if (ImGui::Button("Remove")) {
				remove_index = i;
			}
			ImGui::TreePop();
		}
		ImGui::PopID();
	}

	if (remove_index >= 0) {
		s.spot_lights.erase(s.spot_lights.begin() + remove_index);
	}
}

void drawObjectsUi(State& s) {
	if (ImGui::BeginCombo("Object", objects[size_t(s.selected_object)].name)) {
		for (int i = 0; i < int(objects.size()); ++i) {
			if (ImGui::Selectable(objects[size_t(i)].name, i == s.selected_object)) {
				s.selected_object = i;
			}
		}
		ImGui::EndCombo();
	}

	Object& o = objects[size_t(s.selected_object)];

	ImGui::DragFloat3("Position", o.position, 0.05f);
	ImGui::SliderFloat3("Rotation (deg)", o.rotation_deg, -180.0f, 180.0f);
	ImGui::SliderFloat3("Scale", o.scale, 0.1f, 15.0f);

	ImGui::Separator();
	ImGui::TextUnformatted("Material (Blinn-Phong)");
	ImGui::ColorEdit3("Albedo", o.material.albedo);
	ImGui::ColorEdit3("Specular color", o.material.specular);
	ImGui::SliderFloat("Shininess", &o.material.shininess, 1.0f, 512.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
}

void drawAnimationUi(Animation& a) {
	if (ImGui::Button(a.playing ? "Pause" : "Play")) {
		a.playing = !a.playing;
	}
	ImGui::SameLine();
	if (ImGui::Button("Restart")) {
		a.clock = 0.0f;
		a.spin_angle = 0.0f;
	}

	ImGui::SliderFloat("Speed", &a.speed, 0.0f, 5.0f);

	const char* trajectories[] = { "Lissajous curve", "Trefoil knot", "Breathing spiral" };
	ImGui::Combo("Trajectory", &a.trajectory, trajectories, 3);

	ImGui::SliderFloat("Radius", &a.radius, 0.5f, 8.0f);
	ImGui::SliderFloat("Height amplitude", &a.height, 0.0f, 4.0f);

	if (a.trajectory == Lissajous) {
		ImGui::SliderInt3("Frequencies XYZ", a.freq, 1, 7);
		ImGui::SliderFloat("Phase (deg)", &a.phase_deg, 0.0f, 360.0f);
	} else if (a.trajectory == Spiral) {
		ImGui::SliderInt("Vertical frequency", &a.freq[1], 1, 7);
	}

	ImGui::SliderFloat("Spin speed (deg/s)", &a.spin_speed_deg, -360.0f, 360.0f);
	ImGui::DragFloat3("Spin axis", a.spin_axis, 0.01f, -1.0f, 1.0f);
}

}

bool initialize() {
	setupInitialState();

	std::vector<Vertex> vertices;
	std::vector<uint16_t> indices;

	std::vector<Vertex> mesh_vertices;
	std::vector<uint16_t> mesh_indices;

	makeOctahedron(mesh_vertices, mesh_indices);
	meshes[MeshOctahedron] = { uint32_t(indices.size()), uint32_t(mesh_indices.size()), int32_t(vertices.size()) };
	vertices.insert(vertices.end(), mesh_vertices.begin(), mesh_vertices.end());
	indices.insert(indices.end(), mesh_indices.begin(), mesh_indices.end());

	makePlane(mesh_vertices, mesh_indices);
	meshes[MeshPlane] = { uint32_t(indices.size()), uint32_t(mesh_indices.size()), int32_t(vertices.size()) };
	vertices.insert(vertices.end(), mesh_vertices.begin(), mesh_vertices.end());
	indices.insert(indices.end(), mesh_indices.begin(), mesh_indices.end());

	bool ok = createBuffer(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                       vertices.data(), vertex_buffer) &&
	          createBuffer(indices.size() * sizeof(uint16_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
	                       indices.data(), index_buffer) &&
	          createBuffer(sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, nullptr, scene_buffer) &&
	          createBuffer(initial_light_capacity * sizeof(GpuPointLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                       nullptr, point_buffer) &&
	          createBuffer(initial_light_capacity * sizeof(GpuSpotLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                       nullptr, spot_buffer);

	for (size_t i = 0; ok && i < objects.size(); ++i) {
		ok = createBuffer(sizeof(ModelData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, nullptr, objects[i].uniform);
	}

	if (!ok || !createDescriptors() || !createPipeline()) {
		destroyResources();
		return false;
	}

	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);
	destroyResources();
}

void update(double time) {
	State& s = state;

	const float dt = s.last_time < 0.0 ? 0.0f : float(std::fmin(time - s.last_time, 0.1));
	s.last_time = time;

	updateCamera(s.camera, dt);

	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(460.0f, 760.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Camera & Lighting - Lab 2");

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		drawCameraUi(s.camera);
	}

	if (ImGui::CollapsingHeader("Ambient & directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
		drawLightingUi(s);
	}

	char header[64];

	std::snprintf(header, sizeof(header), "Point lights (%d)###point_lights", int(s.point_lights.size()));
	if (ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) {
		drawPointLightsUi(s);
	}

	std::snprintf(header, sizeof(header), "Spot lights (%d)###spot_lights", int(s.spot_lights.size()));
	if (ImGui::CollapsingHeader(header, ImGuiTreeNodeFlags_DefaultOpen)) {
		drawSpotLightsUi(s);
	}

	if (ImGui::CollapsingHeader("Objects & materials", ImGuiTreeNodeFlags_DefaultOpen)) {
		drawObjectsUi(s);
	}

	if (ImGui::CollapsingHeader("Animation (main octahedron)")) {
		drawAnimationUi(s.animation);
	}

	if (s.animation.playing) {
		s.animation.clock += dt * s.animation.speed;
		s.animation.spin_angle += math::radians(s.animation.spin_speed_deg) * dt * s.animation.speed;
	}

	if (ImGui::CollapsingHeader("Matrices")) {
		showMatrix("Model (selected object):", modelMatrix(s.selected_object));
		showMatrix("View:", buildView(s.camera));
		showMatrix("Projection:", buildProjection(s.camera, currentAspect()));
	}

	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	auto& context = graphics::internal::context;
	const State& s = state;
	VkCommandBuffer cb = fd.command_buffer;

	const float width = float(context.swapchain_extent.width);
	const float height = float(context.swapchain_extent.height);

	std::vector<GpuPointLight> point_data;
	point_data.reserve(s.point_lights.size());
	for (const PointLight& l : s.point_lights) {
		point_data.push_back({
			{ l.position[0], l.position[1], l.position[2], 1.0f },
			{ l.color[0], l.color[1], l.color[2], l.intensity },
		});
	}

	std::vector<GpuSpotLight> spot_data;
	spot_data.reserve(s.spot_lights.size());
	for (const SpotLight& l : s.spot_lights) {
		Vec3 direction = math::normalize(toVec3(l.target) - toVec3(l.position));
		if (math::length(direction) < 0.5f) {
			direction = { 0.0f, -1.0f, 0.0f };
		}

		const float inner = std::min(l.inner_deg, l.outer_deg);
		const float outer = std::max(l.inner_deg, l.outer_deg);

		spot_data.push_back({
			{ l.position[0], l.position[1], l.position[2], 1.0f },
			{ direction.x, direction.y, direction.z, 0.0f },
			{ l.color[0], l.color[1], l.color[2], l.intensity },
			{ std::cos(math::radians(inner)), std::cos(math::radians(outer)), 0.0f, 0.0f },
		});
	}

	growStorageBuffer(point_buffer, 1, point_data.size() * sizeof(GpuPointLight));
	growStorageBuffer(spot_buffer, 2, spot_data.size() * sizeof(GpuSpotLight));

	const size_t point_capacity = size_t(point_buffer.size / sizeof(GpuPointLight));
	const size_t spot_capacity = size_t(spot_buffer.size / sizeof(GpuSpotLight));
	const size_t point_count = std::min(point_data.size(), point_capacity);
	const size_t spot_count = std::min(spot_data.size(), spot_capacity);

	writeBuffer(point_buffer, point_data.data(), point_count * sizeof(GpuPointLight));
	writeBuffer(spot_buffer, spot_data.data(), spot_count * sizeof(GpuSpotLight));

	const Mat4 view = buildView(s.camera);
	const Mat4 projection = buildProjection(s.camera, width / height);
	const Mat4 view_proj = projection * view;

	const Vec3 dir = directionFromAngles(s.directional.yaw_deg, s.directional.pitch_deg);
	const float dir_scale = s.directional.enabled ? s.directional.intensity : 0.0f;

	SceneData scene{};
	std::memcpy(scene.view_proj, view_proj.m, sizeof(scene.view_proj));
	scene.camera_position[0] = s.camera.position[0];
	scene.camera_position[1] = s.camera.position[1];
	scene.camera_position[2] = s.camera.position[2];
	scene.camera_position[3] = 1.0f;
	scene.ambient[0] = s.ambient_color[0] * s.ambient_intensity;
	scene.ambient[1] = s.ambient_color[1] * s.ambient_intensity;
	scene.ambient[2] = s.ambient_color[2] * s.ambient_intensity;
	scene.dir_direction[0] = dir.x;
	scene.dir_direction[1] = dir.y;
	scene.dir_direction[2] = dir.z;
	scene.dir_color[0] = s.directional.color[0] * dir_scale;
	scene.dir_color[1] = s.directional.color[1] * dir_scale;
	scene.dir_color[2] = s.directional.color[2] * dir_scale;
	scene.counts[0] = uint32_t(point_count);
	scene.counts[1] = uint32_t(spot_count);
	writeBuffer(scene_buffer, &scene, sizeof(scene));

	for (size_t i = 0; i < objects.size(); ++i) {
		const Object& o = objects[i];
		const Mat4 model = modelMatrix(int(i));
		const Mat4 normal = math::normalMatrix(model);

		ModelData data{};
		std::memcpy(data.model, model.m, sizeof(data.model));
		std::memcpy(data.normal_matrix, normal.m, sizeof(data.normal_matrix));
		data.albedo[0] = o.material.albedo[0];
		data.albedo[1] = o.material.albedo[1];
		data.albedo[2] = o.material.albedo[2];
		data.albedo[3] = 1.0f;
		data.specular[0] = o.material.specular[0];
		data.specular[1] = o.material.specular[1];
		data.specular[2] = o.material.specular[2];
		data.specular[3] = o.material.shininess;
		writeBuffer(objects[i].uniform, &data, sizeof(data));
	}

	vkResetCommandBuffer(cb, 0);

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cb, &begin);

	VkClearValue clear_values[2];
	clear_values[0].color = { { 0.02f, 0.025f, 0.04f, 1.0f } };
	clear_values[1].depthStencil = { 1.0f, 0 };

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .offset = { 0, 0 }, .extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(cb, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	const VkViewport viewport = { 0.0f, 0.0f, width, height, 0.0f, 1.0f };
	const VkRect2D scissor = { { 0, 0 }, context.swapchain_extent };
	vkCmdSetViewport(cb, 0, 1, &viewport);
	vkCmdSetScissor(cb, 0, 1, &scissor);

	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cb, 0, 1, &vertex_buffer.buffer, &offset);
	vkCmdBindIndexBuffer(cb, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT16);

	vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
	                        0, 1, &scene_set, 0, nullptr);

	for (const Object& o : objects) {
		const Mesh& mesh = meshes[o.mesh];

		vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        1, 1, &o.set, 0, nullptr);
		vkCmdDrawIndexed(cb, mesh.index_count, 1, mesh.first_index, mesh.vertex_offset, 0);
	}

	vkCmdEndRenderPass(cb);
	vkEndCommandBuffer(cb);
}

}
