#include "application.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <imgui.h>

#include "math3d.hpp"
#include "octahedron.hpp"

// Путь к каталогу со скомпилированными шейдерами (*.spv) задаётся в CMakeLists.txt.
#ifndef SHADER_DIR
#define SHADER_DIR "shaders/"
#endif

namespace application {

namespace {

using math::Mat4;
using math::Vec3;

// ------------------------------------------------------------------ данные GPU

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
};

// Push constants: 64 байта (mat4) + 16 байт (vec4) = 80 байт (минимум по стандарту — 128).
struct PushConstants {
	float mvp[16];
	float tint[4];
};
static_assert(sizeof(PushConstants) == 80, "Unexpected push constants layout");

Buffer vertex_buffer;
Buffer index_buffer;
uint32_t index_count = 0;

VkShaderModule shader_vert = VK_NULL_HANDLE;
VkShaderModule shader_frag = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

// -------------------------------------------------------- состояние интерфейса

enum Projection : int { Perspective = 0, Orthographic = 1 };
enum Trajectory : int { Lissajous = 0, Trefoil = 1, Spiral = 2 };

struct State {
	// Проекция и камера
	int projection = Perspective;
	float fov_deg = 60.0f;
	float ortho_size = 6.0f; // половина высоты видимой области
	float near_plane = 0.1f;
	float far_plane = 100.0f;
	float camera_distance = 12.0f;

	// Ручные трансформации
	float position[3] = { 0.0f, 0.0f, 0.0f };
	float rotation_deg[3] = { 20.0f, 30.0f, 0.0f };
	float scale[3] = { 1.5f, 1.5f, 1.5f };
	bool uniform_scale = true;

	// Анимация
	bool playing = true;
	float speed = 1.0f;
	int trajectory = Lissajous;
	float radius = 3.0f;
	float height = 1.5f;
	int freq[3] = { 3, 2, 1 };
	float phase_deg = 90.0f;
	float spin_speed_deg = 60.0f;
	float spin_axis[3] = { 0.4f, 1.0f, 0.2f };

	double last_time = -1.0;
	float anim_clock = 0.0f;
	float spin_angle = 0.0f; // радианы

	// Цвет фигуры (умножается на цвета вершин)
	float color[3] = { 1.0f, 1.0f, 1.0f };

	// Результат update(), используется в render()
	Mat4 model = math::identity();
};

State state;

// ------------------------------------------------------------ вспомогательное

Vec3 toVec3(const float v[3]) {
	return { v[0], v[1], v[2] };
}

// Смещение фигуры по траектории в момент параметра t.
Vec3 trajectoryOffset(const State& s, float t) {
	switch (s.trajectory) {
	case Lissajous: {
		// Фигура Лиссажу в 3D: независимые частоты по каждой оси.
		const float phase = math::radians(s.phase_deg);
		return {
			s.radius * std::sin(float(s.freq[0]) * t + phase),
			s.height * std::sin(float(s.freq[1]) * t),
			s.radius * std::sin(float(s.freq[2]) * t),
		};
	}
	case Trefoil: {
		// Узел-трилистник (trefoil knot).
		const float k = s.radius / 3.0f;
		return {
			k * (std::sin(t) + 2.0f * std::sin(2.0f * t)),
			-s.height * std::sin(3.0f * t),
			k * (std::cos(t) - 2.0f * std::cos(2.0f * t)),
		};
	}
	case Spiral:
	default: {
		// Спираль с «дышащим» радиусом и колебаниями по высоте.
		const float r = s.radius * (0.65f + 0.35f * std::sin(0.3f * t));
		return {
			r * std::cos(t),
			s.height * std::sin(float(s.freq[1]) * 0.5f * t),
			r * std::sin(t),
		};
	}
	}
}

// Модельная матрица: M = T * Rz * Ry * Rx * Rspin * S
// (к вершине сначала применяется масштаб, затем повороты, затем перенос).
Mat4 buildModel(const State& s, Vec3 trajectory_offset) {
	const Mat4 T = math::translate(toVec3(s.position) + trajectory_offset);
	const Mat4 R = math::rotateZ(math::radians(s.rotation_deg[2])) *
	               math::rotateY(math::radians(s.rotation_deg[1])) *
	               math::rotateX(math::radians(s.rotation_deg[0]));
	const Mat4 Rspin = math::rotateAxis(toVec3(s.spin_axis), s.spin_angle);
	const Mat4 S = math::scale(toVec3(s.scale));
	return T * R * Rspin * S;
}

Mat4 buildView(const State& s) {
	// Камера стоит в точке (0, 0, distance) и смотрит в начало координат.
	return math::translate({ 0.0f, 0.0f, -s.camera_distance });
}

Mat4 buildProjection(const State& s, float aspect) {
	if (s.projection == Perspective) {
		return math::perspective(math::radians(s.fov_deg), aspect, s.near_plane, s.far_plane);
	}
	return math::orthographic(s.ortho_size, aspect, s.near_plane, s.far_plane);
}

void showMatrix(const char* label, const Mat4& m) {
	ImGui::TextUnformatted(label);
	for (int row = 0; row < 4; ++row) {
		ImGui::Text("%8.3f %8.3f %8.3f %8.3f", m.m[0][row], m.m[1][row], m.m[2][row], m.m[3][row]);
	}
}

// ---------------------------------------------------------------- Vulkan-часть

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

// Буфер в памяти, видимой CPU: для крошечной статической геометрии этого достаточно
// (для больших моделей данные копируют через staging-буфер в device-local память).
bool createBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
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

	if (vmaCreateBuffer(allocator, &buffer_info, &allocation_info,
	                    &out.buffer, &out.allocation, &result) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan buffer\n";
		return false;
	}

	std::memcpy(result.pMappedData, data, size_t(size));
	vmaFlushAllocation(allocator, out.allocation, 0, VK_WHOLE_SIZE);
	return true;
}

bool createPipeline() {
	auto& context = graphics::internal::context;

	if (!createShaderModule(std::string(SHADER_DIR) + "octahedron.vert.spv", shader_vert) ||
	    !createShaderModule(std::string(SHADER_DIR) + "octahedron.frag.spv", shader_frag)) {
		return false;
	}

	const VkPushConstantRange push_range = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.offset = 0,
		.size = sizeof(PushConstants),
	};

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_range,
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
		  .offset = uint32_t(offsetof(Vertex, color)) },
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

	// Viewport и scissor задаются динамически (при изменении размера окна
	// не нужно пересоздавать pipeline).
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
	if (shader_vert != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, shader_vert, nullptr);
	if (shader_frag != VK_NULL_HANDLE) vkDestroyShaderModule(context.device, shader_frag, nullptr);
	if (vertex_buffer.buffer != VK_NULL_HANDLE)
		vmaDestroyBuffer(context.allocator, vertex_buffer.buffer, vertex_buffer.allocation);
	if (index_buffer.buffer != VK_NULL_HANDLE)
		vmaDestroyBuffer(context.allocator, index_buffer.buffer, index_buffer.allocation);

	pipeline = VK_NULL_HANDLE;
	pipeline_layout = VK_NULL_HANDLE;
	shader_vert = shader_frag = VK_NULL_HANDLE;
	vertex_buffer = {};
	index_buffer = {};
}

} // namespace

// ------------------------------------------------------------ публичный интерфейс

bool initialize() {
	std::vector<Vertex> vertices;
	std::vector<uint16_t> indices;
	makeOctahedron(vertices, indices);
	index_count = uint32_t(indices.size());

	if (!createBuffer(vertices.data(), vertices.size() * sizeof(Vertex),
	                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer) ||
	    !createBuffer(indices.data(), indices.size() * sizeof(uint16_t),
	                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer) ||
	    !createPipeline()) {
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

	// --- время анимации (накапливаем dt, поэтому пауза и смена скорости не дают скачков)
	const float dt = s.last_time < 0.0 ? 0.0f : float(std::fmin(time - s.last_time, 0.1));
	s.last_time = time;

	// --- интерфейс
	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(400.0f, 640.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Octahedron - Lab 1");

	if (ImGui::CollapsingHeader("Projection & camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::RadioButton("Perspective", &s.projection, Perspective);
		ImGui::SameLine();
		ImGui::RadioButton("Orthographic", &s.projection, Orthographic);

		if (s.projection == Perspective) {
			ImGui::SliderFloat("FOV (deg)", &s.fov_deg, 20.0f, 120.0f);
		} else {
			ImGui::SliderFloat("Ortho size (half height)", &s.ortho_size, 1.0f, 20.0f);
		}
		ImGui::SliderFloat("Near plane", &s.near_plane, 0.01f, 5.0f);
		ImGui::SliderFloat("Far plane", &s.far_plane, 10.0f, 200.0f);
		ImGui::SliderFloat("Camera distance", &s.camera_distance, 2.0f, 40.0f);
	}

	if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::SliderFloat3("Position", s.position, -6.0f, 6.0f);
		ImGui::SliderFloat3("Rotation (deg)", s.rotation_deg, -180.0f, 180.0f);

		ImGui::Checkbox("Uniform scale", &s.uniform_scale);
		if (s.uniform_scale) {
			if (ImGui::SliderFloat("Scale", &s.scale[0], 0.1f, 5.0f)) {
				s.scale[1] = s.scale[2] = s.scale[0];
			}
		} else {
			ImGui::SliderFloat3("Scale XYZ", s.scale, 0.1f, 5.0f);
		}

		if (ImGui::Button("Reset transform")) {
			const State defaults;
			std::memcpy(s.position, defaults.position, sizeof(s.position));
			std::memcpy(s.rotation_deg, defaults.rotation_deg, sizeof(s.rotation_deg));
			std::memcpy(s.scale, defaults.scale, sizeof(s.scale));
		}
	}

	if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::Button(s.playing ? "Pause" : "Play")) {
			s.playing = !s.playing;
		}
		ImGui::SameLine();
		if (ImGui::Button("Restart")) {
			s.anim_clock = 0.0f;
			s.spin_angle = 0.0f;
		}

		ImGui::SliderFloat("Speed", &s.speed, 0.0f, 5.0f);

		const char* trajectories[] = { "Lissajous curve", "Trefoil knot", "Breathing spiral" };
		ImGui::Combo("Trajectory", &s.trajectory, trajectories, 3);

		ImGui::SliderFloat("Radius", &s.radius, 0.5f, 8.0f);
		ImGui::SliderFloat("Height amplitude", &s.height, 0.0f, 4.0f);

		if (s.trajectory == Lissajous) {
			ImGui::SliderInt3("Frequencies XYZ", s.freq, 1, 7);
			ImGui::SliderFloat("Phase (deg)", &s.phase_deg, 0.0f, 360.0f);
		} else if (s.trajectory == Spiral) {
			ImGui::SliderInt("Vertical frequency", &s.freq[1], 1, 7);
		}

		ImGui::SliderFloat("Spin speed (deg/s)", &s.spin_speed_deg, -360.0f, 360.0f);
		ImGui::DragFloat3("Spin axis", s.spin_axis, 0.01f, -1.0f, 1.0f);
	}

	if (ImGui::CollapsingHeader("Color", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Tint (x vertex color)", s.color);
	}

	// --- продвижение анимации
	if (s.playing) {
		s.anim_clock += dt * s.speed;
		s.spin_angle += math::radians(s.spin_speed_deg) * dt * s.speed;
	}

	// --- итоговая модельная матрица
	s.model = buildModel(s, trajectoryOffset(s, s.anim_clock));

	if (ImGui::CollapsingHeader("Matrices")) {
		const auto& extent = graphics::internal::context.swapchain_extent;
		const float aspect = float(extent.width) / float(extent.height);
		showMatrix("Model:", s.model);
		showMatrix("View:", buildView(s));
		showMatrix("Projection:", buildProjection(s, aspect));
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

	// Размер swapchain мог измениться внутри prepare(), поэтому aspect считаем здесь.
	const float width = float(context.swapchain_extent.width);
	const float height = float(context.swapchain_extent.height);

	const Mat4 mvp = buildProjection(s, width / height) * buildView(s) * s.model;

	PushConstants push{};
	std::memcpy(push.mvp, mvp.m, sizeof(push.mvp));
	push.tint[0] = s.color[0];
	push.tint[1] = s.color[1];
	push.tint[2] = s.color[2];
	push.tint[3] = 1.0f;

	vkResetCommandBuffer(cb, 0);

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cb, &begin);

	VkClearValue clear_values[2];
	clear_values[0].color = { { 0.05f, 0.06f, 0.09f, 1.0f } };
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

	vkCmdPushConstants(cb, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);

	vkCmdDrawIndexed(cb, index_count, 1, 0, 0, 0);

	vkCmdEndRenderPass(cb);
	vkEndCommandBuffer(cb);
}

} // namespace application
