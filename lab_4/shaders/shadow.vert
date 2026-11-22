#version 450

layout(location = 0) in vec3 in_position;

layout(set = 0, binding = 0, std140) uniform SceneData {
	mat4 view_proj;
	mat4 light_view_proj;
	vec4 camera_position;
	vec4 ambient;
	vec4 dir_direction;
	vec4 dir_color;
	vec4 shadow_params;
	uvec4 counts;
} scene;

layout(set = 1, binding = 0, std140) uniform ModelData {
	mat4 model;
	mat4 normal_matrix;
	vec4 albedo;
	vec4 specular;
} object;

void main() {
	gl_Position = scene.light_view_proj * object.model * vec4(in_position, 1.0);
}
