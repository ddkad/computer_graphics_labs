#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(push_constant) uniform PushConstants {
	mat4 mvp;   // proj * view * model
	vec4 tint;  // цвет из ColorEdit (rgb), a не используется
} pc;

layout(location = 0) out vec3 out_color;

void main() {
	gl_Position = pc.mvp * vec4(in_position, 1.0);
	// цвет вершины (по позиции в локальных координатах) * выбранный цвет
	out_color = in_color * pc.tint.rgb;
}
