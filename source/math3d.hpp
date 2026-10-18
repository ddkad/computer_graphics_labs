#pragma once

// Минимальная математическая библиотека для лабораторной работы №1.
//
// Соглашения:
//  * Mat4 хранится по СТОЛБЦАМ: m[столбец][строка]. Так же устроен mat4 в GLSL,
//    поэтому матрицу можно копировать в push constants / uniform «как есть».
//  * Векторы — столбцы, преобразование записывается как  v' = M * v.
//  * Система координат правая (right-handed), камера в пространстве вида
//    смотрит вдоль оси -Z, ось +Y направлена вверх.
//  * Матрицы проекции рассчитаны на Vulkan: диапазон глубины NDC z ∈ [0, 1],
//    ось Y в NDC направлена ВНИЗ (поэтому в проекции стоит знак «минус» по Y).

#include <cmath>

namespace math {

constexpr float pi = 3.14159265358979323846f;

constexpr float radians(float degrees) {
	return degrees * pi / 180.0f;
}

struct Vec3 {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline Vec3 operator*(Vec3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }

inline float length(Vec3 v) {
	return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

struct Vec4 {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float w = 0.0f;
};

struct Mat4 {
	float m[4][4]; // m[столбец][строка]
};

inline Mat4 identity() {
	Mat4 r{};
	r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.0f;
	return r;
}

// C = A * B  (сначала применяется B, затем A)
inline Mat4 operator*(const Mat4& a, const Mat4& b) {
	Mat4 r{};
	for (int c = 0; c < 4; ++c) {
		for (int row = 0; row < 4; ++row) {
			float sum = 0.0f;
			for (int k = 0; k < 4; ++k) {
				sum += a.m[k][row] * b.m[c][k];
			}
			r.m[c][row] = sum;
		}
	}
	return r;
}

inline Vec4 operator*(const Mat4& a, Vec4 v) {
	return {
		a.m[0][0] * v.x + a.m[1][0] * v.y + a.m[2][0] * v.z + a.m[3][0] * v.w,
		a.m[0][1] * v.x + a.m[1][1] * v.y + a.m[2][1] * v.z + a.m[3][1] * v.w,
		a.m[0][2] * v.x + a.m[1][2] * v.y + a.m[2][2] * v.z + a.m[3][2] * v.w,
		a.m[0][3] * v.x + a.m[1][3] * v.y + a.m[2][3] * v.z + a.m[3][3] * v.w,
	};
}

// ---------------------------------------------------------------- аффинные

inline Mat4 translate(Vec3 t) {
	Mat4 r = identity();
	r.m[3][0] = t.x;
	r.m[3][1] = t.y;
	r.m[3][2] = t.z;
	return r;
}

inline Mat4 scale(Vec3 s) {
	Mat4 r = identity();
	r.m[0][0] = s.x;
	r.m[1][1] = s.y;
	r.m[2][2] = s.z;
	return r;
}

inline Mat4 rotateX(float a) {
	const float c = std::cos(a), s = std::sin(a);
	Mat4 r = identity();
	r.m[1][1] = c;  r.m[2][1] = -s;
	r.m[1][2] = s;  r.m[2][2] = c;
	return r;
}

inline Mat4 rotateY(float a) {
	const float c = std::cos(a), s = std::sin(a);
	Mat4 r = identity();
	r.m[0][0] = c;  r.m[2][0] = s;
	r.m[0][2] = -s; r.m[2][2] = c;
	return r;
}

inline Mat4 rotateZ(float a) {
	const float c = std::cos(a), s = std::sin(a);
	Mat4 r = identity();
	r.m[0][0] = c;  r.m[1][0] = -s;
	r.m[0][1] = s;  r.m[1][1] = c;
	return r;
}

// Поворот на угол a вокруг произвольной оси (формула Родрига).
inline Mat4 rotateAxis(Vec3 axis, float a) {
	const float len = length(axis);
	if (len < 1e-6f) {
		return identity();
	}
	const float x = axis.x / len, y = axis.y / len, z = axis.z / len;
	const float c = std::cos(a), s = std::sin(a), t = 1.0f - c;

	Mat4 r = identity();
	r.m[0][0] = t * x * x + c;      r.m[1][0] = t * x * y - s * z;  r.m[2][0] = t * x * z + s * y;
	r.m[0][1] = t * x * y + s * z;  r.m[1][1] = t * y * y + c;      r.m[2][1] = t * y * z - s * x;
	r.m[0][2] = t * x * z - s * y;  r.m[1][2] = t * y * z + s * x;  r.m[2][2] = t * z * z + c;
	return r;
}

// -------------------------------------------------------------------- проекции

// Перспективная проекция (RH, глубина [0,1], Y вниз).
// fov_y — вертикальный угол обзора в радианах, aspect = ширина / высота.
inline Mat4 perspective(float fov_y, float aspect, float near_plane, float far_plane) {
	const float f = 1.0f / std::tan(fov_y * 0.5f);
	Mat4 r{};
	r.m[0][0] = f / aspect;
	r.m[1][1] = -f;                                              // переворот Y для Vulkan
	r.m[2][2] = far_plane / (near_plane - far_plane);
	r.m[2][3] = -1.0f;                                           // w_clip = -z_view
	r.m[3][2] = (near_plane * far_plane) / (near_plane - far_plane);
	return r;
}

// Ортографическая проекция (RH, глубина [0,1], Y вниз), симметричная:
// видимая область по Y — [-half_height, +half_height], по X — с учётом aspect.
inline Mat4 orthographic(float half_height, float aspect, float near_plane, float far_plane) {
	const float half_width = half_height * aspect;
	Mat4 r{};
	r.m[0][0] = 1.0f / half_width;
	r.m[1][1] = -1.0f / half_height;                             // переворот Y для Vulkan
	r.m[2][2] = 1.0f / (near_plane - far_plane);
	r.m[3][2] = near_plane / (near_plane - far_plane);
	r.m[3][3] = 1.0f;
	return r;
}

} // namespace math
