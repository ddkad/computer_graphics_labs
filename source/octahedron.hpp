#pragma once

// Геометрия правильного октаэдра (вариант лабораторной работы).
//
// Октаэдр вписан в единичную сферу: 6 вершин лежат на осях координат
// (±1,0,0), (0,±1,0), (0,0,±1). Граней 8 (треугольники), рёбер 12.
// Цвет каждой вершины назначается процедурно по её позиции в локальной
// системе координат: color = position * 0.5 + 0.5  (отрезок [-1,1] -> [0,1]).

#include <cstddef>
#include <cstdint>
#include <vector>

struct Vertex {
	float position[3];
	float color[3];
};

inline void makeOctahedron(std::vector<Vertex>& vertices, std::vector<uint16_t>& indices) {
	// Индексы вершин: 0:+X  1:-X  2:+Y  3:-Y  4:+Z  5:-Z
	const float positions[6][3] = {
		{ +1.0f, 0.0f, 0.0f }, { -1.0f, 0.0f, 0.0f },
		{ 0.0f, +1.0f, 0.0f }, { 0.0f, -1.0f, 0.0f },
		{ 0.0f, 0.0f, +1.0f }, { 0.0f, 0.0f, -1.0f },
	};

	vertices.clear();
	for (const auto& p : positions) {
		vertices.push_back({
			{ p[0], p[1], p[2] },
			{ p[0] * 0.5f + 0.5f, p[1] * 0.5f + 0.5f, p[2] * 0.5f + 0.5f },
		});
	}

	// Каждая грань — один «октант» (знаки sx, sy, sz): вершины ±X, ±Y, ±Z.
	// Чтобы обход вершин был против часовой стрелки при взгляде снаружи
	// (правая система координат), порядок зависит от чётности знаков:
	//   sx*sy*sz > 0  ->  (X, Y, Z)
	//   sx*sy*sz < 0  ->  (X, Z, Y)
	indices.clear();
	for (int sx = -1; sx <= 1; sx += 2) {
		for (int sy = -1; sy <= 1; sy += 2) {
			for (int sz = -1; sz <= 1; sz += 2) {
				const uint16_t vx = sx > 0 ? 0 : 1;
				const uint16_t vy = sy > 0 ? 2 : 3;
				const uint16_t vz = sz > 0 ? 4 : 5;

				indices.push_back(vx);
				if (sx * sy * sz > 0) {
					indices.push_back(vy);
					indices.push_back(vz);
				} else {
					indices.push_back(vz);
					indices.push_back(vy);
				}
			}
		}
	}
}
