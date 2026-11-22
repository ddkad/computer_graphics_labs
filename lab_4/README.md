# Лабораторная работа №4: Тени (shadow mapping)

## Условие варианта / Задание
В данной работе необходимо реализовать тени от направленного источника света (солнца) методом shadow mapping с использованием графического API Vulkan.

В рамках работы реализовано:
1. Построение карты глубины (shadow map) размером 2048×2048 из позиции источника света с использованием ортографической проекции.
2. Отдельный проход рендера (depth-only pass) без цветовых аттачментов — сцена рендерится только в карту глубины через `VkRenderingInfo`.
3. Сравнение глубины фрагмента с глубиной из карты теней во фрагментном шейдере с учётом `depth bias` для борьбы с артефактами самозатенения (shadow acne).
4. PCF-фильтрация (Percentage-Closer Filtering) с настраиваемым радиусом в текселях для сглаживания краёв тени.
5. Сэмплер теней с `compareEnable = VK_TRUE`, `VK_COMPARE_OP_LESS`, `CLAMP_TO_BORDER` и белой границей (фрагменты вне карты считаются освещёнными).
6. Настройка теней через интерфейс ImGui: включение/выключение, площадь покрытия ортографической матрицы (`Area half-size`), `Depth bias`, радиус PCF, дистанция света, цель (`target`).
7. Тень от движущегося главного октаэдра обновляется каждый кадр — матрица `model` объекта пишется в `ModelData` перед рендером.
8. Использование согласованного UBO `SceneData` (224 байта) с `light_view_proj`, `shadow_params` и `counts`, общего для основного и теневого шейдеров.

## Стек технологий
* **Язык:** C++ (стандарт C++20)
* **Графическое API:** Vulkan
* **Библиотеки:** GLFW, VkBootstrap, VulkanMemoryAllocator (VMA), Dear ImGui, stb_image (для загрузки текстур).
* **Сборка:** CMake

## Как собрать и запустить

### Требования
1. Установленный Vulkan SDK (с добавлением пути в PATH).
2. Компилятор C++ с поддержкой C++20.
3. Утилита CMake (версия 3.20+).
4. Git (для загрузки зависимостей).

### Шаги сборки (Windows / MinGW)
1. Откройте терминал в корне папки lab_4.
2. Сгенерируйте файлы сборки:
   cmake -B build -DCMAKE_BUILD_TYPE=Release
3. Соберите проект:
   cmake --build build --config Release
4. Компиляция шейдеров (если не настроена автоматизация):
   glslc shaders/shadow.vert -o shaders/shadow.vert.spv
   glslc shaders/octahedron.vert -o shaders/octahedron.vert.spv
   glslc shaders/octahedron.frag -o shaders/octahedron.frag.spv

### Запуск
Запустите приложение из папки сборки. Убедитесь, что папки shaders (с файлами .spv) и textures (с изображениями) находятся рядом с исполняемым файлом или в рабочей директории:
cd build
./vulkan-starter-app

## Управление
В левой панели ImGui доступны следующие разделы:

### Раздел "Shadows"
* **Shadows enabled:** Включить/выключить тени (флаг `shadow_params.x`).
* **Area half-size:** Половина размера ортографической области света; слишком малое значение обрезает тени по краю карты.
* **Depth bias:** Смещение глубины для борьбы с acne / peter-panning.
* **PCF radius (texels):** Радиус PCF-фильтра (0 — без сглаживания).
* **Light distance:** Расстояние камеры света от цели.
* **Look-at target:** Точка, на которую смотрит свет.

### Раздел "Ambient & directional light"
* **Yaw / Pitch / Color / Intensity:** Ориентация и параметры солнца (влияет на тени).

### Раздел "Animation (main octahedron)"
* **Play / Pause / Restart:** Управление анимацией главного октаэдра.
* **Trajectory:** Выбор траектории движения (Lissajous curve, Trefoil knot, Breathing spiral).
* **Speed / Radius / Height amplitude / Frequencies:** Параметры траектории.
* **Spin speed / Spin axis:** Вращение объекта вокруг своей оси.

### Раздел "Camera"
* **Projection:** Perspective / Orthographic.
* **Position / Yaw / Pitch / Roll:** Положение и ориентация камеры.
* **FOV / Ortho size / Near / Far:** Параметры проекции.
* **Move speed / Mouse sensitivity:** Скорость движения и чувствительность мыши.