# ToF Viewer

Интерактивный визуализатор для ToF камеры с преобразованием координат.

Основной скрипт - это python/ir_click_viewer_simple.py


## Быстрый старт

```bash
# Установка
pip install opencv-python numpy

# Запуск
python ir_click_viewer_simple.py
```

## Использование

- **ЛКМ** - клик по изображению для получения координат
- **Q** - выход

## Конфигурация

Все настройки в классе `Config`:

```python
class Config:
    DISPLAY_SIZE = (640, 480)        # Размер окна
    DEFAULT_DISTANCE_M = 1.49        # Расстояние (метры)
    DEFAULT_TOF_RANGE_M = 7.5        # Диапазон ToF
    
    TOF_PARAMS = {
        "ToF::StreamFps": 5,         # FPS камеры
        "ToF::Exposure": 5.6,        # Экспозиция
        "ToF::Gain": 2,              # Усиление
        # ...
    }
```

## Что показывает

**На экране:**
- Матрица камеры K (левый верхний угол)
- Пиксельные координаты клика
- 2D мировые координаты (X, Y)
- 3D мировые координаты (X, Y, Z)

**В консоли:**
```
CLICK display=(320,240) source=(320,240) depth=1490.0 mm 
world2D=(0.15,0.23) world3D=(0.15,0.23,1.49)
```

## Структура

```
Config              # Настройки приложения
ViewerState         # Состояние (координаты, depth, intrinsics)
CoordinateTransformer  # Преобразование пиксель ↔ мир
```

**Основные функции:**
- `extract_camera_intrinsics()` - извлечение параметров камеры
- `handle_mouse_click()` - обработка кликов
- `run_viewer_loop()` - главный цикл
- `draw_*_overlay()` - отрисовка UI

## Решение проблем

| Ошибка | Решение |
|--------|---------|
| `pyvidu module not found` | Установите Vidu SDK |
| `Device init failed` | Проверьте подключение камеры |
| `No ToF stream found` | Камера не поддерживает ToF |

## Зависимости

- Python 3.7+
- opencv-python
- numpy
- pyvidu (Vidu SDK)
- coordinate_transformer.py
