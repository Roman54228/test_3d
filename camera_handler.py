"""
Модуль для работы с камерой и потоками видео
"""
import numpy as np
import cv2
import pyvidu as vidu
import time
from typing import Tuple, Optional, List


class CameraHandler:
    """Класс для работы с камерой и обработки видеопотока"""
    
    def __init__(self, stream_index: int = 1):
        self.device = vidu.PDdevice()
        self.stream_index = stream_index
        self.stream = None
        self.stream_name = None
        self.fps_counter = 0
        self.start_time = time.time()
        self.previous_time = 0
        
    def initialize(self) -> bool:
        """Инициализация устройства камеры"""
        if not self.device.init():
            print("GENERAL ERROR: Failed to initialize camera device")
            return False
        return True
    
    def setup_stream(self) -> bool:
        """Настройка видеопотока"""
        try:
            self.stream = vidu.PDstream(self.device, self.stream_index)
            self.stream.init()
            self.stream_name = self.stream.getStreamName()
            print(f"Stream initialized: {self.stream_name}")
            
            if self.stream_name == "ToF":
                self._configure_tof_camera()
            
            return True
        except Exception as e:
            print(f"Error setting up stream: {e}")
            return False
    
    def _configure_tof_camera(self):
        """Конфигурация ToF камеры"""
        configs = {
            "ToF::StreamFps": 100,
            "ToF::Exposure": 0.1,
            "ToF::Distance": 7.5,
            "ToF::DepthMedianBlur": 0,
            "ToF::DepthFlyingPixelRemoval": 1,
            "ToF::Threshold": 100,
            "ToF::Gain": 9,
            "ToF::AutoExposure": 0,
            "ToF::DepthSmoothStrength": 0,
            "ToF::DepthCompletion": 0
        }
        
        for param, value in configs.items():
            self.stream.set(param, value)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Получение кадра из потока"""
        if not self.stream:
            return None
            
        images = self.stream.getPyImage()
        if len(images) != 2:
            return None
            
        # Обработка ToF изображения
        if self.stream_name == "ToF":
            return self._process_tof_image(images[1])
        
        return images[0] if images else None
    
    def _process_tof_image(self, image: np.ndarray) -> np.ndarray:
        """Обработка ToF изображения"""
        # Переворот изображения
        image = cv2.flip(image, -1)
        
        # Конвертация в 8-бит
        ir_image_8bit = cv2.convertScaleAbs(image, alpha=(2000.0 / 60000.0))
        _, ir_image_8bit = cv2.threshold(ir_image_8bit, 10, 60, cv2.THRESH_TOZERO)
        
        # Конвертация в BGR
        image = np.copy(ir_image_8bit)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        return image
    
    def get_camera_parameters(self) -> Tuple[Optional[object], Optional[object]]:
        """Получение параметров камеры"""
        if not self.stream:
            return None, None
            
        intrinsic = vidu.intrinsics()
        extrinsic = vidu.extrinsics()
        self.stream.getCamPara(intrinsic, extrinsic)
        return intrinsic, extrinsic
    
    def calculate_fps(self) -> float:
        """Расчет FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.previous_time >= 1.0:
            fps = self.fps_counter / (current_time - self.previous_time)
            self.previous_time = current_time
            self.fps_counter = 0
            return fps
        
        return 0.0
    
    def close(self):
        """Закрытие потока"""
        if self.stream:
            self.stream.close()
            self.stream = None