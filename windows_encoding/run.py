# -*- coding: utf-8 -*-
import sys
import os
import locale
import cv2
import numpy as np

# Установить кодировку для Windows
if sys.platform.startswith('win'):
    # Установить кодировку консоли
    os.system('chcp 65001')
    
    # Установить кодировку для Python
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    
    # Установить локаль
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

print("ViduSDK Camera Stream Demo")
print("=" * 40)

try:
    import pyvidu as vidu
    print("✓ pyvidu module imported successfully")
    
    # Создаем объекты для калибровки
    intrinsic = vidu.intrinsics()
    extrinsic = vidu.extrinsics()
    
    # Создаем устройство
    device = vidu.PDdevice()
    print("✓ Device created")
    
    # Инициализируем устройство
    if not device.init():
        print("✗ Device initialization failed")
        print("Make sure camera is connected and drivers are installed")
        exit(-1)
    
    print("✓ Device initialized successfully")
    print(f"✓ Available devices: {device.getSerialsNumber()}")
    
    # Получаем количество потоков
    stream_num = device.getStreamNum()
    print(f"✓ Number of streams: {stream_num}")
    
    # Обрабатываем каждый поток
    for i in range(stream_num):
        print(f"\n--- Stream {i} ---")
        
        with vidu.PDstream(device, i) as stream:
            # Инициализируем поток
            if not stream.init():
                print(f"✗ Stream {i} initialization failed")
                continue
            
            print(f"✓ Stream {i} initialized successfully")
            stream_name = stream.getStreamName()
            print(f"✓ Stream name: {stream_name}")
            
            # Получаем параметры камеры
            if stream.getCamPara(intrinsic, extrinsic):
                print("✓ Camera parameters obtained")
                vidu.print_intrinsics(intrinsic)
                vidu.print_extrinsics(extrinsic)
            
            # Настраиваем параметры потока
            if stream_name == "ToF":
                stream.set("Distance", 2.5)
                stream.set("StreamFps", 30)
                print("✓ ToF stream configured")
            elif stream_name == "RGB":
                stream.set("StreamFps", 30)
                print("✓ RGB stream configured")
            elif stream_name == "PCL":
                stream.set("ToF::Distance", 2.5)
                print("✓ PCL stream configured")
            
            print(f"\n🎥 Starting {stream_name} stream...")
            print("Press 'q' to quit, 's' to save frame, 'c' to capture")
            
            frame_count = 0
            
            while True:
                try:
                    # Получаем кадры
                    if stream_name == "IMU":
                        # Для IMU данных
                        imudata, ret = stream.GetImuData()
                        if ret:
                            # Создаем пустое изображение для отображения IMU данных
                            blank = np.zeros((600, 800, 3), np.uint8)
                            blank.fill(255)
                            
                            # Отображаем данные акселерометра
                            text = f"Accelerometer time: {imudata.AcceTime}"
                            cv2.putText(blank, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            
                            text = f"x: {imudata.AcceData[0]:.6f}, y: {imudata.AcceData[1]:.6f}, z: {imudata.AcceData[2]:.6f}"
                            cv2.putText(blank, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            
                            # Отображаем данные гироскопа
                            text = f"Gyroscope time: {imudata.GyroTime}"
                            cv2.putText(blank, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            
                            text = f"x: {imudata.GyroData[0]:.6f}, y: {imudata.GyroData[1]:.6f}, z: {imudata.GyroData[2]:.6f}"
                            cv2.putText(blank, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            
                            cv2.imshow(f"IMU Stream {i}", blank)
                    else:
                        # Для обычных изображений
                        images = stream.getPyImage()
                        if images:
                            for j, image in enumerate(images):
                                if image is not None and image.size > 0:
                                    # Добавляем информацию о кадре
                                    frame_info = f"{stream_name} {i}-{j} | Frame: {frame_count}"
                                    cv2.putText(image, frame_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    
                                    cv2.imshow(f"{stream_name} Stream {i}-{j}", image)
                    
                    frame_count += 1
                    
                    # Обработка клавиш
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("✓ Quitting...")
                        break
                    elif key == ord('s'):
                        # Сохраняем текущий кадр
                        if stream_name != "IMU" and images:
                            for j, image in enumerate(images):
                                if image is not None and image.size > 0:
                                    filename = f"capture_{stream_name}_{i}_{j}_{frame_count}.png"
                                    cv2.imwrite(filename, image)
                                    print(f"✓ Saved: {filename}")
                    elif key == ord('c'):
                        # Захватываем кадр (то же что и 's')
                        if stream_name != "IMU" and images:
                            for j, image in enumerate(images):
                                if image is not None and image.size > 0:
                                    filename = f"capture_{stream_name}_{i}_{j}_{frame_count}.png"
                                    cv2.imwrite(filename, image)
                                    print(f"✓ Captured: {filename}")
                
                except Exception as e:
                    print(f"✗ Error processing frame: {e}")
                    continue
            
            # Закрываем окна для этого потока
            cv2.destroyAllWindows()
            print(f"✓ Stream {i} finished")

except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure pyvidu module is installed")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    print(f"Error type: {type(e)}")

print("\n�� Demo finished!")
