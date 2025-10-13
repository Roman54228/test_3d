# -*- coding: utf-8 -*-
import sys
import os
import locale

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

print("Testing device initialization with Windows encoding...")
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")
print(f"Default encoding: {sys.getdefaultencoding()}")
print(f"File system encoding: {sys.getfilesystemencoding()}")

try:
    import pyvidu as vidu
    print("pyvidu module imported successfully")
    
    device = vidu.PDdevice()
    print("Device created")
    
    # Попробуем инициализацию с обработкой ошибок
    try:
        result = device.init()
        print(f"Device init result: {result}")
        if result:
            print("Device init successful!")
            print("Available devices:", vidu.getDeviceSerialsNumberList())
        else:
            print("Device init failed - no devices found")
    except UnicodeDecodeError as e:
        print(f"Unicode error: {e}")
        print("Trying alternative approach...")
        
        # Попробуем установить другую кодировку
        import codecs
        original_open = open
        def open_utf8(filename, mode='r', encoding='utf-8', errors='replace'):
            return original_open(filename, mode, encoding=encoding, errors=errors)
        
        # Заменим встроенную функцию open
        __builtins__['open'] = open_utf8
        
        # Попробуем снова
        result = device.init()
        print(f"Device init result (with error handling): {result}")
        
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e)}")
