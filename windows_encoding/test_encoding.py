# -*- coding: utf-8 -*-
import sys
import os

# Принудительно установить кодировку
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

import pyvidu as vidu

print("Testing device initialization...")
device = vidu.PDdevice()
print("Device created")

try:
    result = device.init()
    print(f"Device init result: {result}")
    if result:
        print("Device init successful!")
        print("Available devices:", vidu.getDeviceSerialsNumberList())
    else:
        print("Device init failed - no devices found")
except Exception as e:
    print(f"Error during device init: {e}")
    print(f"Error type: {type(e)}")
