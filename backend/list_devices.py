#!/usr/bin/env python3
import sounddevice as sd

print("Available audio devices:")
devices = sd.query_devices()
for i, device in enumerate(devices):
    input_ch = device.get('max_input_channels', 0)
    output_ch = device.get('max_output_channels', 0)
    name = device.get('name', 'Unknown')
    hostapi = device.get('hostapi', 0)
    
    device_type = []
    if input_ch > 0:
        device_type.append(f"INPUT({input_ch}ch)")
    if output_ch > 0:
        device_type.append(f"OUTPUT({output_ch}ch)")
        
    print(f"[{i}] {name} - {', '.join(device_type)} - hostapi:{hostapi}")

print(f"\nDefault input: [{sd.default.device[0]}]")
print(f"Default output: [{sd.default.device[1]}]")
