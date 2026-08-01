import cv2
import numpy as np
import wave
import struct
import math
import os
import shutil
import random

artifact_dir = r"C:\Users\swath\.gemini\antigravity-ide\brain\8088e671-a94b-411f-9c13-d6fc89604b07"
base_dir = r"D:\Smart Healthcare\samples"

def move_images():
    files = os.listdir(artifact_dir)
    mapping = {
        "pneumonia": "pneumonia",
        "asthma": "asthma",
        "infection": "respiratory_infection",
        "normal": "normal"
    }
    
    for f in files:
        if f.endswith(".png"):
            for k, folder in mapping.items():
                if f.startswith(f"{k}_xray"):
                    src = os.path.join(artifact_dir, f)
                    dst = os.path.join(base_dir, folder, f"image_{folder}.png")
                    shutil.copy(src, dst)
                    print(f"Copied {f} to {folder}")

def generate_audio(filename, condition):
    sample_rate = 44100
    duration = 3.0
    
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        for i in range(int(sample_rate * duration)):
            t = float(i) / sample_rate
            pulse = math.sin(2.0 * math.pi * 60.0 * t) * math.exp(-3 * (t % 1.0))
            
            if condition == "pneumonia":
                # Add crackles
                if random.random() < 0.1:
                    pulse += (random.random() - 0.5) * 0.5
            elif condition == "asthma":
                # Add wheezing (high freq sine)
                pulse += math.sin(2.0 * math.pi * 400.0 * t) * 0.3 * math.exp(-2 * (t % 1.0))
            elif condition == "respiratory_infection":
                # Irregular breathing
                pulse = math.sin(2.0 * math.pi * 70.0 * t) * math.exp(-3 * ((t * 1.2) % 1.0))
            
            value = int(pulse * 32767.0)
            # Clip
            value = max(-32768, min(32767, value))
            data = struct.pack('<h', value)
            wav_file.writeframesraw(data)
            
def generate_video(filename, condition):
    width, height = 640, 480
    fps = 30
    duration = 3.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for i in range(int(fps * duration)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Base color based on condition
        if condition == "pneumonia":
            color = (50, 50, 100) # cloudy
        elif condition == "asthma":
            color = (50, 100, 50) # tense
        elif condition == "respiratory_infection":
            color = (100, 50, 50) # inflamed
        else:
            color = (100, 100, 100) # normal
            
        cv2.rectangle(frame, (0, 0), (width, height), color, -1)
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        x_pos = int((i / (fps * duration)) * width)
        cv2.line(frame, (x_pos, 0), (x_pos, height), (255, 255, 255), 5)
        
        cv2.putText(frame, f"{condition.upper()} SCAN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
        
    out.release()

conditions = ["pneumonia", "asthma", "respiratory_infection", "normal"]
for c in conditions:
    print(f"Generating for {c}...")
    generate_audio(os.path.join(base_dir, c, f"audio_{c}.wav"), c)
    generate_video(os.path.join(base_dir, c, f"video_{c}.mp4"), c)
    
move_images()
print("Done!")
