# Convert videos to mp3
import os
import subprocess

files = os.listdir("videos")

for file in files:
    if file.endswith(".mp4"):
        # Remove extension
        name = file.replace(".mp4", "")
        
        # Replace spaces with underscore (optional)
        safe_name = name.replace(" ", "_")
        
        print(f"Converting: {file}")
        
        subprocess.run([
            "ffmpeg",
            "-i", f"videos/{file}",
            f"audios/{safe_name}.mp3"
        ])