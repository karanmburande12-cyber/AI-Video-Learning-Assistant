import whisper
import json
import os

# Load model (heavy mat lo agar GPU nahi hai)
model = whisper.load_model("base")

# Folders
AUDIO_FOLDER = "audios"
OUTPUT_FOLDER = "jsons"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

audios = os.listdir(AUDIO_FOLDER)

for audio in audios:
    # Only process mp3 files
    if not audio.endswith(".mp3"):
        continue

    # Expected format: Lecture_1__Variables__Data_Types.mp3
    name = audio.replace(".mp3", "")

    parts = name.split("_")

    # Extract lecture number
    number = "Unknown"
    for part in parts:
        if part.isdigit():
            number = part
            break

    # Extract title (rest of name)
    title = " ".join([p for p in parts if not p.isdigit() and p.lower() != "lecture"])

    print(f"Processing: {number} - {title}")

    try:
        result = model.transcribe(
            audio=f"{AUDIO_FOLDER}/{audio}",
            task="transcribe"   # same language output
        )
    except Exception as e:
        print(f" Error in {audio}: {e}")
        continue

    # Create chunks
    chunks = []
    for segment in result["segments"]:
        chunks.append({
            "number": number,
            "title": title,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })

    # Final JSON structure
    chunks_with_metadata = {
        "chunks": chunks,
        "full_text": result["text"]
    }

    # Save JSON
    json_name = audio.replace(".mp3", ".json")

    with open(f"{OUTPUT_FOLDER}/{json_name}", "w", encoding="utf-8") as f:
        json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)

    print(f"Done: {audio}")