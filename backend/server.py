from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from datetime import datetime
from pathlib import Path

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the recordings directory
recordings_dir = Path("recordings")
app.mount("/recordings", StaticFiles(directory=str(recordings_dir), html=True), name="recordings")

@app.get("/recordings")
async def list_recordings():
    """List all recordings in the recordings directory."""
    if not recordings_dir.exists():
        return []
    
    recordings = []
    for video_file in recordings_dir.glob("*.mp4"):
        name = video_file.stem
        # Get the date from the filename (presentation_YYYYMMDD_HHMMSS)
        try:
            date_str = name.split("_", 1)[1]  # YYYYMMDD_HHMMSS
            date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
            formatted_date = date.strftime("%B %d, %Y %I:%M %p")
        except:
            formatted_date = "Unknown date"
        
        json_file = video_file.with_suffix(".json")
        
        recordings.append({
            "name": f"Recording {formatted_date}",
            "date": formatted_date,
            "videoPath": f"/recordings/{video_file.name}",
            "jsonPath": f"/recordings/{json_file.name}" if json_file.exists() else None
        })
    
    # Sort recordings by date (newest first)
    recordings.sort(key=lambda x: x["date"], reverse=True)
    return recordings

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
