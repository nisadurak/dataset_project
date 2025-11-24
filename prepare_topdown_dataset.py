import subprocess
from pathlib import Path

# ================== AYARLAR ==================
FFMPEG_PATH = r"C:\\ffmpeg\\bin\\ffmpeg.exe"  # ffmpeg.exe tam yolu

CATEGORY = "Third-Person"


VIDEOS = [
    "videos/videoplayback .mp4"
   
    
    
]


SEGMENTS = [
    ("start", "00:10:00", "00:03:00"),   
    ("mid",   "00:13:01", "00:03:00"),   
    ("end",   "00:17:01", "00:03:00"),  
]


OUTPUT_ROOT = Path("dataset_raw")  


# ================== FONKSİYON ==================

def extract_frames():
    for vid_idx, video_path in enumerate(VIDEOS, start=1):
        video_path = Path(video_path)

        if not video_path.exists():
            print(f"[UYARI] Video yok: {video_path}")
            continue

        print(f"\n[+] Video {vid_idx}: {video_path}")

        for seg_name, start_time, duration in SEGMENTS:
           
            out_dir = OUTPUT_ROOT / CATEGORY / f"game{vid_idx}_{seg_name}"
            out_dir.mkdir(parents=True, exist_ok=True)

      
            out_pattern = str(out_dir / "%04d.jpg")
              

            cmd = [
                FFMPEG_PATH,
                "-ss", start_time,   # başlangıç
                "-t", duration,      # uzunluk
                "-i", str(video_path),
                "-vf", "fps=1",      # saniyede 1 kare
                out_pattern,
                "-loglevel", "error",    # konsolu çöp yapmasın
                "-y",                   # var olan dosyaları ez
            ]

            print(f"  -> {seg_name} segmenti ({start_time} + {duration})")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"  [HATA] ffmpeg çalışırken sorun: {e}")

    print("\n[✓] İşlem bitti. Frame'ler şu klasörde:")
    print(f"    {OUTPUT_ROOT / CATEGORY}")


if __name__ == "__main__":
    extract_frames()
