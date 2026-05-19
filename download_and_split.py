import os
import yt_dlp
from pydub import AudioSegment
from tqdm import tqdm

# =========================
# CONFIG
# =========================

YOUTUBE_URLS = [
#     "https://www.youtube.com/watch?v=S4ydSSwu1Ws", 
#     "https://www.youtube.com/watch?v=djUiWYzCRFI", 
    # "https://www.youtube.com/watch?v=5jnEizYzQl0"
    # "https://www.youtube.com/watch?v=z5XdX_ryHoc", 
    # "https://www.youtube.com/watch?v=OC48yGWuVNY&list=PLdJ0-R0hxLDNmcI3wJLE2sgN7hTCwtMAp"
    "https://www.youtube.com/watch?v=CgbYmfvfc0k",
    "https://www.youtube.com/watch?v=SIevRg-qBNo", 
    "https://www.youtube.com/watch?v=_u2qggffbYM",
    "https://www.youtube.com/watch?v=7JI-uvmH0jA"
]

DOWNLOAD_DIR = "downloads"
CLIPS_DIR = "clips"

CLIP_DURATION_MS = 5000  # 5 seconds

# =========================
# CREATE FOLDERS
# =========================

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

# =========================
# DOWNLOAD FUNCTION
# =========================

def download_audio(url):

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{DOWNLOAD_DIR}/%(title)s.%(ext)s',

        'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
        }],

        'retries': 10,
        'fragment_retries': 100,
        'socket_timeout': 30,

        'quiet': False,
        'noplaylist': True,
        'download_archive': 'downloaded.txt',
        }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# =========================
# SPLIT AUDIO FUNCTION
# =========================

def split_audio(file_path):

    audio = AudioSegment.from_wav(file_path)

    total_length = len(audio)

    filename = os.path.splitext(os.path.basename(file_path))[0]

    clip_num = 0

    for start in range(0, total_length, CLIP_DURATION_MS):

        end = start + CLIP_DURATION_MS

        clip = audio[start:end]

        clip_name = f"{filename}_clip_{clip_num}.wav"

        clip_path = os.path.join(CLIPS_DIR, clip_name)

        clip.export(clip_path, format="wav")

        clip_num += 1

# =========================
# MAIN PIPELINE
# =========================

print("\nDownloading Audio...\n")

for url in YOUTUBE_URLS:
    download_audio(url)

print("\nSplitting Audio Clips...\n")

for file in tqdm(os.listdir(DOWNLOAD_DIR)):

    if file.endswith(".wav"):

        file_path = os.path.join(DOWNLOAD_DIR, file)

        split_audio(file_path)

print("\nDone!")
print(f"All clips saved in: {CLIPS_DIR}")