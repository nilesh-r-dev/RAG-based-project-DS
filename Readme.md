# How to use this RAG AI Teaching assistant on your own data
## Step 1 - Collect your videos
Move all your video files to the videos folder

## Step 2 - Convert to mp3
Convert all the video files to mp3 by ruunning video_to_mp3

## Step 3 - Convert mp3 to json 
Convert all the mp3 files to json by ruunning mp3_to_json

## Step 4 - Convert the json files to Vectors
Use the file preprocess_json to convert the json files to a dataframe with Embeddings and save it as a joblib pickle

## Step 5 - Prompt generation and feeding to LLM

Read the joblib file and load it into the memory. Then create a relevant prompt as per the user query and feed it to the LLM


## Required Tools & Setup

1. Python Libraries

Install required packages:

pip install pandas numpy scikit-learn joblib requests


2. Install FFmpeg (for video → mp3)

👉 Required for video_to_mp3.py

Download: https://ffmpeg.org/download.html
Add to system PATH


3. Install Whisper (for speech-to-text)
pip install openai-whisper


4. Install Ollama (for LLM + Embeddings)

👉 Download: https://ollama.com



project/
│
├── videos/              # Input video files
├── audios/              # Converted mp3 files
├── jsons/               # Transcribed text chunks
├── embeddings.joblib    # Final vector database
│
├── video_to_mp3.py
├── mp3_to_json.py
├── preprocess_json.py
├── process_incoming.py
│
└── README.md
