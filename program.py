from transformers import AutoTokenizer, AutoModelForSequenceClassification

from youtube_transcript_api import YouTubeTranscriptApi
# this is a fragile library
import yt_dlp

import os
# very much think I want to change this out for tinydb
# to avoid sql syntax
import sqlite3

DB_PATH = 'results.db'

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                video_id TEXT NOT NULL,
                video_title TEXT NOT NULL,
                channel_name TEXT NOT NULL,
                text TEXT NOT NULL,
                model_used TEXT NOT NULL,
                score REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

def save_result(result: dict): 
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO results (video_id, video_title, channel_name, text, model_used, score) VALUES (?, ?, ?, ?, ?, ?)",
            (result["video_id"], result["video_title"], result["channel_name"], result["text"], result["model_used"], result["score"])
        )

if not os.path.exists(DB_PATH):
    init_db()

def extract_transcript_text(transcript_data: list[dict], join_with: str = " ") -> str:
    """
    Extracts only the spoken text from youtube_transcript_api transcript data.
    
    Args:
        transcript_data: List of dicts from youtube_transcript_api 
                        (e.g., YouTubeTranscriptApi.get_transcript(video_id))
        join_with: String to join text segments (default: single space)
    
    Returns:
        Clean string of just the spoken words
    """
    text_segments = [entry['text'].replace('\n', ' ') for entry in transcript_data if 'text' in entry]
    return join_with.join(text_segments)

def video_is_saved(video_id: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT video_id, video_title, channel_name, text, model_used, score FROM results WHERE video_id = ? LIMIT 1",
            (video_id,)
        )
        row = cursor.fetchone()

    if row is None:
        return False
    
    print({
        #"video_id": row[0],
        "video_title": row[1],
        "channel_name": row[2],
        #"text": row[3],
        #"model_used": row[4],
        "score": row[5],
    })
    return True

video_ids = ["aKTOS0Nrlug",
            "pAnGwRiQ4-4",
            "Y0Oa4Lp5fLE",
            "di0KgqNDqhA",
            "_C-ZzlGS8Vk", 
            "HAnw168huqA", 
            "nBtOEmUqASQ", 
            "9pcrzvK_U0k", 
            "-Y23nfAOiXQ",
            "9Ge0sMm65jo"]

for video in video_ids:

    if video_is_saved(video) is True:
        continue

    ytt_api = YouTubeTranscriptApi()

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video}", download=False)
    # 'Unknown' is the fallback value if a value isn't returned
    title = info.get('title', 'Unknown')
    channel_name = info.get('uploader', 'Unknown')

    yt_fetch = ytt_api.fetch(video, languages=['en', 'en-US'])

    model_name = "HuggingFaceTB/fineweb-edu-classifier"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    text = extract_transcript_text(yt_fetch.to_raw_data())
    inputs = tokenizer(str(text), return_tensors="pt", padding="longest", truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits.squeeze(-1).float().detach().numpy()
    score = logits.item()

    result = {
        "video_id": yt_fetch.video_id,
        "video_title": title,
        "channel_name": channel_name,
        "text": text,
        "model_used": model_name,
        "score": score,
    }

    print(result)
    save_result(result)
