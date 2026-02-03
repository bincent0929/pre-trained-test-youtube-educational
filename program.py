from transformers import AutoTokenizer, AutoModelForSequenceClassification

from youtube_transcript_api import YouTubeTranscriptApi
from pytubefix import YouTube as yt_info_grab

import sqlite3

DB_PATH = 'results.db'

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

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                video_id TEXT NOT NULL,
                text TEXT NOT NULL,
                score REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

video_ids = ["aKTOS0Nrlug", "pAnGwRiQ4-4", "Y0Oa4Lp5fLE", "di0KgqNDqhA", "_C-ZzlGS8Vk", "HAnw168huqA"]

ytt_api = YouTubeTranscriptApi()

yt_info = yt_info_grab(f"https://www.youtube.com/watch?v={video_ids[0]}")
yt_fetch = ytt_api.fetch(video_ids[0], languages=['en', 'en-US'])

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")

text = extract_transcript_text(yt_fetch.to_raw_data())
inputs = tokenizer(str(text), return_tensors="pt", padding="longest", truncation=True)
outputs = model(**inputs)
logits = outputs.logits.squeeze(-1).float().detach().numpy()
score = logits.item()

result = {
    "text": text,
    "video_id": yt_fetch.video_id,
    "video_title": yt_info.title,
    "score": score,
}

print(result)
# {'text': 'This is a test sentence.', 'score': 0.07964489609003067, 'int_score': 0}
