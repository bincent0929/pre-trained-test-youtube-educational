from transformers import AutoTokenizer, AutoModelForSequenceClassification
from youtube_transcript_api import YouTubeTranscriptApi

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
    text_segments = [entry['text'] for entry in transcript_data if 'text' in entry]
    return join_with.join(text_segments)

#video_id = "aKTOS0Nrlug"
#video_id = "pAnGwRiQ4-4"
#video_id = "Y0Oa4Lp5fLE"
video_id = "di0KgqNDqhA"
ytt_api = YouTubeTranscriptApi()
yt_fetch = ytt_api.fetch(video_id)

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")
model = AutoModelForSequenceClassification.from_pretrained("HuggingFaceTB/fineweb-edu-classifier")

text = extract_transcript_text(yt_fetch.to_raw_data())
inputs = tokenizer(str(text), return_tensors="pt", padding="longest", truncation=True)
outputs = model(**inputs)
logits = outputs.logits.squeeze(-1).float().detach().numpy()
score = logits.item()
result = {
    "text": text,
    "video_title": yt_fetch.video_id,
    "score": score,
    "int_score": int(round(max(0, min(score, 5)))),
}

print(result)
# {'text': 'This is a test sentence.', 'score': 0.07964489609003067, 'int_score': 0}
