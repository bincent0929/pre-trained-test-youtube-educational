"""
Export training data from results.db with smart sampling strategy.

This script:
1. Queries all videos from the database
2. Converts pre-trained model scores to initial binary labels
3. Applies smart sampling to handle long transcripts:
   - Short (<500 words): Use full text
   - Long: Extract intro (15%), middle samples, conclusion (15%)
4. Outputs JSON for annotation and CSV for quick review
"""

import sqlite3
import json
import csv
from typing import List, Dict

# Configuration
DB_PATH = 'results.db'
SCORE_THRESHOLD = 2.5  # Scores >= 2.5 are initially labeled as educational
MAX_TOKENS_PER_CHUNK = 400  # Leave room for special tokens (BERT limit is 512)
SHORT_TRANSCRIPT_THRESHOLD = 500  # Words

def estimate_tokens(text: str) -> int:
    """
    Rough token estimation: ~4 characters per token for BERT tokenizer.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return len(text) // 4

def smart_sample_transcript(text: str, video_id: str) -> List[Dict]:
    """
    Smart sampling strategy for long transcripts.

    Strategy:
    - Short transcripts (<500 words): Use full text
    - Long transcripts: Extract intro (15%), middle (3-4 chunks), conclusion (15%)
    - Each chunk aims for ~400 tokens with contextual coverage

    Args:
        text: Full transcript text
        video_id: Video identifier

    Returns:
        List of chunk dictionaries with metadata
    """
    chunks = []
    words = text.split()
    total_words = len(words)

    if total_words <= SHORT_TRANSCRIPT_THRESHOLD:
        # Short transcript - use the whole thing
        chunks.append({
            'segment_type': 'full',
            'text': text,
            'position': 0.0,
            'word_count': total_words
        })
    else:
        # Long transcript - smart sampling

        # Intro: First 15%
        intro_end = int(total_words * 0.15)
        intro_text = ' '.join(words[:intro_end])
        chunks.append({
            'segment_type': 'intro',
            'text': intro_text,
            'position': 0.0,
            'word_count': len(words[:intro_end])
        })

        # Middle: Sample 3-4 representative chunks
        middle_start = intro_end
        middle_end = int(total_words * 0.85)
        middle_section = words[middle_start:middle_end]

        if len(middle_section) > 0:
            # Calculate how many chunks we can fit
            chunk_size = MAX_TOKENS_PER_CHUNK * 4  # Convert tokens to approximate words
            num_middle_chunks = max(1, min(4, len(middle_section) // chunk_size))

            # Evenly distribute chunks across middle section
            step = len(middle_section) // num_middle_chunks

            for i in range(num_middle_chunks):
                start_idx = i * step
                end_idx = min(start_idx + chunk_size, len(middle_section))
                chunk_text = ' '.join(middle_section[start_idx:end_idx])

                chunks.append({
                    'segment_type': 'middle',
                    'text': chunk_text,
                    'position': (middle_start + start_idx) / total_words,
                    'word_count': len(middle_section[start_idx:end_idx])
                })

        # Conclusion: Last 15%
        conclusion_start = int(total_words * 0.85)
        conclusion_text = ' '.join(words[conclusion_start:])
        chunks.append({
            'segment_type': 'conclusion',
            'text': conclusion_text,
            'position': 0.85,
            'word_count': len(words[conclusion_start:])
        })

    return chunks

def export_for_annotation():
    """
    Export data from database in annotation-ready format.

    Creates:
    - training_data_for_annotation.json: Full dataset with all fields
    - training_data_summary.csv: Quick overview for review
    """
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute('''
        SELECT video_id, video_title, channel_name, text, score
        FROM results
        ORDER BY score DESC
    ''')

    annotation_data = []

    print("Exporting data from database...")
    print("="*80)

    for row in cursor.fetchall():
        video_id, title, channel, text, score = row

        # Convert score to initial binary label
        initial_label = 1 if score >= SCORE_THRESHOLD else 0
        label_str = "Educational" if initial_label == 1 else "Non-Educational"

        print(f"\nProcessing: {title[:60]}")
        print(f"  Channel: {channel}")
        print(f"  Score: {score:.3f} → Initial Label: {label_str}")

        # Apply smart sampling
        chunks = smart_sample_transcript(text, video_id)
        print(f"  Created {len(chunks)} chunks: {[c['segment_type'] for c in chunks]}")

        # Create annotation entries for each chunk
        for idx, chunk in enumerate(chunks):
            annotation_data.append({
                'video_id': video_id,
                'video_title': title,
                'channel_name': channel,
                'chunk_id': f"{video_id}_{idx}",
                'chunk_index': idx,
                'segment_type': chunk['segment_type'],
                'position': chunk['position'],
                'word_count': chunk['word_count'],
                'estimated_tokens': estimate_tokens(chunk['text']),
                'text': chunk['text'],
                'original_score': score,
                'initial_label': initial_label,
                'annotated_label': None,  # To be filled during annotation
                'special_tokens': [],      # To be filled during annotation
                'annotation_notes': ''     # Optional notes
            })

    conn.close()

    # Save to JSON for annotation
    output_json = 'training_data_for_annotation.json'
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(annotation_data, indent=2, fp=f)

    # Save to CSV for quick review
    output_csv = 'training_data_summary.csv'
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'chunk_id', 'video_id', 'video_title', 'channel_name',
            'segment_type', 'word_count', 'estimated_tokens',
            'initial_label', 'text_preview'
        ])
        writer.writeheader()

        for item in annotation_data:
            writer.writerow({
                'chunk_id': item['chunk_id'],
                'video_id': item['video_id'],
                'video_title': item['video_title'],
                'channel_name': item['channel_name'],
                'segment_type': item['segment_type'],
                'word_count': item['word_count'],
                'estimated_tokens': item['estimated_tokens'],
                'initial_label': item['initial_label'],
                'text_preview': item['text'][:100].replace('\n', ' ') + '...'
            })

    # Print summary statistics
    print("\n" + "="*80)
    print("EXPORT SUMMARY")
    print("="*80)
    print(f"Total chunks created: {len(annotation_data)}")
    print(f"Educational (label=1): {sum(1 for x in annotation_data if x['initial_label'] == 1)}")
    print(f"Non-educational (label=0): {sum(1 for x in annotation_data if x['initial_label'] == 0)}")
    print(f"\nSegment type distribution:")

    segment_counts = {}
    for item in annotation_data:
        seg_type = item['segment_type']
        segment_counts[seg_type] = segment_counts.get(seg_type, 0) + 1

    for seg_type, count in sorted(segment_counts.items()):
        print(f"  {seg_type}: {count}")

    print(f"\nFiles created:")
    print(f"  - {output_json}")
    print(f"  - {output_csv}")
    print(f"\nNext step: Run 'python annotate_data.py' to begin annotation")

if __name__ == '__main__':
    export_for_annotation()
