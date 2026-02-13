"""
Prepare annotated data for Google Colab training.

This script:
1. Loads annotated data
2. Inserts special tokens at specified word positions
3. Creates train/validation split (80/20)
4. Generates metadata file
5. Outputs JSON files ready for Colab upload
"""

import json
import random
from typing import List, Dict

# Special token markers (will be inserted into text)
TOKEN_MARKERS = {
    'edu': {'start': '[EDU_START]', 'end': '[EDU_END]'},
    'non_edu': {'start': '[NON_EDU_START]', 'end': '[NON_EDU_END]'}
}

def insert_special_tokens(text: str, special_tokens: List[Dict]) -> str:
    """
    Insert special tokens at specified word positions in text.

    Args:
        text: Original text
        special_tokens: List of token specifications with start, end, type

    Returns:
        Text with special tokens inserted
    """
    if not special_tokens:
        return text

    words = text.split()

    # Sort tokens by start position (reverse order for insertion from end)
    # This prevents index shifting when inserting
    sorted_tokens = sorted(special_tokens, key=lambda x: x['start'], reverse=True)

    for token in sorted_tokens:
        start = token['start']
        end = token['end']
        token_type = token['type']

        # Validate indices
        if start < 0 or end > len(words) or start >= end:
            print(f"Warning: Invalid token range {start}:{end} for text with {len(words)} words. Skipping.")
            continue

        # Get token markers
        if token_type not in TOKEN_MARKERS:
            print(f"Warning: Unknown token type '{token_type}'. Skipping.")
            continue

        markers = TOKEN_MARKERS[token_type]

        # Insert end marker first (to maintain correct indices)
        words.insert(end, markers['end'])
        # Then insert start marker
        words.insert(start, markers['start'])

    return ' '.join(words)

def prepare_dataset():
    """
    Prepare annotated data for training.

    Reads annotated_training_data.json and creates:
    - train_dataset.json: Training set
    - val_dataset.json: Validation set
    - dataset_metadata.json: Dataset information
    """
    # Load annotated data
    try:
        with open('annotated_training_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: File 'annotated_training_data.json' not found.")
        print("Please run 'python annotate_data.py' first to annotate your data.")
        return

    print("Preparing dataset for Google Colab training...")
    print("="*80)

    # Filter only annotated chunks (where annotated_label is not None)
    annotated_data = [chunk for chunk in data if chunk.get('annotated_label') is not None]

    if len(annotated_data) == 0:
        print("Error: No annotated data found.")
        print("Please run 'python annotate_data.py' to annotate your data first.")
        return

    print(f"Found {len(annotated_data)} annotated chunks out of {len(data)} total")

    # Prepare training samples
    training_samples = []

    for chunk in annotated_data:
        # Insert special tokens into text
        text_with_tokens = insert_special_tokens(
            chunk['text'],
            chunk.get('special_tokens', [])
        )

        training_samples.append({
            'text': text_with_tokens,
            'label': chunk['annotated_label'],
            'video_id': chunk['video_id'],
            'chunk_id': chunk['chunk_id'],
            'video_title': chunk['video_title'],
            'channel_name': chunk['channel_name'],
            'segment_type': chunk['segment_type']
        })

    # Shuffle data for random split
    random.seed(42)
    random.shuffle(training_samples)

    # Split into train/validation (80/20)
    split_idx = int(len(training_samples) * 0.8)
    train_data = training_samples[:split_idx]
    val_data = training_samples[split_idx:]

    # Calculate statistics
    total_samples = len(training_samples)
    edu_count = sum(1 for x in training_samples if x['label'] == 1)
    non_edu_count = sum(1 for x in training_samples if x['label'] == 0)

    train_edu = sum(1 for x in train_data if x['label'] == 1)
    train_non_edu = sum(1 for x in train_data if x['label'] == 0)

    val_edu = sum(1 for x in val_data if x['label'] == 1)
    val_non_edu = sum(1 for x in val_data if x['label'] == 0)

    # Count special tokens used
    special_token_count = sum(
        len(chunk.get('special_tokens', []))
        for chunk in annotated_data
    )

    # Save train dataset
    with open('train_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, indent=2, fp=f)

    # Save validation dataset
    with open('val_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, indent=2, fp=f)

    # Create metadata
    metadata = {
        'dataset_info': {
            'total_samples': total_samples,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'train_val_split': 0.8,
            'random_seed': 42
        },
        'label_distribution': {
            'total': {
                'educational': edu_count,
                'non_educational': non_edu_count
            },
            'train': {
                'educational': train_edu,
                'non_educational': train_non_edu
            },
            'validation': {
                'educational': val_edu,
                'non_educational': val_non_edu
            }
        },
        'special_tokens': [
            '[EDU_START]',
            '[EDU_END]',
            '[NON_EDU_START]',
            '[NON_EDU_END]'
        ],
        'special_tokens_count': special_token_count,
        'class_names': {
            0: 'non_educational',
            1: 'educational'
        }
    }

    # Save metadata
    with open('dataset_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, indent=2, fp=f)

    # Print summary
    print("\n" + "="*80)
    print("DATASET PREPARATION COMPLETE")
    print("="*80)
    print(f"\nTotal samples: {total_samples}")
    print(f"  Training set: {len(train_data)} ({len(train_data)/total_samples*100:.1f}%)")
    print(f"  Validation set: {len(val_data)} ({len(val_data)/total_samples*100:.1f}%)")

    print(f"\nLabel distribution:")
    print(f"  Overall:")
    print(f"    Educational: {edu_count} ({edu_count/total_samples*100:.1f}%)")
    print(f"    Non-educational: {non_edu_count} ({non_edu_count/total_samples*100:.1f}%)")

    print(f"  Training set:")
    print(f"    Educational: {train_edu}")
    print(f"    Non-educational: {train_non_edu}")

    print(f"  Validation set:")
    print(f"    Educational: {val_edu}")
    print(f"    Non-educational: {val_non_edu}")

    print(f"\nSpecial tokens:")
    print(f"  Total annotations with special tokens: {special_token_count}")
    print(f"  Token vocabulary: {', '.join(metadata['special_tokens'])}")

    # Check for class imbalance
    imbalance_ratio = max(edu_count, non_edu_count) / min(edu_count, non_edu_count) if min(edu_count, non_edu_count) > 0 else float('inf')
    if imbalance_ratio > 2.0:
        print(f"\n⚠ WARNING: Class imbalance detected (ratio: {imbalance_ratio:.2f}:1)")
        print("  Consider using class weights in training or collecting more data for minority class")

    print(f"\n✓ Files created:")
    print(f"  - train_dataset.json ({len(train_data)} samples)")
    print(f"  - val_dataset.json ({len(val_data)} samples)")
    print(f"  - dataset_metadata.json")

    print(f"\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Upload these 3 files to Google Colab:")
    print("   - train_dataset.json")
    print("   - val_dataset.json")
    print("   - dataset_metadata.json")
    print()
    print("2. Open bert_edu_classifier_training.ipynb in Colab")
    print("3. Set runtime to GPU (Runtime → Change runtime type → T4 GPU)")
    print("4. Run all cells to train your model")
    print("="*80)

if __name__ == '__main__':
    prepare_dataset()
