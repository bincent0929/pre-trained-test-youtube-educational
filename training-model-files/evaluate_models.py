"""
Compare pre-trained vs fine-tuned model predictions.

This script evaluates both models on the existing database videos
to compare their predictions and assess improvements from fine-tuning.
"""

import sqlite3
from use_finetuned_model import EducationalClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

DB_PATH = 'results.db'
PRETRAINED_MODEL = "HuggingFaceTB/fineweb-edu-classifier"

def load_pretrained_model():
    """Load the original pre-trained model."""
    print("Loading pre-trained model...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print(f"✓ Pre-trained model loaded ({PRETRAINED_MODEL})")
    return tokenizer, model, device

def predict_pretrained(text, tokenizer, model, device):
    """Get prediction from pre-trained model."""
    inputs = tokenizer(str(text), return_tensors="pt", padding="longest", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.squeeze(-1).float()
    score = logits.item()

    return score

def sample_text(text, max_words=400):
    """Sample first 400 words from text for consistent comparison."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words])

def compare_models():
    """
    Compare pre-trained vs fine-tuned model predictions on database videos.
    """
    print("="*100)
    print("MODEL COMPARISON: Pre-trained vs Fine-tuned")
    print("="*100)

    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"\nError: Database '{DB_PATH}' not found.")
        return

    # Load models
    print("\n1. Loading models...")
    print("-"*100)

    try:
        finetuned = EducationalClassifier('./fine_tuned_model')
    except FileNotFoundError:
        print("\nError: Fine-tuned model not found at './fine_tuned_model'")
        print("Please ensure you have:")
        print("  1. Trained the model in Google Colab")
        print("  2. Downloaded fine_tuned_model.zip")
        print("  3. Extracted it to this directory")
        return

    pretrained_tokenizer, pretrained_model, pretrained_device = load_pretrained_model()

    # Load data from database
    print("\n2. Loading videos from database...")
    print("-"*100)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute('''
        SELECT video_id, video_title, channel_name, text, score
        FROM results
        ORDER BY score DESC
    ''')

    videos = cursor.fetchall()
    conn.close()

    print(f"✓ Loaded {len(videos)} videos")

    # Compare predictions
    print("\n3. Comparing predictions...")
    print("="*100)
    print(f"{'Video Title':<45} {'Channel':<20} {'Pre-trained':<15} {'Fine-tuned':<15} {'Agreement':<12}")
    print("="*100)

    agreements = 0
    results = []

    for video_id, title, channel, text, pretrained_score in videos:
        # Sample text for fair comparison (first 400 words)
        sample = sample_text(text, max_words=400)

        # Get fine-tuned prediction
        finetuned_result = finetuned.predict(sample)

        # Convert pre-trained score to label (using threshold 2.5)
        pretrained_label = 'educational' if pretrained_score >= 2.5 else 'non_educational'

        # Check agreement
        agree = pretrained_label == finetuned_result['label']
        if agree:
            agreements += 1

        # Store result
        results.append({
            'title': title,
            'channel': channel,
            'pretrained_score': pretrained_score,
            'pretrained_label': pretrained_label,
            'finetuned_label': finetuned_result['label'],
            'finetuned_confidence': finetuned_result['confidence'],
            'agree': agree
        })

        # Print row
        title_short = title[:42] + "..." if len(title) > 45 else title
        channel_short = channel[:17] + "..." if len(channel) > 20 else channel

        pretrained_str = f"{pretrained_score:.2f} ({pretrained_label[:3]})"
        finetuned_str = f"{finetuned_result['label'][:3]} ({finetuned_result['confidence']:.1%})"
        agree_str = "✓ Yes" if agree else "✗ No"

        print(f"{title_short:<45} {channel_short:<20} {pretrained_str:<15} {finetuned_str:<15} {agree_str:<12}")

    print("="*100)

    # Summary statistics
    print("\n4. Summary Statistics")
    print("="*100)

    print(f"Total videos compared: {len(videos)}")
    print(f"Agreement rate: {agreements}/{len(videos)} ({agreements/len(videos)*100:.1f}%)")
    print(f"Disagreement rate: {len(videos)-agreements}/{len(videos)} ({(len(videos)-agreements)/len(videos)*100:.1f}%)")

    # Label distribution
    print(f"\nLabel distribution:")

    pretrained_edu = sum(1 for r in results if r['pretrained_label'] == 'educational')
    pretrained_non = sum(1 for r in results if r['pretrained_label'] == 'non_educational')
    finetuned_edu = sum(1 for r in results if r['finetuned_label'] == 'educational')
    finetuned_non = sum(1 for r in results if r['finetuned_label'] == 'non_educational')

    print(f"  Pre-trained:")
    print(f"    Educational: {pretrained_edu} ({pretrained_edu/len(videos)*100:.1f}%)")
    print(f"    Non-educational: {pretrained_non} ({pretrained_non/len(videos)*100:.1f}%)")

    print(f"  Fine-tuned:")
    print(f"    Educational: {finetuned_edu} ({finetuned_edu/len(videos)*100:.1f}%)")
    print(f"    Non-educational: {finetuned_non} ({finetuned_non/len(videos)*100:.1f}%)")

    # Detailed disagreements
    disagreements = [r for r in results if not r['agree']]
    if disagreements:
        print(f"\n5. Detailed Disagreements ({len(disagreements)} cases)")
        print("="*100)

        for r in disagreements:
            print(f"\nVideo: {r['title']}")
            print(f"  Channel: {r['channel']}")
            print(f"  Pre-trained: {r['pretrained_label']} (score: {r['pretrained_score']:.3f})")
            print(f"  Fine-tuned: {r['finetuned_label']} (confidence: {r['finetuned_confidence']:.1%})")

    print("\n" + "="*100)
    print("Comparison complete!")
    print("="*100)

    # Confidence analysis for fine-tuned model
    print("\n6. Fine-tuned Model Confidence Analysis")
    print("="*100)

    confidences = [r['finetuned_confidence'] for r in results]
    avg_confidence = sum(confidences) / len(confidences)
    min_confidence = min(confidences)
    max_confidence = max(confidences)

    high_conf = sum(1 for c in confidences if c >= 0.9)
    medium_conf = sum(1 for c in confidences if 0.7 <= c < 0.9)
    low_conf = sum(1 for c in confidences if c < 0.7)

    print(f"Average confidence: {avg_confidence:.2%}")
    print(f"Min confidence: {min_confidence:.2%}")
    print(f"Max confidence: {max_confidence:.2%}")

    print(f"\nConfidence distribution:")
    print(f"  High (≥90%): {high_conf} ({high_conf/len(videos)*100:.1f}%)")
    print(f"  Medium (70-90%): {medium_conf} ({medium_conf/len(videos)*100:.1f}%)")
    print(f"  Low (<70%): {low_conf} ({low_conf/len(videos)*100:.1f}%)")

    if low_conf > 0:
        print(f"\n⚠ Low confidence predictions:")
        for r in sorted(results, key=lambda x: x['finetuned_confidence']):
            if r['finetuned_confidence'] < 0.7:
                print(f"  - {r['title'][:60]}: {r['finetuned_label']} ({r['finetuned_confidence']:.1%})")


def main():
    """Main entry point."""
    try:
        compare_models()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
