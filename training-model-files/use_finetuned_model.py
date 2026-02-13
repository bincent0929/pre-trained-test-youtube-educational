"""
Inference wrapper for fine-tuned BERT educational classifier.

This module provides a simple interface to use the fine-tuned model
for predicting whether text is educational or not.

Usage:
    classifier = EducationalClassifier('./fine_tuned_model')
    result = classifier.predict("Your text here")
    print(result['is_educational'], result['confidence'])
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict
import os

class EducationalClassifier:
    """
    Wrapper class for fine-tuned BERT educational content classifier.

    Attributes:
        model_path: Path to fine-tuned model directory
        tokenizer: BERT tokenizer with special tokens
        model: Fine-tuned BERT model
        device: torch device (cuda or cpu)
    """

    def __init__(self, model_path: str = './fine_tuned_model'):
        """
        Initialize classifier with fine-tuned model.

        Args:
            model_path: Path to directory containing model files

        Raises:
            FileNotFoundError: If model files not found
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Please download and extract the fine-tuned model first."
            )

        print(f"Loading model from {model_path}...")

        # Load tokenizer and model
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Setup device (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        print(f"✓ Model loaded successfully")
        print(f"✓ Device: {self.device}")
        print(f"✓ Vocabulary size: {len(self.tokenizer)}")

    def predict(self, text: str) -> Dict:
        """
        Predict if text is educational.

        Args:
            text: Input text to classify

        Returns:
            Dictionary with:
                - is_educational (bool): True if educational
                - confidence (float): Confidence score (0-1)
                - label (str): 'educational' or 'non_educational'
                - probabilities (dict): Probabilities for each class
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Calculate probabilities
        probabilities = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][prediction].item()

        return {
            'is_educational': prediction == 1,
            'confidence': confidence,
            'label': 'educational' if prediction == 1 else 'non_educational',
            'probabilities': {
                'non_educational': probabilities[0][0].item(),
                'educational': probabilities[0][1].item()
            }
        }

    def predict_batch(self, texts: list) -> list:
        """
        Predict for multiple texts at once (more efficient).

        Args:
            texts: List of input texts

        Returns:
            List of prediction dictionaries
        """
        # Tokenize all inputs
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Calculate probabilities
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

        # Format results
        results = []
        for i in range(len(texts)):
            pred = predictions[i].item()
            conf = probabilities[i][pred].item()

            results.append({
                'is_educational': pred == 1,
                'confidence': conf,
                'label': 'educational' if pred == 1 else 'non_educational',
                'probabilities': {
                    'non_educational': probabilities[i][0].item(),
                    'educational': probabilities[i][1].item()
                }
            })

        return results


def main():
    """
    Test the classifier with sample texts.
    """
    print("="*80)
    print("FINE-TUNED EDUCATIONAL CLASSIFIER - TEST")
    print("="*80)

    # Initialize classifier
    try:
        classifier = EducationalClassifier('./fine_tuned_model')
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have:")
        print("  1. Trained the model in Google Colab")
        print("  2. Downloaded fine_tuned_model.zip")
        print("  3. Extracted it to this directory")
        return

    # Test samples
    test_samples = [
        {
            'text': "Today we're going to learn about machine learning algorithms. "
                   "A neural network is composed of layers of neurons that process information.",
            'expected': 'educational'
        },
        {
            'text': "Don't forget to like and subscribe! Check out my merch link in the "
                   "description below. Let me know in the comments what you think!",
            'expected': 'non_educational'
        },
        {
            'text': "The derivative of x squared is 2x, which we can prove using the limit "
                   "definition. Let's work through this step by step.",
            'expected': 'educational'
        },
        {
            'text': "This gameplay is insane! Watch me get this victory royale. "
                   "Smash that like button if you enjoyed!",
            'expected': 'non_educational'
        },
        {
            'text': "In this lecture, we'll explore the fundamentals of quantum mechanics, "
                   "starting with the wave-particle duality and the Heisenberg uncertainty principle.",
            'expected': 'educational'
        }
    ]

    print("\n" + "="*80)
    print("TEST PREDICTIONS")
    print("="*80)

    correct = 0
    for i, sample in enumerate(test_samples, 1):
        result = classifier.predict(sample['text'])

        is_correct = result['label'] == sample['expected']
        if is_correct:
            correct += 1

        print(f"\nSample {i}:")
        print(f"  Text: {sample['text'][:80]}...")
        print(f"  Expected: {sample['expected']}")
        print(f"  Predicted: {result['label']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Educational probability: {result['probabilities']['educational']:.2%}")
        print(f"  Status: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

    print("\n" + "="*80)
    print(f"ACCURACY: {correct}/{len(test_samples)} ({correct/len(test_samples)*100:.1f}%)")
    print("="*80)

    # Test batch prediction
    print("\n" + "="*80)
    print("BATCH PREDICTION TEST")
    print("="*80)

    batch_texts = [s['text'] for s in test_samples]
    batch_results = classifier.predict_batch(batch_texts)

    print(f"Processed {len(batch_results)} texts in batch")
    for i, (sample, result) in enumerate(zip(test_samples, batch_results), 1):
        print(f"  {i}. {result['label']} ({result['confidence']:.2%})")


if __name__ == '__main__':
    main()
