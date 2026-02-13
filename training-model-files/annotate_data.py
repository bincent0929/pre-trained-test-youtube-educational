"""
Interactive annotation interface for training data.

This script provides a command-line interface to:
1. Review each transcript chunk
2. Confirm or change binary labels (educational vs non-educational)
3. Add special tokens to mark definitively educational/non-educational passages
4. Save progress incrementally (can resume if interrupted)
"""

import json
import sys
from typing import List, Dict, Optional

# Special token definitions
SPECIAL_TOKENS = {
    'edu': '[EDU]',
    'non_edu': '[NON_EDU]'
}

class AnnotationInterface:
    """Interactive interface for annotating training data."""

    def __init__(self, data_file='training_data_for_annotation.json'):
        """
        Initialize annotation interface.

        Args:
            data_file: Path to JSON file with data to annotate
        """
        self.data_file = data_file
        self.data = []
        self.current_idx = 0

        # Load data
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"Loaded {len(self.data)} chunks from {data_file}")
        except FileNotFoundError:
            print(f"Error: File '{data_file}' not found.")
            print("Please run 'python export_training_data.py' first.")
            sys.exit(1)

        # Load progress if exists
        self.load_progress()

    def load_progress(self):
        """Load annotation progress if exists."""
        try:
            with open('annotation_progress.json', 'r', encoding='utf-8') as f:
                progress = json.load(f)
                self.current_idx = progress.get('last_index', 0)
                print(f"Resuming from chunk {self.current_idx + 1}")
        except FileNotFoundError:
            self.current_idx = 0
            print("Starting new annotation session")

    def save_progress(self):
        """Save current progress to allow resuming later."""
        # Save annotated data
        with open('annotated_training_data.json', 'w', encoding='utf-8') as f:
            json.dump(self.data, indent=2, fp=f)

        # Save progress marker
        with open('annotation_progress.json', 'w', encoding='utf-8') as f:
            json.dump({'last_index': self.current_idx}, fp=f)

    def display_chunk(self, chunk: Dict):
        """
        Display chunk information for annotation.

        Args:
            chunk: Chunk dictionary with metadata and text
        """
        print("\n" + "="*80)
        print(f"CHUNK {self.current_idx + 1} / {len(self.data)}")
        print("="*80)
        print(f"Video: {chunk['video_title']}")
        print(f"Channel: {chunk['channel_name']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Segment: {chunk['segment_type']} (position: {chunk['position']:.2f})")
        print(f"Words: {chunk['word_count']} (~{chunk['estimated_tokens']} tokens)")
        print(f"Original Score: {chunk['original_score']:.3f}")
        print(f"Initial Label: {'Educational (1)' if chunk['initial_label'] == 1 else 'Non-Educational (0)'}")
        print("-"*80)
        print("TEXT:")
        print("-"*80)

        # Display text with word numbers for reference (first 1000 chars)
        text = chunk['text']
        if len(text) > 1000:
            print(text[:1000])
            print(f"\n[... {len(text) - 1000} more characters ...]")
            print(f"\nFull text has {len(text.split())} words")
        else:
            print(text)

        print("="*80)

    def annotate_chunk(self, chunk: Dict):
        """
        Annotate a single chunk interactively.

        Args:
            chunk: Chunk dictionary to annotate
        """
        self.display_chunk(chunk)

        # Step 1: Confirm or change label
        print("\n" + "-"*80)
        print("STEP 1: Confirm or change the label")
        print("-"*80)
        print("1 = Educational (lectures, tutorials, explanations)")
        print("0 = Non-Educational (entertainment, vlogs, promotional)")
        print(f"Current initial label: {chunk['initial_label']}")

        while True:
            label_input = input(f"\nLabel (1/0, or press Enter to keep {chunk['initial_label']}): ").strip()

            if label_input == '':
                chunk['annotated_label'] = chunk['initial_label']
                break
            elif label_input in ['0', '1']:
                chunk['annotated_label'] = int(label_input)
                break
            else:
                print("Invalid input. Please enter 1, 0, or press Enter.")

        print(f"✓ Label set to: {'Educational' if chunk['annotated_label'] == 1 else 'Non-Educational'}")

        # Step 2: Add special tokens (optional)
        print("\n" + "-"*80)
        print("STEP 2: Add special tokens (optional)")
        print("-"*80)
        print("Mark specific passages that are DEFINITIVELY educational or non-educational.")
        print("\nFormat: start_word:end_word:type")
        print("  - start_word: Starting word index (0-based)")
        print("  - end_word: Ending word index (exclusive)")
        print(f"  - type: 'edu' or 'non_edu'")
        print("\nExample: '0:50:edu' marks words 0-49 as definitively educational")
        print("Enter 'done' when finished, or 'skip' to skip token annotation")

        tokens = []

        while True:
            token_input = input("\nToken annotation (or 'done'/'skip'): ").strip()

            if token_input.lower() in ['done', 'skip', '']:
                break

            # Parse input
            try:
                parts = token_input.split(':')
                if len(parts) != 3:
                    print("  ✗ Invalid format. Use 'start:end:type' (e.g., '0:50:edu')")
                    continue

                start, end, token_type = parts
                start = int(start)
                end = int(end)

                if token_type not in ['edu', 'non_edu']:
                    print("  ✗ Invalid type. Use 'edu' or 'non_edu'")
                    continue

                if start < 0 or end <= start:
                    print("  ✗ Invalid range. Ensure 0 <= start < end")
                    continue

                # Add token
                tokens.append({
                    'start': start,
                    'end': end,
                    'type': token_type
                })

                token_label = SPECIAL_TOKENS.get(token_type, '[UNKNOWN]')
                print(f"  ✓ Added: {token_label} for words {start}-{end-1}")

            except ValueError:
                print("  ✗ Invalid format. Start and end must be integers.")

        chunk['special_tokens'] = tokens

        if len(tokens) > 0:
            print(f"✓ Added {len(tokens)} special token(s)")
        else:
            print("✓ No special tokens added")

        # Step 3: Optional notes
        print("\n" + "-"*80)
        print("STEP 3: Optional notes")
        print("-"*80)

        notes = input("Any notes about this chunk? (or press Enter to skip): ").strip()
        chunk['annotation_notes'] = notes

        if notes:
            print(f"✓ Notes saved: {notes}")

    def run(self):
        """Run the annotation interface."""
        print("\n" + "="*80)
        print("TRAINING DATA ANNOTATION INTERFACE")
        print("="*80)
        print(f"Total chunks: {len(self.data)}")
        print(f"Progress: {self.current_idx}/{len(self.data)} chunks annotated")
        print("\nInstructions:")
        print("  - Review each chunk and confirm/adjust the label")
        print("  - Optionally add special tokens to mark key passages")
        print("  - Progress is saved after each chunk")
        print("  - You can quit anytime and resume later")
        print("="*80)

        # Count already annotated
        already_annotated = sum(1 for chunk in self.data if chunk.get('annotated_label') is not None)
        if already_annotated > 0:
            print(f"\n{already_annotated} chunks already annotated (will be skipped)")

        input("\nPress Enter to begin...")

        # Iterate through chunks
        while self.current_idx < len(self.data):
            chunk = self.data[self.current_idx]

            # Skip if already annotated (unless user wants to re-annotate)
            if chunk.get('annotated_label') is not None:
                print(f"\nChunk {self.current_idx + 1} already annotated. Skipping...")
                self.current_idx += 1
                continue

            # Annotate chunk
            try:
                self.annotate_chunk(chunk)
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Saving progress...")
                self.save_progress()
                print("Progress saved. Run this script again to resume.")
                sys.exit(0)

            # Save progress after each annotation
            self.save_progress()
            self.current_idx += 1

            # Ask to continue
            if self.current_idx < len(self.data):
                print("\n" + "-"*80)
                remaining = len(self.data) - self.current_idx
                print(f"Progress: {self.current_idx}/{len(self.data)} ({remaining} remaining)")

                cont = input("Continue to next chunk? (y/n/q to quit): ").strip().lower()

                if cont in ['q', 'n']:
                    print("\nProgress saved. Exiting...")
                    break

        # Final summary
        print("\n" + "="*80)
        print("ANNOTATION COMPLETE!")
        print("="*80)
        print(f"Processed: {self.current_idx}/{len(self.data)} chunks")

        # Statistics
        annotated_count = sum(1 for chunk in self.data if chunk.get('annotated_label') is not None)
        edu_count = sum(1 for chunk in self.data if chunk.get('annotated_label') == 1)
        non_edu_count = sum(1 for chunk in self.data if chunk.get('annotated_label') == 0)
        token_count = sum(len(chunk.get('special_tokens', [])) for chunk in self.data)

        print(f"\nAnnotated chunks: {annotated_count}")
        print(f"  Educational: {edu_count}")
        print(f"  Non-educational: {non_edu_count}")
        print(f"  Special tokens added: {token_count}")

        print(f"\nAnnotated data saved to: annotated_training_data.json")
        print(f"\nNext step: Run 'python prepare_colab_dataset.py' to prepare for training")

if __name__ == '__main__':
    interface = AnnotationInterface()
    interface.run()
