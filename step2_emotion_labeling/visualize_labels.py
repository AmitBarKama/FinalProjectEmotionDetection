"""
Visualize a grid of sampled images from the FFHQ dataset along with their predicted emotion labels and confidence scores.
Generates a general sample grid, as well as a dedicated grid for each emotion category.

Usage:
    python -m step2_emotion_labeling.visualize_labels
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EMOTION_LABELS_FILTERED_FILE, FFHQ_DIR, EVALUATION_DIR, EMOTION_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def generate_grid(df_subset, title, save_path):
    """Helper to generate a 4x4 grid of images from a given DataFrame subset."""
    n_samples = min(16, len(df_subset))
    if n_samples == 0:
        logger.warning("No samples available to plot for: %s", title)
        return False
        
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    # Sample randomly
    sample_df = df_subset.sample(n=n_samples, random_state=42)
    
    for idx in range(16):
        ax = axes[idx]
        if idx < len(sample_df):
            row = sample_df.iloc[idx]
            filename = row.get("filename")
            if not filename:
                filename = f"{int(row['image_id']):05d}.png"
                
            img_path = FFHQ_DIR / filename
            
            if img_path.exists():
                img = Image.open(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center")
            
            emotion = row["predicted_emotion"]
            confidence = row["confidence"]
            ax.set_title(f"{emotion} ({confidence*100:.1f}%)", fontsize=10, 
                         color="green" if confidence > 0.8 else "orange")
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            
        ax.axis("off")
        
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved visualization to %s", save_path)
    return True

def main():
    if not EMOTION_LABELS_FILTERED_FILE.exists():
        logger.error("Filtered labels file not found: %s", EMOTION_LABELS_FILTERED_FILE)
        sys.exit(1)

    # Load the filtered labels
    df = pd.read_parquet(EMOTION_LABELS_FILTERED_FILE)
    logger.info("Loaded %d filtered labels.", len(df))

    # 1. Generate general sample grid
    general_path = EVALUATION_DIR / "sample_labeled_faces.png"
    generate_grid(df, "Overall Sampled Labeled Faces", general_path)

    # 2. Generate a grid for each individual emotion
    print("\n========================================================")
    print("Generating sample grids for each emotion class:")
    print("========================================================")
    
    for emotion in EMOTION_LABELS:
        df_emotion = df[df["predicted_emotion"] == emotion]
        output_path = EVALUATION_DIR / f"sample_{emotion}.png"
        success = generate_grid(df_emotion, f"Sampled Labeled Faces: {emotion}", output_path)
        if success:
            print(f"- {emotion}: {output_path}")
        else:
            print(f"- {emotion}: (No samples found)")
            
    print("========================================================\n")

if __name__ == "__main__":
    main()
