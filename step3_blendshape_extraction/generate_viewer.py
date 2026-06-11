"""
Generate an interactive HTML-based dataset viewer to browse images,
their predicted emotion labels, and their extracted blendshape values.

Usage:
    python -m step3_blendshape_extraction.generate_viewer
"""

import sys
import json
import logging
from pathlib import Path
import pandas as pd

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COMBINED_DATASET_FILE, EMOTION_LABELS, BLENDSHAPE_NAMES, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    if not COMBINED_DATASET_FILE.exists():
        logger.error("Combined dataset not found: %s. Run build_dataset first.", COMBINED_DATASET_FILE)
        sys.exit(1)

    # Load dataset
    df = pd.read_parquet(COMBINED_DATASET_FILE)
    logger.info("Loaded dataset with %d samples.", len(df))

    # To keep the HTML file responsive and fast, we'll sample up to 50 images per emotion
    sampled_dfs = []
    for emotion in EMOTION_LABELS:
        sub_df = df[df["predicted_emotion"] == emotion]
        n_samples = min(50, len(sub_df))
        if n_samples > 0:
            sampled_dfs.append(sub_df.sample(n=n_samples, random_state=42))
            
    sample_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info("Sampled %d images for the interactive viewer.", len(sample_df))

    # Convert to JSON structure
    records = []
    for _, row in sample_df.iterrows():
        image_id = row["image_id"]
        # Format the ID to match the image filenames (e.g. 00123.png)
        try:
            filename = f"{int(image_id):05d}.png"
        except ValueError:
            filename = f"{image_id}.png"
            
        record = {
            "id": image_id,
            "filename": filename,
            "predicted_emotion": row["predicted_emotion"],
            "confidence": float(row["confidence"]),
            "probabilities": {emo: float(row[f"prob_{emo}"]) for emo in EMOTION_LABELS},
            "blendshapes": {bs: float(row[bs]) for bs in BLENDSHAPE_NAMES if bs in row}
        }
        records.append(record)

    # Generate HTML content with embedded CSS/JS and data
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion & Blendshape Dataset Viewer</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0e17;
            --bg-secondary: #121824;
            --bg-tertiary: #1b2336;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.15);
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --border: #2e3a52;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: 'Outfit', sans-serif;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }}
        /* Sidebar Styles */
        .sidebar {{
            width: 320px;
            background-color: var(--bg-secondary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            height: 100%;
        }}
        .sidebar-header {{
            padding: 20px;
            border-bottom: 1px solid var(--border);
        }}
        .sidebar-header h1 {{
            font-size: 1.25rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .filter-controls {{
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
        }}
        select, input {{
            width: 100%;
            background-color: var(--bg-primary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 8px 12px;
            border-radius: 6px;
            font-family: inherit;
            outline: none;
        }}
        select:focus, input:focus {{
            border-color: var(--accent);
        }}
        .image-list {{
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }}
        .image-item {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px;
            margin-bottom: 8px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            background-color: var(--bg-tertiary);
            border: 1px solid transparent;
        }}
        .image-item:hover {{
            background-color: rgba(59, 130, 246, 0.05);
            border-color: rgba(59, 130, 246, 0.2);
        }}
        .image-item.active {{
            background-color: var(--accent-glow);
            border-color: var(--accent);
        }}
        .image-item img {{
            width: 50px;
            height: 50px;
            border-radius: 6px;
            object-fit: cover;
            background-color: #000;
        }}
        .image-item-details {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .image-item-id {{
            font-size: 0.8rem;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
        }}
        .image-item-emotion {{
            font-weight: 600;
            font-size: 0.9rem;
        }}
        /* Main Content Styles */
        .main-content {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .viewer-header {{
            padding: 20px 40px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .viewer-body {{
            flex: 1;
            display: flex;
            padding: 40px;
            gap: 40px;
            overflow-y: auto;
        }}
        .card {{
            background-color: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
        }}
        .left-col {{
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 24px;
            max-width: 400px;
        }}
        .right-col {{
            flex: 2;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }}
        .image-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #000;
            border-radius: 8px;
            overflow: hidden;
            aspect-ratio: 1;
        }}
        .image-container img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        .metric-row {{
            margin-bottom: 12px;
        }}
        .metric-header {{
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            margin-bottom: 6px;
        }}
        .bar-container {{
            background-color: var(--bg-primary);
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
        }}
        .bar-fill {{
            height: 100%;
            background-color: var(--accent);
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        .bar-fill.active {{
            background-color: var(--success);
        }}
        /* Blendshapes Section */
        .blendshapes-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 12px;
        }}
        .bs-card {{
            background-color: var(--bg-tertiary);
            border: 1px solid var(--border);
            padding: 10px 14px;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            gap: 4px;
            transition: border-color 0.2s ease;
        }}
        .bs-card.active {{
            border-color: rgba(16, 185, 129, 0.4);
            background-color: rgba(16, 185, 129, 0.05);
        }}
        .bs-name {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-overflow: ellipsis;
            white-space: nowrap;
            overflow: hidden;
        }}
        .bs-val-row {{
            display: flex;
            align-items: baseline;
            justify-content: space-between;
        }}
        .bs-value {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 1.1rem;
        }}
        .bs-value.high {{
            color: var(--success);
        }}
        .bs-mini-bar {{
            height: 3px;
            background-color: rgba(255,255,255,0.05);
            border-radius: 1.5px;
            width: 100%;
            margin-top: 4px;
            overflow: hidden;
        }}
        .bs-mini-bar-fill {{
            height: 100%;
            background-color: var(--success);
        }}
        .legend {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            display: flex;
            gap: 12px;
            align-items: center;
        }}
        .dot {{
            height: 8px;
            width: 8px;
            border-radius: 50%;
            display: inline-block;
        }}
        .dot.active {{
            background-color: var(--success);
        }}
    </style>
</head>
<body>
    <div class="sidebar">
        <div class="sidebar-header">
            <h1>Dataset Inspector</h1>
            <div class="filter-controls">
                <select id="emotionFilter" onchange="filterImages()">
                    <option value="all">All Emotions</option>
                    {"".join(f'<option value="{emo}">{emo}</option>' for emo in EMOTION_LABELS)}
                </select>
            </div>
            <input type="text" id="searchBox" placeholder="Search Image ID..." oninput="filterImages()">
        </div>
        <div class="image-list" id="imageList"></div>
    </div>

    <div class="main-content">
        <div class="viewer-header">
            <h2 id="titleId">Select an image from the list</h2>
            <div class="legend">
                <span><span class="dot active"></span> Active Blendshape (&gt;0.10)</span>
            </div>
        </div>
        <div class="viewer-body" id="viewerBody" style="display: none;">
            <div class="left-col">
                <div class="card">
                    <div class="image-container">
                        <img id="detailImage" src="" alt="Face Image">
                    </div>
                </div>
                <div class="card">
                    <h3 style="margin-bottom: 16px;">Emotion Predictions</h3>
                    <div id="emotionList"></div>
                </div>
            </div>
            <div class="right-col">
                <div class="card" style="flex: 1; display: flex; flex-direction: column;">
                    <h3 style="margin-bottom: 16px;">Blendshapes (52 Coefficients)</h3>
                    <div class="blendshapes-grid" id="blendshapeGrid" style="overflow-y: auto; flex: 1;"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const dataset = {json.dumps(records)};
        let selectedIndex = 0;

        function init() {{
            renderList(dataset);
            if (dataset.length > 0) {{
                selectImage(0);
            }}
        }}

        function renderList(items) {{
            const list = document.getElementById('imageList');
            list.innerHTML = '';
            items.forEach((item, index) => {{
                const div = document.createElement('div');
                div.className = `image-item ${{index === selectedIndex ? 'active' : ''}}`;
                div.id = `item-${{item.id}}`;
                div.onclick = () => selectImage(dataset.indexOf(item));

                div.innerHTML = `
                    <img src="data/ffhq_thumbnails/${{item.filename}}" alt="thumb">
                    <div class="image-item-details">
                        <span class="image-item-emotion">${{item.predicted_emotion}}</span>
                        <span class="image-item-id">ID: ${{item.id}} (${{(item.confidence * 100).toFixed(0)}}%)</span>
                    </div>
                `;
                list.appendChild(div);
            }});
        }}

        function filterImages() {{
            const emotionFilter = document.getElementById('emotionFilter').value;
            const searchVal = document.getElementById('searchBox').value.trim();
            
            const filtered = dataset.filter(item => {{
                const matchesEmotion = emotionFilter === 'all' || item.predicted_emotion === emotionFilter;
                const matchesSearch = searchVal === '' || item.id.includes(searchVal);
                return matchesEmotion && matchesSearch;
            }});

            renderList(filtered);
            if (filtered.length > 0) {{
                const newIndex = dataset.indexOf(filtered[0]);
                selectImage(newIndex);
            }}
        }}

        function selectImage(index) {{
            if (index < 0 || index >= dataset.length) return;
            
            // De-select old item
            const oldActive = document.querySelector('.image-item.active');
            if (oldActive) oldActive.classList.remove('active');

            selectedIndex = index;
            const item = dataset[index];

            // Select new item
            const newActive = document.getElementById(`item-${{item.id}}`);
            if (newActive) newActive.classList.add('active');

            // Show viewer
            document.getElementById('viewerBody').style.display = 'flex';
            document.getElementById('titleId').innerText = `Image ID: ${{item.id}} (File: ${{item.filename}})`;

            // Update Image
            document.getElementById('detailImage').src = `data/ffhq_thumbnails/${{item.filename}}`;

            // Render Emotions
            const emotionList = document.getElementById('emotionList');
            emotionList.innerHTML = '';
            for (const [emo, val] of Object.entries(item.probabilities)) {{
                const isPrediction = emo === item.predicted_emotion;
                const row = document.createElement('div');
                row.className = 'metric-row';
                row.innerHTML = `
                    <div class="metric-header">
                        <span style="${{isPrediction ? 'font-weight: 800;' : ''}}">${{emo}} ${{isPrediction ? '★' : ''}}</span>
                        <span style="font-family: monospace;">${{(val * 100).toFixed(1)}}%</span>
                    </div>
                    <div class="bar-container">
                        <div class="bar-fill ${{isPrediction ? 'active' : ''}}" style="width: ${{val * 100}}%;"></div>
                    </div>
                `;
                emotionList.appendChild(row);
            }}

            // Render Blendshapes
            const grid = document.getElementById('blendshapeGrid');
            grid.innerHTML = '';
            
            // Sort blendshapes by value descending
            const sortedBs = Object.entries(item.blendshapes).sort((a, b) => b[1] - a[1]);

            sortedBs.forEach(([name, val]) => {{
                const card = document.createElement('div');
                const isActive = val > 0.10;
                card.className = `bs-card ${{isActive ? 'active' : ''}}`;
                
                card.innerHTML = `
                    <span class="bs-name" title="${{name}}">${{name}}</span>
                    <div class="bs-val-row">
                        <span class="bs-value ${{val > 0.10 ? 'high' : ''}}">${{val.toFixed(4)}}</span>
                    </div>
                    <div class="bs-mini-bar">
                        <div class="bs-mini-bar-fill" style="width: ${{val * 100}}%; background-color: ${{isActive ? 'var(--success)' : 'var(--text-secondary)'}}"></div>
                    </div>
                `;
                grid.appendChild(card);
            }});
        }}

        window.onload = init;
    </script>
</body>
</html>
"""

    # Save to root folder
    viewer_path = PROJECT_ROOT / "dataset_viewer.html"
    with open(viewer_path, "w") as f:
        f.write(html_template)

    logger.info("Saved HTML dataset viewer to %s", viewer_path)
    print(f"\n========================================================")
    print(f"Dataset Viewer generated successfully!")
    print(f"File location: {viewer_path}")
    print(f"Simply open this file in any web browser to explore!")
    print(f"========================================================\n")

if __name__ == "__main__":
    main()
