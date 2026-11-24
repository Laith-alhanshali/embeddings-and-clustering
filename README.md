# Embeddings & Clustering

Text and image embeddings, dimensionality reduction and clustering, built for the
**Data Mining** course at Poznań University of Technology (Erasmus+, 2026).

The project covers the full pipeline: collecting a dataset from scratch, turning it
into vectors, reducing and visualising those vectors, and clustering them.

## What's inside

| Notebook | What it does |
|---|---|
| `dm_project.ipynb` | Scrapes laptop listings from x-kom.pl, embeds them, classifies and visualises them, then clusters them |
| `embeddings_benchmark.ipynb` | Compares image-embedding methods on a downstream classification task |

### 1. Custom dataset (`dm_project.ipynb`)
- Scraped laptop listings (title, price, specifications) from x-kom.pl with **Selenium** and saved them with **pandas**.
- Embedded each listing with **Sentence Transformers** (`all-MiniLM-L6-v2`, 384 dimensions).

### 2. Classifier performance vs. latent dimensions
- Used the laptop brand as the label (top 5 brands) and trained a **Random Forest** on PCA-reduced embeddings.
- Measured macro F1 as the number of PCA components grew (2 to 300, capped by the sample count).

### 3. 2D visualisation
- Projected the embeddings to 2D with **PCA** and **t-SNE** and compared the two.

### 4. Clustering
- Chose k with the elbow method, silhouette score and Davies-Bouldin index.
- Ran **K-Means** (k picked by silhouette) and **DBSCAN** (on PCA-reduced vectors, with an estimated `eps`).
- Clusters came out as product lines, e.g. HP Victus, Lenovo Legion 5 and MacBook Pro M5 listings each ended up together. DBSCAN flagged unusual listings as noise.

### 5. Embedding benchmark (`embeddings_benchmark.ipynb`)
Image embeddings compared on **Caltech101** (20 classes, 400 images, 80/20 stratified split) using an **XGBoost** classifier and macro-averaged metrics:

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Custom (16x16 pixel grid) | 0.421 | 0.375 | 0.373 |
| ResNet-18 (pretrained, ImageNet) | 0.908 | 0.888 | **0.885** |
| Colour histogram (8 bins) | 0.340 | 0.338 | 0.329 |
| Colour histogram (32 bins) | 0.261 | 0.313 | 0.282 |
| Colour histogram (64 bins) | 0.311 | 0.313 | 0.289 |

**Takeaway:** features from a pretrained ResNet-18 beat hand-built features by a wide margin. More histogram bins did not help.
With only 400 images these numbers are indicative, not definitive.

## Tech stack
Python, pandas, NumPy, scikit-learn, XGBoost, Sentence Transformers, PyTorch / timm (pretrained ResNet-18), UMAP, Selenium, Plotly, Matplotlib.

## Run it

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

- The scraped dataset is included as `xkom_laptops_multi_page.csv` (public product listings, scraped from x-kom.pl in June 2026 for coursework). To re-scrape, run the scraping section of `dm_project.ipynb`; the site layout may have changed since, so the selectors might need updating.
- Caltech101 and the Sentence Transformers model download automatically on first run.

## Notes
- The cluster-evaluation and plotting helpers were adapted from the course lab material.
- Built as coursework, so the scale is small by design.
