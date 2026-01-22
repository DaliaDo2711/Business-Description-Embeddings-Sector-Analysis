# Business Description Embeddings & Sector Analysis  
Data Visualization Project using Python
Goodle colab link: *https://colab.research.google.com/drive/1TaYNW8-jS2N2k3q7bMClZQaGZNKy2n0L?usp=sharing*

## Project Overview
This project analyzes and visualizes relationships between public companies based on their **business descriptions** using **pre-computed sentence transformer embeddings** and **UMAP** for dimensionality reduction.

Instead of relying on traditional financial metrics, this analysis focuses on the **semantic meaning of company descriptions**. By projecting 768-dimensional text embeddings into 2D and 3D spaces, the project reveals how companies cluster by sector and industry, highlights cross-sector similarities, and identifies meaningful outliers.

A key objective of this project is not only analysis, but also the ability to **clearly explain complex concepts** such as embeddings and UMAP through intuitive visualizations and a short video presentation.

---

## Project Motivation
Business descriptions contain rich information about what companies actually do, but this information is difficult to compare at scale using raw text.

By converting descriptions into numerical embeddings, companies can be compared based on **meaning rather than labels or keywords**. This approach enables exploration of:
- Natural sector and industry structure
- Overlapping business models across sectors
- Companies whose descriptions do not align with their assigned sector

---

## Dataset
The dataset contains information for publicly traded companies, including:
- Ticker symbols and company names
- Business descriptions
- Pre-computed **768-dimensional sentence transformer embeddings**
- Sector and industry classifications

## Data Preparation & Cleaning
### Data Preparation (one-time)
The original dataset was provided in Parquet format and downloaded from Google Drive.  
It was converted to CSV for portability and ease of reuse.

Original one-time preparation steps (run in Google Colab):

```python
!gdown 1A2m71ytdnGze4svxtrNRPCIkpXuveB61
#Downloading...
#From: https://drive.google.com/uc?id=1A2m71ytdnGze4svxtrNRPCIkpXuveB61
#To: /content/companyInfo.parquet
#100% 9.04M/9.04M [00:00<00:00, 77.0MB/s]

import pandas as pd
df0 = pd.read_parquet("companyInfo.parquet")

df0

df0.to_csv("companyInfo.csv", index=False)
```
### Data Cleaning
The dataset contained missing values, particularly for ETFs and special securities.

The following cleaning steps were performed:
- Identified 729 missing values in Sector and Industry
- Companies with ETF-like descriptions (e.g., “seeks to track index”, “fund objective”) and missing sector data were labeled as:
  - Sector = ETF  
  - Industry = ETF
- Remaining missing values were labeled as **Unknown** to preserve all observations

After preprocessing, the dataset contained **no missing values** and was ready for analysis.

---

## Explaining Sentence Embeddings
A sentence embedding is a numerical vector that captures the semantic meaning of a full sentence or document.

In this project:
- Each business description is represented as a **768-dimensional vector**
- These embeddings allow comparison of companies based on meaning, not exact wording

To make embeddings more intuitive, the 768 dimensions were reshaped into a **16×16×3 RGB representation**, creating a visual “fingerprint” for each company. This visualization demonstrates how dense vectors compress large amounts of textual meaning, following the idea that *a picture is worth a thousand words*.

---

## Explaining UMAP
UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique used to project high-dimensional data into lower dimensions while preserving meaningful structure.

UMAP was applied to reduce the 768-dimensional embedding space into **2D and 3D visualizations**.

Key hyperparameters used:
- **n_neighbors = 40**  
  Preserves global structure while still capturing local relationships
- **min_dist = 0.01**  
  Produces tight, well-separated clusters

To build intuition, the **woolly mammoth example** was used to demonstrate how adjusting these parameters changes cluster shape and separation. This analogy helps explain how similar effects appear in business sector visualizations.

---

## Analysis & Key Findings

### Embedding Space Visualization
UMAP projections were created with points colored by sector and industry. These visualizations show how companies group based on semantic similarity in their business descriptions.

### Sector-Level Clustering
Clear sector-level structure emerged:
- Technology, Healthcare, Financials, Energy, Utilities, and ETFs formed distinct clusters
- ETFs appeared as a compact, isolated cluster due to highly standardized fund-related language
- Companies labeled as Unknown were scattered, indicating weak or inconsistent description patterns

### Industry-Level Patterns
Meaningful sub-clusters appeared within sectors:
- Technology companies separated into:
  - Software and platform-focused firms
  - Hardware and semiconductor-focused firms
- Industries such as Oil & Gas and Healthcare showed especially tight clustering, reflecting consistent terminology

### Outlier Analysis
Several notable outliers were identified:
- Many outliers originated from the Energy sector, particularly Oil & Gas companies
- Companies with Unknown sector labels often appeared near known clusters, suggesting potential inferred classification
- Certain industrial firms, such as marine transportation companies, emerged as unique semantic outliers

### Cross-Sector Similarities
Semantic overlap was observed across sectors:
- ETF-related companies clustered together regardless of underlying sector due to shared fund language
- Insurance companies from Financials and Healthcare formed a consistent cluster driven by terms like “policy”, “coverage”, and “underwriting”
- Fintech-oriented financial firms appeared closer to Technology companies, reflecting convergence in modern, platform-based business language

---

## Visualizations Included
- Embedding explanation using 16×16×3 RGB icons
- UMAP parameter demonstration using the woolly mammoth example
- 2D UMAP visualizations colored by sector
- 2D UMAP visualizations colored by industry
- 3D UMAP visualization for improved cluster separation

---

## Video Presentation
A presentation accompanies this project and follows this structure:
1. Project motivation and overview
2. Explanation of sentence embeddings with visual aids
3. Explanation of UMAP and parameter impact
4. Key visualizations and analytical insights
5. Conclusion and synthesis

Presentation link: *https://drive.google.com/file/d/1Z1TozFnAcBfsV3UIGhqtdAGlY7N8nYS5/view?usp=sharing*
Powerpoint slide link: *https://docs.google.com/presentation/d/1b_ip4_Oe6yJpi9KbI3H7VlJTA0cAt5bb/edit?usp=drive_link&ouid=118255482125549420422&rtpof=true&sd=true*

---

## Key Takeaways
- Sentence embeddings enable comparison of companies based on semantic meaning
- UMAP reveals meaningful sector and industry structure when parameters are chosen carefully
- Some sectors exhibit strong linguistic cohesion, while others naturally overlap
- Embedding-based analysis highlights cross-sector convergence and identifies informative outliers

--- 

