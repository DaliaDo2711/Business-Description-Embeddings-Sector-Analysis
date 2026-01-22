"""
Business Description Embeddings & Sector Analysis
GitHub-ready script converted from your Google Colab workflow.

What this script does:
1) Load companyInfo.csv
2) Data cleaning:
   - Fill missing Sector/Industry
   - Detect ETF-like descriptions and label as ETF
   - Label remaining missing as Unknown
   - Save companyInfo_cleaned.csv
3) Embedding explanation:
   - Convert a 768-d embedding to a 16x16x3 RGB icon and save it
4) UMAP:
   - Run 2D UMAP (Sector + Industry plots)
   - Run 3D UMAP (Sector plot)
   - Compare UMAP parameter settings side-by-side (matplotlib)
5) Outliers:
   - Find extreme UMAP points
6) Cross-sector similarity highlighting plot (Plotly)

How to run:
python business_description_embeddings_full.py

Dependencies:
pandas, numpy, matplotlib, umap-learn, plotly
"""

import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import umap
import plotly.express as px


# -----------------------------
# Helpers: column detection
# -----------------------------
def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Missing required column. Looked for one of: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def detect_schema(df: pd.DataFrame) -> dict:
    """
    Try to adapt to small column-name differences across exports.
    """
    schema = {
        "ticker": _pick_col(df, ["Ticker", "ticker", "Symbol", "symbol"], required=False),
        "sector": _pick_col(df, ["Sector", "sector"], required=True),
        "industry": _pick_col(df, ["Industry", "industry"], required=True),
        "description": _pick_col(df, ["Description", "description", "Business Description", "business_description"], required=False),
        "embeddings": _pick_col(df, ["embeddings", "embedding", "Embeddings", "Embedding"], required=True),
    }

    # If there is no description column, we can still run the pipeline,
    # but ETF detection will be limited (only Unknown fill, no ETF keyword logic).
    return schema


# -----------------------------
# Part 1: Data cleaning
# -----------------------------
def clean_data(
    df: pd.DataFrame,
    sector_col: str,
    industry_col: str,
    description_col: str | None,
) -> pd.DataFrame:
    df = df.copy()

    # Identify missing Sector & Industry (your original logic: both missing)
    missing_both = df[sector_col].isna() & df[industry_col].isna()

    # ETF detection (only if description exists)
    if description_col is not None:
        etf_keywords = [
            "etf", "fund", "index", "investment objective", "seeks to track", "trust",
            "exchange-traded", "exchange traded"
        ]
        desc_lower = df[description_col].fillna("").astype(str).str.lower()

        df["is_etf"] = desc_lower.apply(lambda x: any(k in x for k in etf_keywords))

        # Assign ETF to Sector and Industry for detected ETFs
        df.loc[missing_both & df["is_etf"], sector_col] = "ETF"
        df.loc[missing_both & df["is_etf"], industry_col] = "ETF"

        # Label remaining unknowns
        df.loc[missing_both & ~df["is_etf"], sector_col] = "Unknown"
        df.loc[missing_both & ~df["is_etf"], industry_col] = "Unknown"
    else:
        # No description column available: just label missing as Unknown
        df.loc[missing_both, sector_col] = "Unknown"
        df.loc[missing_both, industry_col] = "Unknown"

    # Standardize formatting
    df[sector_col] = df[sector_col].astype(str).str.strip().str.title()
    df[industry_col] = df[industry_col].astype(str).str.strip().str.title()

    # Optional: drop helper column
    if "is_etf" in df.columns:
        # keep it if you want to show your ETF detection; otherwise drop
        pass

    return df


# -----------------------------
# Part 2: Embedding parsing
# -----------------------------
def fix_embedding_string(embedding_str: str) -> str:
    """
    Fix badly formatted embedding strings from CSV exports.
    Your Colab version:
    - remove newlines and extra spaces
    - insert commas between numbers when missing
    """
    if embedding_str is None:
        return "[]"
    s = str(embedding_str).strip()
    s = re.sub(r"\s+", " ", s)  # collapse whitespace/newlines

    # If it already looks like a Python list with commas, leave it.
    # Otherwise, try inserting commas between adjacent numbers.
    # This regex inserts commas where there's a space between numeric tokens.
    s = re.sub(r"(?<=[0-9]) (?=-?[0-9])", ",", s)
    return s


def parse_embedding(embedding_value) -> np.ndarray:
    """
    Accepts:
    - stringified list
    - actual list
    Returns np.ndarray of shape (768,)
    """
    if isinstance(embedding_value, (list, tuple, np.ndarray)):
        arr = np.array(embedding_value, dtype=float)
        return arr

    fixed = fix_embedding_string(embedding_value)
    try:
        arr = np.array(ast.literal_eval(fixed), dtype=float)
    except Exception as e:
        raise ValueError(f"Could not parse embedding. Example value starts with: {str(embedding_value)[:80]}") from e
    return arr


def build_embedding_matrix(df: pd.DataFrame, emb_col: str) -> np.ndarray:
    embs = df[emb_col].apply(parse_embedding)
    mat = np.vstack(embs.values)
    return mat


# -----------------------------
# Part 3: Embedding RGB icon
# -----------------------------
def save_embedding_rgb_icon(
    df: pd.DataFrame,
    emb_col: str,
    out_path: str = "embedding_rgb_icon.png",
    row_index: int = 0,
) -> None:
    """
    Visualize 768 dims as 16x16x3 RGB icon, save to file.
    """
    emb = parse_embedding(df.loc[row_index, emb_col])
    if emb.size != 768:
        raise ValueError(f"Expected 768-d embedding. Got {emb.size} dims.")

    reshaped = emb.reshape((16, 16, 3))

    # normalize to [0, 255]
    ptp = np.ptp(reshaped)
    if ptp == 0:
        rgb = np.zeros_like(reshaped, dtype=np.uint8)
    else:
        rgb = 255 * (reshaped - reshaped.min()) / ptp
        rgb = rgb.astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.axis("off")
    plt.title("Visualized Embedding (16x16x3)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Part 4: UMAP utilities
# -----------------------------
def run_umap(embedding_matrix: np.ndarray, n_neighbors: int, min_dist: float, n_components: int, random_state: int = 42) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )
    return reducer.fit_transform(embedding_matrix)


def plot_umap_params_side_by_side(
    coords_list: list[tuple[np.ndarray, str]],
    sector_codes: np.ndarray,
    out_path: str = "umap_param_comparison.png",
) -> None:
    """
    Matplotlib scatter side-by-side for parameter impact.
    """
    fig, axs = plt.subplots(1, len(coords_list), figsize=(18, 5))
    if len(coords_list) == 1:
        axs = [axs]

    for i, (coords, label) in enumerate(coords_list):
        axs[i].scatter(
            coords[:, 0],
            coords[:, 1],
            c=sector_codes,
            cmap="tab20",
            s=10,
            alpha=0.8
        )
        axs[i].set_title(label)
        axs[i].axis("off")

    plt.suptitle("UMAP Visualization: Parameter Impact", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plotly_umap_2d(df: pd.DataFrame, xcol: str, ycol: str, color_col: str, hover_col: str | None, title: str):
    fig = px.scatter(
        df,
        x=xcol,
        y=ycol,
        color=color_col,
        hover_name=hover_col if hover_col else None,
        title=title,
        height=650
    )
    fig.show()


def plotly_umap_3d(df: pd.DataFrame, xcol: str, ycol: str, zcol: str, color_col: str, hover_col: str | None, title: str):
    fig = px.scatter_3d(
        df,
        x=xcol,
        y=ycol,
        z=zcol,
        color=color_col,
        hover_name=hover_col if hover_col else None,
        title=title,
        opacity=0.7,
        height=750
    )
    fig.show()


# -----------------------------
# Part 5: Outliers
# -----------------------------
def find_outliers(df: pd.DataFrame, xcol: str, ycol: str, threshold: float = 10.0) -> pd.DataFrame:
    out = df[(df[xcol] > threshold) | (df[xcol] < -threshold) | (df[ycol] > threshold) | (df[ycol] < -threshold)]
    return out


# -----------------------------
# Part 6: Cross-sector highlighting
# -----------------------------
def add_cross_sector_groups(df: pd.DataFrame, ticker_col: str | None) -> pd.DataFrame:
    """
    Your group logic, if ticker exists.
    """
    if not ticker_col:
        return df

    fintech_like = ["AFRM", "ALLY", "ADS"]
    insurance_group = ["AFL", "AIG", "AIZ", "AON", "ACGL", "AJG", "ALL", "AFG"]
    etf_funds = ["ACES", "AGG", "ADS", "AEL", "AGO", "AMG", "APAM"]

    def label_cluster(ticker: str) -> str:
        if ticker in fintech_like:
            return "Fintech-like Financials"
        if ticker in insurance_group:
            return "Insurance Group"
        if ticker in etf_funds:
            return "ETF / Investment Fund"
        return "Other"

    df = df.copy()
    df["CrossSectorGroup"] = df[ticker_col].astype(str).apply(label_cluster)
    return df


def main():
    # -----------------------------
    # Load raw CSV
    # -----------------------------
    raw_path = "companyInfo.csv"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Could not find {raw_path} in the current folder. "
            f"Place companyInfo.csv in the same folder as this script."
        )

    df_raw = pd.read_csv(raw_path)
    schema = detect_schema(df_raw)

    ticker_col = schema["ticker"]
    sector_col = schema["sector"]
    industry_col = schema["industry"]
    description_col = schema["description"]
    emb_col = schema["embeddings"]

    # -----------------------------
    # Data Cleaning
    # -----------------------------
    df_clean = clean_data(df_raw, sector_col=sector_col, industry_col=industry_col, description_col=description_col)
    df_clean.to_csv("companyInfo_cleaned.csv", index=False)

    print("\nSector distribution (after cleaning):")
    print(df_clean[sector_col].value_counts().head(25))

    # -----------------------------
    # Parse embeddings into matrix
    # -----------------------------
    embedding_matrix = build_embedding_matrix(df_clean, emb_col=emb_col)
    print("\nEmbedding matrix shape:", embedding_matrix.shape)

    # -----------------------------
    # Embedding RGB icon (16x16x3)
    # -----------------------------
    save_embedding_rgb_icon(df_clean, emb_col=emb_col, out_path="embedding_rgb_icon.png", row_index=0)
    print("\nSaved embedding icon to: embedding_rgb_icon.png")

    # -----------------------------
    # UMAP parameter comparison (side-by-side)
    # -----------------------------
    sector_codes = pd.Categorical(df_clean[sector_col]).codes

    umap_settings = [
        {"n_neighbors": 10, "min_dist": 0.01, "label": "Local (n=10, dist=0.01)"},
        {"n_neighbors": 40, "min_dist": 0.01, "label": "Balanced (n=40, dist=0.01)"},
        {"n_neighbors": 80, "min_dist": 0.5, "label": "Global (n=80, dist=0.5)"},
    ]

    coords_list = []
    for s in umap_settings:
        coords = run_umap(embedding_matrix, s["n_neighbors"], s["min_dist"], n_components=2)
        coords_list.append((coords, s["label"]))

    plot_umap_params_side_by_side(coords_list, sector_codes, out_path="umap_param_comparison.png")
    print("Saved UMAP parameter comparison to: umap_param_comparison.png")

    # -----------------------------
    # Main 2D UMAP (balanced)
    # -----------------------------
    coords_2d = run_umap(embedding_matrix, n_neighbors=40, min_dist=0.01, n_components=2)

    df_clean = df_clean.copy()
    df_clean["UMAP_2D_1"] = coords_2d[:, 0]
    df_clean["UMAP_2D_2"] = coords_2d[:, 1]

    plotly_umap_2d(
        df_clean,
        xcol="UMAP_2D_1",
        ycol="UMAP_2D_2",
        color_col=sector_col,
        hover_col=ticker_col,
        title="UMAP Projection of Company Embeddings (Sector) | n_neighbors=40, min_dist=0.01"
    )

    plotly_umap_2d(
        df_clean,
        xcol="UMAP_2D_1",
        ycol="UMAP_2D_2",
        color_col=industry_col,
        hover_col=ticker_col,
        title="UMAP Projection of Company Embeddings (Industry) | n_neighbors=40, min_dist=0.01"
    )

    # -----------------------------
    # 3D UMAP
    # -----------------------------
    coords_3d = run_umap(embedding_matrix, n_neighbors=40, min_dist=0.01, n_components=3)
    df_clean["UMAP_3D_1"] = coords_3d[:, 0]
    df_clean["UMAP_3D_2"] = coords_3d[:, 1]
    df_clean["UMAP_3D_3"] = coords_3d[:, 2]

    plotly_umap_3d(
        df_clean,
        xcol="UMAP_3D_1",
        ycol="UMAP_3D_2",
        zcol="UMAP_3D_3",
        color_col=sector_col,
        hover_col=ticker_col,
        title="3D UMAP Projection of Company Embeddings (Colored by Sector)"
    )

    # -----------------------------
    # Outlier detection
    # -----------------------------
    outliers = find_outliers(df_clean, xcol="UMAP_2D_1", ycol="UMAP_2D_2", threshold=10.0)

    cols_to_show = [c for c in [ticker_col, sector_col, industry_col, description_col] if c]
    if len(outliers) > 0 and cols_to_show:
        print("\nPotential outliers (extreme UMAP coords):")
        print(outliers[cols_to_show].head(30).to_string(index=False))
    else:
        print("\nNo outliers found with the current threshold, or missing display columns.")

    # -----------------------------
    # Cross-sector similarities plot
    # -----------------------------
    df_cross = add_cross_sector_groups(df_clean, ticker_col=ticker_col)

    # Rename for consistency with your Colab naming
    df_cross.rename(columns={"UMAP_2D_1": "UMAP_X", "UMAP_2D_2": "UMAP_Y"}, inplace=True)

    if ticker_col:
        fig = px.scatter(
            df_cross,
            x="UMAP_X",
            y="UMAP_Y",
            color="CrossSectorGroup",
            symbol="CrossSectorGroup",
            hover_name=ticker_col,
            title="Cross-Sector Similarities in UMAP Space",
            width=950,
            height=700
        )
        fig.update_traces(marker=dict(size=10))
        fig.show()
    else:
        print("\nTicker column not found, skipping cross-sector plot.")


if __name__ == "__main__":
    main()
