# sentiment_analysis_social_media.py
# -----------------------------------------------------------
# Analyze & visualize sentiment patterns in social media data
# using a CSV (auto-detects likely text/label columns).
#
# Saves plots into ./outputs and prints key summaries.
# -----------------------------------------------------------

import os
import re
import sys
import string
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------- CONFIG ----------------------
CSV_PATH = "/mnt/data/twitter_training.csv"  # change if needed

# If auto-detect fails, set these manually:
TEXT_COL = None    # e.g., "text"
LABEL_COL = None   # e.g., "sentiment"

# Time column candidates (if present, we’ll use for trend plots)
TIME_CANDIDATES = ["created_at", "date", "datetime", "timestamp", "time"]
# ----------------------------------------------------

# --------------- UTILITIES --------------------------
def ensure_outputs_dir(path="outputs"):
    os.makedirs(path, exist_ok=True)
    return path

def head_print(df, n=5):
    print("\n==== HEAD ====")
    print(df.head(n))
    print("\n==== INFO ====")
    print(df.info())

def detect_columns(df, text_col=None, label_col=None):
    cols_lower = {c.lower(): c for c in df.columns}
    text_candidates = ["text", "tweet", "content", "message", "body"]
    label_candidates = ["sentiment", "label", "target", "polarity", "class"]

    if text_col is None:
        for c in text_candidates:
            if c in cols_lower:
                text_col = cols_lower[c]
                break

    if label_col is None:
        for c in label_candidates:
            if c in cols_lower:
                label_col = cols_lower[c]
                break

    # If still None, try heuristics: longest average length -> text, lowest cardinality -> label
    if text_col is None:
        avg_lengths = {c: df[c].astype(str).str.len().mean() for c in df.columns if df[c].dtype == "object"}
        text_col = max(avg_lengths, key=avg_lengths.get) if avg_lengths else df.columns[0]

    if label_col is None:
        nunique = df.nunique().sort_values()
        # pick a low-cardinality (2-10 unique) non-text-like column
        for c in nunique.index:
            if 2 <= nunique[c] <= 10 and c != text_col:
                label_col = c
                break
        if label_col is None:
            # fallback: second column
            label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    return text_col, label_col

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
MULTISPACE_RE = re.compile(r"\s+")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    # keep hashtag word without '#'
    s = re.sub(r"#([A-Za-z0-9_]+)", r"\1", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s

def extract_hashtags(s: str):
    if not isinstance(s, str):
        s = str(s)
    return [h.lower() for h in HASHTAG_RE.findall(s)]

def maybe_parse_datetime(df):
    for c in TIME_CANDIDATES:
        if c in df.columns:
            try:
                out = pd.to_datetime(df[c], errors="coerce", utc=True)
                if out.notna().sum() > 0:
                    return c, out
            except Exception:
                pass
    return None, None

def plot_bar(counts, title, xlabel, ylabel, save_path, rotate=0, top_n=None):
    ensure_outputs_dir(os.path.dirname(save_path) or ".")
    labels, values = zip(*counts) if isinstance(counts, list) else (counts.index, counts.values)
    if top_n is not None and isinstance(counts, list):
        labels = labels[:top_n]
        values = values[:top_n]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if rotate:
        plt.xticks(rotation=rotate, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_line(df, x, y, hue=None, title="", save_path="outputs/line.png"):
    ensure_outputs_dir(os.path.dirname(save_path) or ".")
    plt.figure(figsize=(10, 6))
    if hue:
        for k, grp in df.groupby(hue):
            plt.plot(grp[x], grp[y], label=str(k))
        plt.legend()
    else:
        plt.plot(df[x], df[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_confusion_matrix(cm, classes, title="Confusion Matrix", save_path="outputs/confusion_matrix.png"):
    ensure_outputs_dir(os.path.dirname(save_path) or ".")
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

# --------------- LOAD DATA --------------------------
print(f"Loading CSV from: {"C:/Users/Madhavesh/Downloads/twitter_training.csv"}")
df = pd.read_csv("C:/Users/Madhavesh/Downloads/twitter_training.csv", encoding="utf-8", engine="python", on_bad_lines="skip")
print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
print("Columns:", list(df.columns))

head_print(df)

# --------------- COLUMN DETECTION -------------------
TEXT_COL, LABEL_COL = detect_columns(df, TEXT_COL, LABEL_COL)
print(f"\nUsing TEXT column:  {TEXT_COL}")
print(f"Using LABEL column: {LABEL_COL}")

# Drop rows with missing core fields
df = df.dropna(subset=[TEXT_COL, LABEL_COL]).copy()

# Normalize label values (strip/ lowercase strings)
if df[LABEL_COL].dtype == "object":
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    # Try common sentiment normalizations
    mapping = {
        "pos": "positive", "positive": "positive", "1": "positive",
        "neg": "negative", "negative": "negative", "-1": "negative",
        "neu": "neutral", "neutral": "neutral", "0": "neutral"
    }
    df[LABEL_COL] = df[LABEL_COL].str.lower().map(lambda x: mapping.get(x, x))
else:
    # numeric labels -> map if clearly 3-class or 2-class
    uniq = sorted(df[LABEL_COL].dropna().unique())
    if set(uniq).issubset({-1,0,1}):
        num_map = {-1:"negative", 0:"neutral", 1:"positive"}
        df[LABEL_COL] = df[LABEL_COL].map(num_map)
    elif set(uniq).issubset({0,1}):
        num_map = {0:"negative", 1:"positive"}
        df[LABEL_COL] = df[LABEL_COL].map(num_map)
    else:
        df[LABEL_COL] = df[LABEL_COL].astype(str)

# --------------- BASIC CLEANING ---------------------
df["text_clean"] = df[TEXT_COL].apply(clean_text)

# --------------- SUMMARY & DISTRIBUTION -------------
print("\n==== Sentiment distribution ====")
sent_counts = df[LABEL_COL].value_counts()
print(sent_counts)

outdir = ensure_outputs_dir("outputs")

plot_bar(
    counts=sent_counts.sort_values(ascending=False),
    title="Sentiment Distribution",
    xlabel="Sentiment",
    ylabel="Count",
    save_path=f"{outdir}/sentiment_distribution.png",
    rotate=0
)

# --------------- TIME TRENDS (if timestamps exist) --
time_col, parsed = maybe_parse_datetime(df)
if time_col is not None:
    print(f"\nUsing time column for trends: {time_col}")
    df["_dt"] = parsed
    # drop NaT rows
    dft = df.dropna(subset=["_dt"]).copy()
    if not dft.empty:
        # resample weekly to smooth
        dft = dft.set_index("_dt")
        trend = (
            dft.groupby(LABEL_COL)
               .resample("W")
               .size()
               .reset_index(name="count")
        )
        # overall trend
        overall = (
            dft.resample("W")
               .size()
               .reset_index(name="count")
        )

        plot_line(
            overall, x=time_col, y="count", hue=None,
            title="Overall Volume Over Time (Weekly)",
            save_path=f"{outdir}/overall_volume_trend.png"
        )

        # rename for plotting function
        trend = trend.rename(columns={"_dt": time_col})
        plot_line(
            trend, x=time_col, y="count", hue=LABEL_COL,
            title="Sentiment Volume Over Time (Weekly)",
            save_path=f"{outdir}/sentiment_volume_trend.png"
        )
else:
    print("\nNo time-like column found. Skipping trend charts.")

# --------------- HASHTAGS ---------------------------
print("\nExtracting hashtags...")
df["hashtags"] = df[TEXT_COL].astype(str).apply(extract_hashtags)
all_hashtags = [h for hs in df["hashtags"] for h in hs]
ht_counts = Counter(all_hashtags).most_common(20)
print("Top hashtags:", ht_counts[:10])

if ht_counts:
    plot_bar(
        counts=ht_counts,
        title="Top 20 Hashtags (All Sentiments)",
        xlabel="Hashtag",
        ylabel="Frequency",
        save_path=f"{outdir}/top_hashtags.png",
        rotate=45,
        top_n=20
    )

    # Top hashtags per sentiment
    for s in df[LABEL_COL].dropna().unique():
        subset = df[df[LABEL_COL] == s]
        hts = [h for hs in subset["hashtags"] for h in hs]
        counts = Counter(hts).most_common(15)
        if counts:
            plot_bar(
                counts=counts,
                title=f"Top Hashtags — {s}",
                xlabel="Hashtag",
                ylabel="Frequency",
                save_path=f"{outdir}/top_hashtags_{s}.png",
                rotate=45,
                top_n=15
            )

# --------------- N-GRAMS (uni-, bi-gram) -----------
def top_ngrams(corpus, ngram_range=(1,1), top_k=20, min_df=2):
    vec = CountVectorizer(ngram_range=ngram_range, min_df=min_df, stop_words="english")
    X = vec.fit_transform(corpus)
    sums = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    order = np.argsort(-sums)
    return list(zip(terms[order][:top_k], sums[order][:top_k]))

print("\nTop unigrams (overall):")
uni = top_ngrams(df["text_clean"], (1,1), top_k=25, min_df=2)
print(uni[:10])
plot_bar(
    counts=uni,
    title="Top 25 Unigrams (All)",
    xlabel="Term",
    ylabel="Count",
    save_path=f"{outdir}/top_unigrams.png",
    rotate=45
)

print("\nTop bigrams (overall):")
bi = top_ngrams(df["text_clean"], (2,2), top_k=25, min_df=2)
print(bi[:10])
plot_bar(
    counts=bi,
    title="Top 25 Bigrams (All)",
    xlabel="Term",
    ylabel="Count",
    save_path=f"{outdir}/top_bigrams.png",
    rotate=45
)

# Per-sentiment unigrams
for s in df[LABEL_COL].dropna().unique():
    sub = df[df[LABEL_COL] == s]
    if len(sub) >= 20:
        tg = top_ngrams(sub["text_clean"], (1,1), top_k=20, min_df=2)
        if tg:
            plot_bar(
                counts=tg,
                title=f"Top Unigrams — {s}",
                xlabel="Term",
                ylabel="Count",
                save_path=f"{outdir}/top_unigrams_{s}.png",
                rotate=45
            )

# --------------- SIMPLE MODEL (TF-IDF + LR) --------
# Filter to most common labels (helps if noisy)
label_counts = df[LABEL_COL].value_counts()
keep_labels = label_counts[label_counts >= max(10, int(0.01*len(df)))].index.tolist()
df_model = df[df[LABEL_COL].isin(keep_labels)].copy()

print("\nLabels used for modeling:", keep_labels)

X_train, X_test, y_train, y_test = train_test_split(
    df_model["text_clean"], df_model[LABEL_COL], test_size=0.2, random_state=42, stratify=df_model[LABEL_COL]
)

tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9, stop_words="english")
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

clf = LogisticRegression(max_iter=200, n_jobs=None)
clf.fit(Xtr, y_train)
y_pred = clf.predict(Xte)

print("\n==== Classification Report ====")
print(classification_report(y_test, y_pred, digits=3))

cm = confusion_matrix(y_test, y_pred, labels=keep_labels)
plot_confusion_matrix(
    cm, classes=keep_labels,
    title="Confusion Matrix (TF-IDF + Logistic Regression)",
    save_path=f"{outdir}/confusion_matrix.png"
)

# --------------- SAVE SAMPLE TABLES -----------------
# Export top hashtags and ngrams as CSVs for reference
pd.DataFrame(ht_counts, columns=["hashtag","count"]).to_csv(f"{outdir}/top_hashtags_all.csv", index=False)
pd.DataFrame(uni, columns=["term","count"]).to_csv(f"{outdir}/top_unigrams_all.csv", index=False)
pd.DataFrame(bi, columns=["term","count"]).to_csv(f"{outdir}/top_bigrams_all.csv", index=False)

print("\nAll plots saved to ./outputs. Done!")