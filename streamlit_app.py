# app.py
# To run this app, save the code as 'app.py' and run the command:
# streamlit run app.py
#
# Make sure you have the following directory structure:
# .
# ├── app.py
# ├── data/
# │   ├── google_play_sentiments_labeled.csv
# │   └── google_play_reviews_unlabeled.csv
# └── models/
#     ├── blibli_sentiment_model.h5
#     ├── blibli_tokenizer.pkl
#     └── blibli_history.csv  (Optional: for real training plots)
#     └── ... (files for other apps)
#
# DEPENDENCIES:
# pip install streamlit pandas numpy altair plotly-express wordcloud matplotlib scikit-learn seaborn Sastrawi tensorflow

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import re
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import difflib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="Google Play Review Sentiment Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Data & Model Loading ---

@st.cache_data
def load_data(uploaded_file=None):
    """
    Loads and preprocesses the main LABELED dataset.
    """
    if uploaded_file is not None:
        data_path = uploaded_file
    else:
        data_path = os.path.join("data", "google_play_sentiments_labeled.csv")
    
    if not os.path.exists(data_path) and uploaded_file is None:
        st.error(f"Error: Labeled data file not found at '{data_path}'. Please ensure the 'data' directory and the CSV file exist, or upload one in the Configuration page.")
        st.stop()
    try:
        df = pd.read_csv(data_path, low_memory=False)
        df.rename(columns={'at': 'review_date'}, inplace=True)
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        df['sentiment_label'] = pd.to_numeric(df['sentiment_label'], errors='coerce')
        df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
        df.dropna(subset=['score', 'sentiment_label', 'app_name', 'cleaned_text', 'original_review', 'review_date'], inplace=True)
        df['sentiment_label'] = df['sentiment_label'].astype(int)
        return df
    except Exception as e:
        st.error(f"An error occurred while loading the labeled data: {e}")
        st.stop()

@st.cache_data
def load_unlabeled_data(uploaded_file=None):
    """
    Loads the initial UNLABELED dataset.
    """
    if uploaded_file is not None:
        data_path = uploaded_file
    else:
        data_path = os.path.join("data", "google_play_reviews_unlabeled.csv")

    if not os.path.exists(data_path) and uploaded_file is None:
        st.error(f"Error: Unlabeled data file not found at '{data_path}'. Please ensure 'google_play_reviews_unlabeled.csv' is in the 'data' directory.")
        return pd.DataFrame() # Return empty dataframe to avoid stopping the app
    try:
        df = pd.read_csv(data_path)
        df.rename(columns={'at': 'review_date', 'content': 'original_review'}, inplace=True)
        df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
        df.dropna(subset=['original_review', 'score', 'review_date', 'userName'], inplace=True)
        return df
    except Exception as e:
        st.error(f"An error occurred while loading the unlabeled data: {e}")
        return pd.DataFrame()

@st.cache_resource
def get_stemmer():
    """Initializes and caches the Sastrawi stemmer."""
    factory = StemmerFactory()
    return factory.create_stemmer()

@st.cache_resource
def load_model_and_tokenizer(app_name):
    """Loads a specific model and tokenizer from the 'models' folder."""
    model_path = os.path.join("models", f"{app_name}_sentiment_model.h5")
    tokenizer_path = os.path.join("models", f"{app_name}_tokenizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        return None, None
    try:
        model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer for {app_name}: {e}")
        return None, None

@st.cache_data
def load_history(app_name):
    """Loads the training history for a specific app if it exists."""
    history_path = os.path.join("models", f"{app_name}_history.csv")
    if os.path.exists(history_path):
        return pd.read_csv(history_path)
    return None

# --- Helper Functions ---

def generate_wordcloud_fig(text_series):
    text = ' '.join(text_series.dropna())
    if not text: return None
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', collocations=False, random_state=42).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def get_top_ngrams(corpus, ngram_range=(1, 1), n=20):
    if corpus.empty or corpus.isnull().all(): return pd.DataFrame()
    vec = CountVectorizer(ngram_range=ngram_range, stop_words=None).fit(corpus.dropna())
    bag_of_words = vec.transform(corpus.dropna())
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(words_freq[:n], columns=['N-gram', 'Frequency'])

def categorize_review(review_text, keywords, default_category):
    if not isinstance(review_text, str): return default_category
    review_lower = review_text.lower()
    for category, keys in keywords.items():
        if any(key in review_lower for key in keys): return category
    return default_category

@st.cache_data
def categorize_dataframe(df):
    """Applies categorization logic to the entire dataframe."""
    KEYWORDS = {
      "Bug": ["error", "bug", "force close", "crash", "tidak bisa", "berhenti", "lemot", "lambat", "masalah", "not responding", "eror", "ngebug", "fc"],
      "Permintaan Fitur": ["tambah", "fitur", "semoga", "bisa", "kalau bisa", "request", "saran", "usul", "ditambahkan", "update", "kembangkan"],
      "Rating & Pujian": ["bagus", "keren", "terbaik", "suka", "mantap", "bintang 5", "puas", "membantu", "mudah", "terima kasih", "good job", "love it"]
    }
    DEFAULT_CATEGORY = "Pengalaman Pengguna (User Experience)"
    df['category'] = df['cleaned_text'].apply(lambda text: categorize_review(text, KEYWORDS, DEFAULT_CATEGORY))
    return df

def highlight_changes(before_text, after_text):
    before_words, after_words = str(before_text).split(), str(after_text).split()
    sm = difflib.SequenceMatcher(None, before_words, after_words)
    result = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'replace': result.append(f'<span style="background-color: #ffdd75; padding: 2px 5px; border-radius: 3px;">{" ".join(after_words[j1:j2])}</span>')
        elif tag == 'insert': result.append(f'<span style="background-color: #a6f5a6; padding: 2px 5px; border-radius: 3px;">{" ".join(after_words[j1:j2])}</span>')
        elif tag == 'equal': result.append(" ".join(after_words[j1:j2]))
    return " ".join(result)

def conceptual_xai(text, sentiment):
    """Simulates XAI output by highlighting keywords."""
    positive_words = ["bagus", "keren", "terbaik", "suka", "mantap", "puas", "membantu", "mudah", "terima kasih", "good job", "love it", "cepat", "lancar"]
    negative_words = ["error", "bug", "force close", "crash", "tidak bisa", "berhenti", "lemot", "lambat", "masalah", "not responding", "eror", "ngebug", "fc", "jelek", "buruk"]
    
    highlighted_text = ""
    for word in text.split():
        if word in positive_words:
            highlighted_text += f' <span style="background-color: #2ecc71; padding: 2px 5px; border-radius: 3px;">{word}</span>'
        elif word in negative_words:
            highlighted_text += f' <span style="background-color: #e74c3c; padding: 2px 5px; border-radius: 3px;">{word}</span>'
        else:
            highlighted_text += f" {word}"
    return highlighted_text.strip()

# --- Visualization Pages ---

def show_dashboard_guide():
    st.title("Welcome to the Sentiment Analysis Dashboard!")
    st.markdown("""
    This interactive dashboard is designed to provide a comprehensive, end-to-end view of the sentiment analysis pipeline for Google Play Store reviews of various e-commerce apps.

    **How to Use This Dashboard:**

    1.  **Select a View:** Use the **Navigation** radio buttons in the sidebar to switch between different analytical pages.
    2.  **Filter by App:** For most pages, you can select a specific app from the dropdown in the sidebar to drill down into its data. The "All-App Overview" page provides a comparative look across all apps.
    3.  **Interact with Visuals:** Most charts are interactive. Hover over data points to see details, and use sliders and other widgets to filter and explore the data.

    ---

    ### Page Guide:

    * **Dashboard Guide:** You are here!
    * **Configuration:** Upload your own data, or download the current data and trained models.
    * **All-App Overview:** A high-level comparative dashboard to benchmark apps against each other on key metrics.
    * **Raw Data Visualization (Unlabeled):** An initial look at the raw, unlabeled review data before processing.
    * **EDA Visualization:** Perform deep Exploratory Data Analysis on an app's reviews.
    * **Preprocessing Visualization:** See a step-by-step breakdown of how a raw review is cleaned.
    * **Final Data Visualization (Labeled):** Explore the final, labeled dataset used for model training.
    * **Word Embedding Visualization:** A conceptual look at how Word2Vec learns semantic relationships.
    * **Training Visualization:** View the model architecture and its performance curves.
    * **Evaluation & Prediction:** Evaluate the trained model, analyze its mistakes, and try live predictions.
    """)

def show_configuration_page(df, unlabeled_df):
    st.header("Configuration & Asset Management")

    st.subheader("Data Management")
    st.markdown("Manage the datasets used by this dashboard.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Labeled Dataset**")
        uploaded_labeled_file = st.file_uploader("Upload 'google_play_sentiments_labeled.csv'", type=['csv'], key="labeled_uploader")
        if uploaded_labeled_file:
            st.session_state.uploaded_labeled_file = uploaded_labeled_file
            st.success("Labeled data uploaded. The app will use this data for the current session.")
        
        csv_labeled = categorize_dataframe(df).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Labeled Data as CSV",
            data=csv_labeled,
            file_name='categorized_google_play_reviews.csv',
            mime='text/csv',
        )

    with col2:
        st.markdown("**Unlabeled Dataset**")
        uploaded_unlabeled_file = st.file_uploader("Upload 'google_play_reviews_unlabeled.csv'", type=['csv'], key="unlabeled_uploader")
        if uploaded_unlabeled_file:
            st.session_state.uploaded_unlabeled_file = uploaded_unlabeled_file
            st.success("Unlabeled data uploaded. The app will use this data for the current session.")

        if not unlabeled_df.empty:
            csv_unlabeled = unlabeled_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Unlabeled Data as CSV",
                data=csv_unlabeled,
                file_name='unlabeled_google_play_reviews.csv',
                mime='text/csv',
            )

    st.markdown("---")

    st.subheader("Model & Tokenizer Management")
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Upload Assets**")
        uploaded_models = st.file_uploader("Upload model files (.h5)", type=['h5'], accept_multiple_files=True)
        if uploaded_models:
            for model_file in uploaded_models:
                with open(os.path.join(models_dir, model_file.name), "wb") as f:
                    f.write(model_file.getbuffer())
                st.success(f"Saved model: {model_file.name}")

        uploaded_tokenizers = st.file_uploader("Upload tokenizer files (.pkl)", type=['pkl'], accept_multiple_files=True)
        if uploaded_tokenizers:
            for tokenizer_file in uploaded_tokenizers:
                with open(os.path.join(models_dir, tokenizer_file.name), "wb") as f:
                    f.write(tokenizer_file.getbuffer())
                st.success(f"Saved tokenizer: {tokenizer_file.name}")

        if uploaded_models or uploaded_tokenizers:
            st.warning("New assets uploaded. Clear cache and rerun the app for changes to take effect.")
            if st.button("Clear Cache and Rerun"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()

    with col2:
        st.markdown("**Manage Existing Assets**")
        asset_files = [f for f in os.listdir(models_dir) if f.endswith((".h5", ".pkl"))]

        if not asset_files:
            st.info("No models or tokenizers found in the 'models' directory.")

        for filename in asset_files:
            file_col1, file_col2, file_col3 = st.columns([2, 1, 1])
            
            with file_col1:
                st.write(filename)

            with file_col2:
                 with open(os.path.join(models_dir, filename), "rb") as fp:
                    st.download_button(
                        label="Download",
                        data=fp,
                        file_name=filename,
                        mime="application/octet-stream",
                        key=f"download_{filename}"
                    )
            
            with file_col3:
                delete_key = f"delete_{filename}"
                confirm_key = f"confirm_delete_{filename}"

                if st.button("Delete", key=delete_key):
                    st.session_state[confirm_key] = True

            if st.session_state.get(confirm_key):
                st.warning(f"Are you sure you want to delete `{filename}`?")
                confirm_col1, confirm_col2 = st.columns(2)
                with confirm_col1:
                    if st.button("Yes, delete it", key=f"confirm_yes_{filename}"):
                        try:
                            os.remove(os.path.join(models_dir, filename))
                            st.success(f"Deleted {filename}")
                            del st.session_state[confirm_key]
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting file: {e}")
                with confirm_col2:
                    if st.button("Cancel", key=f"cancel_delete_{filename}"):
                        del st.session_state[confirm_key]
                        st.rerun()


def show_all_apps_overview(df):
    st.header("All-App Comparative Overview")
    st.markdown("A high-level comparison of all e-commerce apps in the dataset.")

    df_categorized = categorize_dataframe(df)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Competitive Scorecard", "Trend Analysis", "Complaint Hotspots", "User Journey Funnel", "Category Deep Dive"])

    with tab1:
        # --- Competitive Health Scorecard ---
        st.subheader("Competitive Health Scorecard")
        
        app_metrics = []
        for app_name in df['app_name'].unique():
            app_df = df_categorized[df_categorized['app_name'] == app_name]
            total_reviews = len(app_df)
            if total_reviews == 0: continue
            
            pos_reviews = len(app_df[app_df['sentiment_name'] == 'positive'])
            neg_reviews = len(app_df[app_df['sentiment_name'] == 'negative'])
            
            bug_reviews = len(app_df[app_df['category'] == 'Bug'])
            feature_reviews = len(app_df[app_df['category'] == 'Permintaan Fitur'])
            
            sentiment_score = (pos_reviews - neg_reviews) / total_reviews
            bug_rate = bug_reviews / total_reviews
            feature_rate = feature_reviews / total_reviews
            
            app_metrics.append({
                'App': app_name,
                'Total Reviews': total_reviews,
                'Sentiment Score': sentiment_score,
                'Bug Rate': bug_rate,
                'Feature Request Rate': feature_rate
            })
        
        scorecard_df = pd.DataFrame(app_metrics).set_index('App')
        st.dataframe(scorecard_df.style
                     .background_gradient(cmap='RdYlGn', subset=['Sentiment Score'])
                     .background_gradient(cmap='YlOrRd', subset=['Bug Rate'])
                     .background_gradient(cmap='Blues', subset=['Feature Request Rate'])
                     .format({
                         'Total Reviews': '{:,}',
                         'Sentiment Score': '{:.2%}',
                         'Bug Rate': '{:.2%}',
                         'Feature Request Rate': '{:.2%}'
                     }))

        # --- Competitive Landscape Matrix ---
        st.subheader("Competitive Landscape Matrix")
        landscape_chart = alt.Chart(scorecard_df.reset_index()).mark_circle().encode(
            x=alt.X('Bug Rate:Q', title='Bug Problem Rate (Higher is Worse)', scale=alt.Scale(zero=False)),
            y=alt.Y('Sentiment Score:Q', title='Overall Sentiment Score (Higher is Better)', scale=alt.Scale(zero=False)),
            size=alt.Size('Total Reviews:Q', title='Review Volume'),
            color=alt.Color('App:N'),
            tooltip=['App', 'Sentiment Score', 'Bug Rate', 'Total Reviews']
        ).properties(
            title="Competitive Landscape: Sentiment vs. Stability"
        ).interactive()
        st.altair_chart(landscape_chart, use_container_width=True)

    with tab2:
        st.subheader("Competitive Trend Analysis")
        
        metric_to_track = st.selectbox("Select Metric to Track:", ["Sentiment Score", "Bug Rate", "Average Score"])
        
        # Resample data by week
        trends_df = df_categorized.set_index('review_date').groupby('app_name').resample('W').agg(
            positive=('sentiment_name', lambda x: (x == 'positive').sum()),
            negative=('sentiment_name', lambda x: (x == 'negative').sum()),
            total=('sentiment_name', 'count'),
            bug_count=('category', lambda x: (x == 'Bug').sum()),
            avg_score=('score', 'mean')
        ).reset_index()

        trends_df['Sentiment Score'] = (trends_df['positive'] - trends_df['negative']) / trends_df['total']
        trends_df['Bug Rate'] = trends_df['bug_count'] / trends_df['total']
        trends_df['Average Score'] = trends_df['avg_score']
        
        trend_chart = alt.Chart(trends_df).mark_line().encode(
            x='review_date:T',
            y=f'{metric_to_track}:Q',
            color='app_name:N',
            tooltip=['app_name', 'review_date', metric_to_track]
        ).properties(title=f"Weekly Trend of {metric_to_track}").interactive()
        st.altair_chart(trend_chart, use_container_width=True)

    with tab3:
        st.subheader("Complaint Hotspots (Bug Category)")
        bug_reviews = df_categorized[df_categorized['category'] == 'Bug']
        
        negative_keywords = ["error", "bug", "crash", "lemot", "lambat", "masalah", "berhenti", "tidak bisa"]
        
        hotspot_data = []
        for app_name in df['app_name'].unique():
            app_bug_reviews = bug_reviews[bug_reviews['app_name'] == app_name]['cleaned_text']
            total_bug_reviews = len(app_bug_reviews)
            if total_bug_reviews > 0:
                for keyword in negative_keywords:
                    count = app_bug_reviews.str.contains(keyword).sum()
                    hotspot_data.append({
                        'App': app_name,
                        'Complaint': keyword,
                        'Frequency': count / total_bug_reviews if total_bug_reviews > 0 else 0
                    })
        
        hotspot_df = pd.DataFrame(hotspot_data)
        
        heatmap = alt.Chart(hotspot_df).mark_rect().encode(
            x='App:N',
            y='Complaint:N',
            color=alt.Color('Frequency:Q', scale=alt.Scale(scheme='reds')),
            tooltip=['App', 'Complaint', alt.Tooltip('Frequency:Q', format='.2%')]
        ).properties(title="Complaint Keyword Frequency within 'Bug' Category")
        st.altair_chart(heatmap, use_container_width=True)

    with tab4:
        st.subheader("User Journey Funnel (Rating to Category)")
        
        sankey_data = df_categorized.groupby(['score', 'category']).size().reset_index(name='count')
        
        all_nodes = pd.concat([
            sankey_data['score'].apply(lambda x: f"{x} Star"), 
            sankey_data['category']
        ]).unique().tolist()
        
        sankey_data['source_id'] = sankey_data['score'].apply(lambda x: all_nodes.index(f"{x} Star"))
        sankey_data['target_id'] = sankey_data['category'].apply(lambda x: all_nodes.index(x))
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
            ),
            link=dict(
                source=sankey_data['source_id'],
                target=sankey_data['target_id'],
                value=sankey_data['count']
            ))])
        fig.update_layout(title_text="Flow from Star Rating to Review Category", font_size=10)
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Category Deep Dive")
        
        # Calculate negative sentiment rate per category
        category_sentiment_counts = df_categorized.groupby(['app_name', 'category', 'sentiment_name']).size().unstack(fill_value=0)
        category_sentiment_counts['total'] = category_sentiment_counts.sum(axis=1)
        category_sentiment_counts['negative_rate'] = category_sentiment_counts['negative'] / category_sentiment_counts['total']
        
        neg_rate_df = category_sentiment_counts.reset_index()

        neg_rate_chart = alt.Chart(neg_rate_df).mark_bar().encode(
            x=alt.X('app_name:N', title='Application'),
            y=alt.Y('negative_rate:Q', title='Negative Sentiment Rate', axis=alt.Axis(format='%')),
            color='app_name:N',
            column='category:N',
            tooltip=['app_name', 'category', alt.Tooltip('negative_rate:Q', format='.2%')]
        ).properties(
            title="Negative Sentiment Rate by Category"
        )
        st.altair_chart(neg_rate_chart)

        st.markdown("---")
        st.subheader("Sentiment Counts by Category (Raw Data)")
        pivot_df = pd.pivot_table(
            df_categorized,
            index=['category', 'sentiment_name'],
            columns='app_name',
            values='review_date',
            aggfunc='count',
            fill_value=0
        )
        st.dataframe(pivot_df)


def show_new_raw_data_visualization(app_df, app_name):
    st.header(f"Raw Data Visualization (Unlabeled) for {app_name.title()}")
    st.markdown("An initial look at the raw, unlabeled dataset scraped from the Google Play Store.")

    if app_df.empty:
        st.warning(f"No unlabeled data found for {app_name}. Please check the dataset.")
        st.stop()

    st.subheader("Key Metrics")
    total_reviews = len(app_df)
    avg_score = app_df['score'].mean()

    col1, col2 = st.columns(2)
    col1.metric("Total Reviews", f"{total_reviews:,}")
    col2.metric("Average Score", f"{avg_score:.2f} ⭐")
    
    st.subheader("Paginated Data Preview")
    st.markdown("Click on column headers to sort.")
    page_size = st.select_slider("Rows per page", options=[10, 25, 50, 100], value=10, key="raw_unlabeled_page_size")
    total_pages = max((total_reviews // page_size) + (1 if total_reviews % page_size > 0 else 0), 1)
    page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="raw_unlabeled_page_num")
    start_idx, end_idx = (page_number - 1) * page_size, page_number * page_size
    st.dataframe(app_df[['userName', 'original_review', 'score', 'review_date']].iloc[start_idx:end_idx])
    st.caption(f"Showing rows {start_idx + 1} to {min(end_idx, total_reviews)} of {total_reviews}")

    st.subheader("Rating Distribution")
    rating_chart = alt.Chart(app_df).mark_bar().encode(
        x=alt.X('score:O', title='Star Rating'),
        y=alt.Y('count():Q', title='Number of Reviews'),
        tooltip=['score:O', 'count()']
    ).properties(title=f"Distribution of Star Ratings for {app_name.title()}").interactive()
    st.altair_chart(rating_chart, use_container_width=True)


def show_final_data_visualization(app_df, app_name):
    st.header(f"Final Data Visualization (Labeled) for {app_name.title()}")
    st.markdown("An overview of the final, labeled dataset used for model training and evaluation.")

    tab1, tab2 = st.tabs(["Key Metrics & Data Preview", "Data Quality & Time Series"])

    with tab1:
        st.subheader("Key Metrics")
        total_reviews = app_df.shape[0]
        avg_score = app_df['score'].mean()
        date_range = app_df['review_date'].min().strftime('%Y-%m-%d') + " to " + app_df['review_date'].max().strftime('%Y-%m-%d')

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", f"{total_reviews:,}")
        col2.metric("Average Score", f"{avg_score:.2f} ⭐")
        col3.metric("Date Range", date_range)

        st.subheader("Paginated Dataset Preview")
        page_size = st.select_slider("Rows per page", options=[10, 25, 50, 100], value=10)
        total_pages = max((total_reviews // page_size) + (1 if total_reviews % page_size > 0 else 0), 1)
        page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start_idx, end_idx = (page_number - 1) * page_size, page_number * page_size
        st.dataframe(app_df.iloc[start_idx:end_idx])
        st.caption(f"Showing rows {start_idx + 1} to {min(end_idx, total_reviews)} of {total_reviews}")

    with tab2:
        st.subheader("Data Quality Check")
        missing_values = app_df.isnull().sum()
        missing_percent = (missing_values / total_reviews * 100).round(2)
        quality_df = pd.DataFrame({'Missing Values': missing_values, '% Missing': missing_percent})
        st.dataframe(quality_df[quality_df['Missing Values'] > 0])
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Review Volume Over Time")
            reviews_by_day = app_df.set_index('review_date').resample('D').size().reset_index(name='count')
            time_chart = alt.Chart(reviews_by_day).mark_line().encode(
                x=alt.X('review_date:T', title='Date'),
                y=alt.Y('count:Q', title='Number of Reviews'),
                tooltip=['review_date:T', 'count:Q']
            ).properties(title="Daily Review Volume").interactive()
            st.altair_chart(time_chart, use_container_width=True)

        with col2:
            st.subheader("Rating Distribution")
            rating_chart = alt.Chart(app_df).mark_bar().encode(
                x=alt.X('score:O', title='Star Rating'),
                y=alt.Y('count():Q', title='Number of Reviews'),
                tooltip=['score:O', 'count()']
            ).properties(title="Distribution of Star Ratings").interactive()
            st.altair_chart(rating_chart, use_container_width=True)


def show_eda_visualization(app_df, app_name):
    st.header(f"Exploratory Data Analysis for {app_name.title()}")
    
    app_df['review_length'] = app_df['cleaned_text'].str.split().str.len()

    tab1, tab2, tab3 = st.tabs(["Sentiment & Score Analysis", "Word & Phrase Analysis", "Advanced Analysis"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Breakdown")
            sentiment_counts = app_df['sentiment_name'].value_counts().reset_index()
            fig = px.pie(sentiment_counts, values='count', names='sentiment_name', title='Proportion of Sentiments',
                         color='sentiment_name', color_discrete_map={'positive': '#2ecc71', 'neutral': '#f1c40f', 'negative': '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Sentiment vs. Star Rating")
            sentiment_score_chart = alt.Chart(app_df).mark_bar().encode(
                x=alt.X('sentiment_name:N', title='Sentiment', sort=['negative', 'neutral', 'positive']),
                y=alt.Y('count():Q', title='Number of Reviews'),
                color=alt.Color('score:O', title='Star Rating'),
                tooltip=['sentiment_name', 'score', 'count()']
            ).properties(title="Sentiment Distribution by Star Rating").interactive()
            st.altair_chart(sentiment_score_chart, use_container_width=True)
            
        st.subheader("Average Sentiment Score Over Time")
        sentiment_by_day = app_df.set_index('review_date').resample('W').agg(
            avg_sentiment=('sentiment_label', 'mean'),
            std_dev=('sentiment_label', 'std')
        ).reset_index()
        sentiment_by_day['upper_band'] = sentiment_by_day['avg_sentiment'] + sentiment_by_day['std_dev']
        sentiment_by_day['lower_band'] = sentiment_by_day['avg_sentiment'] - sentiment_by_day['std_dev']

        base = alt.Chart(sentiment_by_day).encode(x='review_date:T')
        band = base.mark_area(opacity=0.3).encode(
            y='lower_band:Q',
            y2='upper_band:Q'
        ).properties(title="Weekly Average Sentiment Trend with Std. Deviation")
        line = base.mark_line(color='blue').encode(
            y=alt.Y('avg_sentiment:Q', title='Average Sentiment (0=Neg, 2=Pos)'),
            tooltip=['review_date:T', 'avg_sentiment:Q']
        )
        st.altair_chart((band + line).interactive(), use_container_width=True)

    with tab2:
        st.subheader("Most Frequent Terms")
        ngram_option = st.selectbox("Select N-gram:", ('Unigrams', 'Bigrams', 'Trigrams'), key=f'ngram_{app_name}')
        ngram_map = {'Unigrams': (1, 1), 'Bigrams': (2, 2), 'Trigrams': (3, 3)}
        top_ngrams_df = get_top_ngrams(app_df['cleaned_text'], ngram_range=ngram_map[ngram_option], n=20)
        if not top_ngrams_df.empty:
            ngram_chart = alt.Chart(top_ngrams_df).mark_bar().encode(x=alt.X('Frequency:Q'), y=alt.Y('N-gram:N', sort='-x'), tooltip=['N-gram', 'Frequency'])
            st.altair_chart(ngram_chart, use_container_width=True)
        
        st.subheader("Word Clouds by Sentiment")
        sentiments = ['positive', 'negative', 'neutral']
        cols = st.columns(len(sentiments))
        for i, sentiment in enumerate(sentiments):
            with cols[i]:
                st.markdown(f"**{sentiment.title()}**")
                sentiment_df = app_df[app_df['sentiment_name'] == sentiment]
                if not sentiment_df.empty:
                    wordcloud_fig = generate_wordcloud_fig(sentiment_df['cleaned_text'])
                    if wordcloud_fig: st.pyplot(wordcloud_fig, use_container_width=True)
                else: st.info(f"No {sentiment} reviews.")

    with tab3:
        st.subheader("Sentiment by Review Length")
        length_chart = alt.Chart(app_df).mark_boxplot(extent='min-max').encode(
            x=alt.X('sentiment_name:N', title='Sentiment'),
            y=alt.Y('review_length:Q', title='Review Length (words)'),
            color='sentiment_name:N'
        ).properties(title="Review Length Distribution by Sentiment").interactive()
        st.altair_chart(length_chart, use_container_width=True)

def show_preprocessing_visualization(app_df, app_name):
    st.header(f"Preprocessing Pipeline for {app_name.title()}")
    st.markdown("See how each step transforms a review. Changes are highlighted, and statistics are provided.")
    
    if app_df.empty: st.warning("No reviews to process."); st.stop()

    review_index = st.number_input("Select Review Index:", min_value=0, max_value=len(app_df)-1, value=0, step=1)
    original_text = app_df.iloc[review_index]['original_review']

    st.subheader("Original Review")
    st.info(original_text)

    slang_dict = {'yg': 'yang', 'ga': 'tidak', 'gak': 'tidak', 'gk': 'tidak', 'gaada': 'tidak ada', 'nya': '', 'aja': 'saja', 'sih': '', 'dah': 'sudah', 'deh': '', 'sm': 'sama', 'utk': 'untuk', 'bgt': 'banget', 'lg': 'lagi', 'klo': 'kalau', 'kalo': 'kalau', 'udh': 'sudah', 'udah': 'sudah', 'dlm': 'dalam', 'jg': 'juga', 'dr': 'dari', 'trs': 'terus', 'bgtu': 'begitu', 'dgn': 'dengan', 'gua': 'saya', 'gue': 'saya', 'tdk': 'tidak', 'tpi': 'tapi', 'jd': 'jadi', 'blm': 'belum', 'g': 'tidak'}
    stemmer = get_stemmer()

    # Step-by-step transformation
    text_s0 = str(original_text)
    len_s0 = len(text_s0.split())

    text_s1 = text_s0.lower()
    len_s1 = len(text_s1.split())
    with st.expander("Step 1: Lowercasing", expanded=True):
        st.markdown(highlight_changes(text_s0, text_s1), unsafe_allow_html=True)
        st.caption(f"Word count: {len_s0} -> {len_s1}")

    words_s1 = text_s1.split()
    replaced_slang = [word for word in words_s1 if word in slang_dict]
    normalized_words = [slang_dict.get(word, word) for word in words_s1]
    text_s2 = " ".join(normalized_words)
    len_s2 = len(text_s2.split())
    with st.expander("Step 2: Slang Normalization", expanded=True):
        st.markdown(highlight_changes(text_s1, text_s2), unsafe_allow_html=True)
        st.caption(f"Word count: {len_s1} -> {len_s2}. Slang words replaced: {len(replaced_slang)} {'(' + ', '.join(replaced_slang) + ')' if replaced_slang else ''}")

    text_s3 = re.sub(r'[^a-zA-Z\s]', '', text_s2)
    len_s3 = len(text_s3.split())
    with st.expander("Step 3: Punctuation & Character Removal", expanded=True):
        st.markdown(highlight_changes(text_s2, text_s3), unsafe_allow_html=True)
        st.caption(f"Word count: {len_s2} -> {len_s3}")
    
    text_s4 = stemmer.stem(text_s3)
    len_s4 = len(text_s4.split())
    with st.expander("Step 4: Stemming", expanded=True):
        st.markdown(highlight_changes(text_s3, text_s4), unsafe_allow_html=True)
        st.caption(f"Word count: {len_s3} -> {len_s4}")

    st.subheader("Final Cleaned Text")
    st.success(app_df.iloc[review_index]['cleaned_text'])

def show_word_embedding_visualization(app_name):
    st.header("Word Embedding Visualization")
    st.markdown("""
    Word Embedding represents words as numerical vectors. Here, we conceptualize a **Word2Vec** model with a **CBOW** architecture.
    """)
    
    col1, col2 = st.columns([2, 1])

    similar_words_db = {
        "bagus": {"cluster": "Praise", "words": ["keren", "mantap", "terbaik", "sempurna", "hebat"]},
        "error": {"cluster": "Bugs", "words": ["bug", "masalah", "crash", "rusak", "berhenti"]},
        "cepat": {"cluster": "Performance", "words": ["lancar", "responsif", "ringan", "gesit", "kilat"]},
        "lambat": {"cluster": "Performance", "words": ["lemot", "berat", "lama", "lelet", "kurang responsif"]},
    }

    with col2:
        st.subheader("Find Similar Words")
        search_word = st.text_input("Enter a word:", value='bagus', key=f"search_{app_name}").lower()

        if search_word in similar_words_db:
            st.success(f"Words conceptually similar to **'{search_word}'**:")
            for word in similar_words_db[search_word]["words"]: st.markdown(f"- {word}")
        else:
            st.warning("Word not in our conceptual dictionary. Try: bagus, error, cepat, lambat.")

        st.subheader("Conceptual Vector Arithmetic")
        st.markdown("`'bagus' - 'jelek' + 'lambat' = 'cepat'`")


    with col1:
        st.subheader("Conceptual t-SNE Plot")
        words_data = []
        seed = sum(ord(c) for c in app_name)
        np.random.seed(seed)

        for base_word, data in similar_words_db.items():
            center_x, center_y = np.random.rand(2) * 10
            words_data.append({'x': center_x, 'y': center_y, 'word': base_word, 'cluster': data['cluster']})
            for sim_word in data['words']:
                offset_x, offset_y = (np.random.rand(2) - 0.5) * 2
                words_data.append({'x': center_x + offset_x, 'y': center_y + offset_y, 'word': sim_word, 'cluster': data['cluster']})
        
        plot_df = pd.DataFrame(words_data)
        highlighted_cluster = similar_words_db.get(search_word, {}).get("cluster", "")

        scatter = alt.Chart(plot_df).mark_circle(size=120).encode(
            x=alt.X('x', axis=None), y=alt.Y('y', axis=None),
            color=alt.Color('cluster:N', legend=alt.Legend(title="Word Clusters")),
            tooltip=['word', 'cluster'],
            opacity=alt.condition(alt.datum.cluster == highlighted_cluster, alt.value(1.0), alt.value(0.3)),
            size=alt.condition(alt.datum.cluster == highlighted_cluster, alt.value(250), alt.value(120))
        ).properties(title=f'Conceptual 2D Projection for {app_name.title()}').interactive()
        st.altair_chart(scatter, use_container_width=True)

def show_training_visualization(app_name):
    st.header(f"Training Visualization for {app_name.title()}")
    
    tab1, tab2 = st.tabs(["Model Architecture", "Training Performance"])

    with tab1:
        st.subheader("CNN-LSTM Architecture")
        st.code("""
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=120),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.LSTM(units=32),
    tf.keras.layers.Dense(3, activation='softmax')
])
        """, language='python')
        st.subheader("Model Summary")
        st.code("""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 120, 128)          1280000   
 conv1d (Conv1D)             (None, 118, 64)           24640     
 max_pooling1d (MaxPooling1D) (None, 59, 64)            0         
 lstm (LSTM)                 (None, 32)                12416     
 dense (Dense)               (None, 3)                 99        
=================================================================
Total params: 1,317,155
        """, language='text')

    with tab2:
        st.subheader("Training & Validation Performance")
        history_df = load_history(app_name)
        if history_df is not None:
            st.info(f"Loaded real training history from `models/{app_name}_history.csv`")
            history_df.columns = [c.lower().replace('val_', 'validation_') for c in history_df.columns]
            history_df_melted = history_df.reset_index().melt('index', var_name='Metric', value_name='Value')
            history_df_melted.rename(columns={'index': 'Epoch'}, inplace=True)
        else:
            st.warning(f"Could not find `models/{app_name}_history.csv`. Displaying conceptual data.")
            epochs = np.arange(1, 16)
            seed = sum(ord(c) for c in app_name)
            np.random.seed(seed)
            train_acc = np.sort(0.65 + (epochs / 40) + np.random.uniform(-0.03, 0.03, 15))
            val_acc = np.clip(np.sort(0.68 + (epochs / 50) + np.random.uniform(-0.04, 0.04, 15)), 0, train_acc + 0.05)
            train_loss = np.sort(0.8 - (epochs / 30) + np.random.uniform(-0.03, 0.03, 15))[::-1]
            val_loss = np.clip(np.sort(0.75 - (epochs / 40) + np.random.uniform(-0.04, 0.04, 15)), train_loss - 0.05, 1)
            history_df = pd.DataFrame({'Epoch': epochs, 'accuracy': train_acc, 'validation_accuracy': val_acc, 'loss': train_loss, 'validation_loss': val_loss})
            history_df_melted = history_df.melt('Epoch', var_name='Metric', value_name='Value')

        col1, col2 = st.columns(2)
        with col1:
            acc_chart = alt.Chart(history_df_melted[history_df_melted['Metric'].str.contains('accuracy')]).mark_line(point=True).encode(
                x='Epoch:Q', y=alt.Y('Value:Q', scale=alt.Scale(domain=[0.5, 1.0]), title='Accuracy'), color='Metric:N',
                tooltip=['Epoch', 'Metric', alt.Tooltip('Value', format='.4f')]
            ).properties(title='Accuracy: Training vs. Validation').interactive()
            st.altair_chart(acc_chart, use_container_width=True)
        with col2:
            loss_chart = alt.Chart(history_df_melted[history_df_melted['Metric'].str.contains('loss')]).mark_line(point=True).encode(
                x='Epoch:Q', y=alt.Y('Value:Q', scale=alt.Scale(domain=[0, 1.0]), title='Loss'), color='Metric:N',
                tooltip=['Epoch', 'Metric', alt.Tooltip('Value', format='.4f')]
            ).properties(title='Loss: Training vs. Validation').interactive()
            st.altair_chart(loss_chart, use_container_width=True)

def show_evaluation_and_prediction(app_df, app_name):
    st.header(f"Evaluation & Prediction for {app_name.title()}")
    
    model, tokenizer = load_model_and_tokenizer(app_name)
    if model is None or tokenizer is None:
        st.error(f"Could not load model/tokenizer. Please ensure `{app_name}_sentiment_model.h5` and `{app_name}_tokenizer.pkl` exist in the `models` folder.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Live Predictor", "Performance on Test Set", "Error Analysis"])
    
    class_names = ['negative', 'neutral', 'positive']
    max_len = 120

    with tab1:
        st.subheader("Try the Live Sentiment Predictor")
        user_review = st.text_area("Enter a review text (in Indonesian):", "Aplikasi ini sangat bagus dan mudah digunakan!", key=f"live_predictor_{app_name}")
        if st.button("Classify Review", key=f"classify_{app_name}"):
            if user_review:
                stemmer = get_stemmer()
                cleaned_review = stemmer.stem(re.sub(r'[^a-zA-Z\s]', '', user_review.lower()))
                sequence = tokenizer.texts_to_sequences([cleaned_review])
                padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
                prediction = model.predict(padded_sequence)
                pred_index = np.argmax(prediction)
                st.success(f"**Predicted Sentiment:** {class_names[pred_index].title()}")
                st.warning(f"**Model Confidence:** {prediction[0][pred_index]:.2%}")
                
                st.subheader("Prediction Explanation (Conceptual)")
                st.markdown(conceptual_xai(cleaned_review, class_names[pred_index]), unsafe_allow_html=True)

            else:
                st.error("Please enter a review to classify.")

    with tab2:
        st.subheader("Model Performance Metrics")
        sequences = tokenizer.texts_to_sequences(app_df['cleaned_text'].astype(str))
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
        
        all_predictions = model.predict(padded_sequences)
        all_y_pred = np.argmax(all_predictions, axis=1)
        all_confidence = np.max(all_predictions, axis=1)
        y_true = app_df['sentiment_label']

        st.info("Use the slider to see how performance changes if you only consider predictions above a certain confidence level.")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        
        indices = np.where(all_confidence >= conf_threshold)[0]
        
        if len(indices) > 0:
            y_true_filtered, y_pred_filtered = y_true.iloc[indices], all_y_pred[indices]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Coverage", f"{len(indices) / len(y_true):.1%}", help="Percentage of test samples meeting the confidence threshold.")
                report = classification_report(y_true_filtered, y_pred_filtered, target_names=class_names, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))
            
            with col2:
                cm = confusion_matrix(y_true_filtered, y_pred_filtered)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
                ax.set_xlabel('Predicted Labels'); ax.set_ylabel('True Labels'); ax.set_title(f'Confusion Matrix (Confidence >= {conf_threshold:.0%})')
                st.pyplot(fig)
        else:
            st.warning("No predictions meet this high confidence threshold.")

    with tab3:
        st.subheader("Misclassified Review Inspector")
        st.markdown("Analyze where the model made mistakes to understand its weaknesses.")
        
        misclassified_mask = all_y_pred != y_true
        misclassified_df = app_df[misclassified_mask].copy()
        misclassified_df['predicted_label'] = [class_names[i] for i in all_y_pred[misclassified_mask]]
        misclassified_df['true_label'] = [class_names[i] for i in y_true[misclassified_mask]]
        misclassified_df['confidence'] = all_confidence[misclassified_mask]
        
        true_label_filter = st.multiselect("Filter by True Label:", options=class_names, default=class_names)
        pred_label_filter = st.multiselect("Filter by Predicted Label:", options=class_names, default=class_names)
        
        filtered_errors = misclassified_df[
            misclassified_df['true_label'].isin(true_label_filter) &
            misclassified_df['predicted_label'].isin(pred_label_filter)
        ]

        st.dataframe(filtered_errors[['original_review', 'true_label', 'predicted_label', 'confidence']])
        st.caption(f"Showing {len(filtered_errors)} of {len(misclassified_df)} total misclassified reviews.")

# --- Main App ---
def main():
    st.sidebar.title("Sentiment Analysis Dashboard")
    
    # Check for uploaded file in session state
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'uploaded_unlabeled_file' not in st.session_state:
        st.session_state.uploaded_unlabeled_file = None

    df = load_data(st.session_state.uploaded_file)
    unlabeled_df = load_unlabeled_data(st.session_state.uploaded_unlabeled_file)
    unique_apps = unlabeled_df['app_name'].unique().tolist() if not unlabeled_df.empty else df['app_name'].unique().tolist()
    
    st.sidebar.title("Navigation")
    page_selection = st.sidebar.radio("Go to:", list(PAGE_OPTIONS.keys()))
    
    app_name = "All Apps" # Default
    if page_selection == "All-App Overview":
        app_name = "All Apps"
    elif page_selection == "Dashboard Guide":
        app_name = None
    elif page_selection == "Configuration":
        app_name = None
    else:
        app_name = st.sidebar.selectbox("Select an App to Analyze:", unique_apps)

    st.sidebar.markdown("---")
    st.sidebar.info("This dashboard provides an end-to-end visualization of the sentiment analysis pipeline.")

    page_function = PAGE_OPTIONS[page_selection]
    
    if page_selection == "All-App Overview":
        page_function(df)
    elif page_selection == "Dashboard Guide":
        page_function()
    elif page_selection == "Configuration":
        page_function(df, unlabeled_df)
    elif page_selection == "Raw Data Visualization (Unlabeled)":
        app_df_unlabeled = unlabeled_df[unlabeled_df['app_name'] == app_name].copy()
        page_function(app_df_unlabeled, app_name)
    elif page_selection in ["Final Data Visualization (Labeled)", "EDA Visualization", "Preprocessing Visualization", "Evaluation & Prediction"]:
        app_df = df[df['app_name'] == app_name].copy()
        page_function(app_df, app_name)
    else:
        page_function(app_name)

# Page dictionary must be defined after the functions themselves
PAGE_OPTIONS = {
    "Dashboard Guide": show_dashboard_guide,
    "Configuration": show_configuration_page,
    "All-App Overview": show_all_apps_overview,
    "Raw Data Visualization (Unlabeled)": show_new_raw_data_visualization,
    "EDA Visualization": show_eda_visualization,
    "Preprocessing Visualization": show_preprocessing_visualization,
    "Final Data Visualization (Labeled)": show_final_data_visualization,
    "Word Embedding Visualization": show_word_embedding_visualization,
    "Training Visualization": show_training_visualization,
    "Evaluation & Prediction": show_evaluation_and_prediction,
}

if __name__ == "__main__":
    main()