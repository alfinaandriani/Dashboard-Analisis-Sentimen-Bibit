import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load data yang sudah dianalisis
df = pd.read_csv("Bibit_sentiment.csv")

# Fungsi konversi score ke sentimen
def score_to_sentiment(score):
    if score <= 2:
        return 'negatif'
    elif score == 3:
        return 'netral'
    else:
        return 'positif'

# Tambahkan kolom sentimen asli berdasarkan score
df['score_group'] = df['score'].apply(score_to_sentiment)

# =============================
# PIE CHART DISTRIBUSI SENTIMEN
# =============================
st.title("Dashboard Analisis Sentimen Bibit")

st.subheader("Distribusi Sentimen Berdasarkan Score vs MNB")
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
colors = ['mediumseagreen', 'darkgreen', 'lightgreen']

# Pie chart untuk score asli
df['score_group'].value_counts().plot.pie(
    autopct='%1.1f%%', ax=axs[0], colors=colors
)
axs[0].set_title("Distribusi Sentimen berdasarkan score rating")
axs[0].set_ylabel("")

# Pie chart untuk prediksi
df['sentiment_predicted'].value_counts().plot.pie(
    autopct='%1.1f%%', ax=axs[1], colors=colors
)
axs[1].set_title("Distribusi Sentimen oleh MNB")
axs[1].set_ylabel("")

st.pyplot(fig)

# ======================
# WORDCLOUD PER SENTIMEN
# ======================
st.subheader("Visualisasi WordCloud berdasarkan Sentimen Prediksi")

# Fungsi menampilkan WordCloud
def show_wordcloud(text, title, colormap):
    wordcloud = WordCloud(
        width=500,
        height=500,
        background_color='white',
        colormap=colormap,
        max_words=200,
        random_state=42
    ).generate(' '.join(text))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16)
    st.pyplot(fig)

colormaps = {
    'negatif': 'Reds',
    'netral': 'Greys',
    'positif': 'Greens'
}

# Pilih sentimen untuk WordCloud
sentimen_pilihan = st.selectbox(
    "Pilih Sentimen untuk WordCloud",
    options=['positif', 'netral', 'negatif']
)

texts = df[df['sentiment_predicted'] == sentimen_pilihan]['content_clean']
title = f'WordCloud Sentimen: {sentimen_pilihan.capitalize()}'
show_wordcloud(texts, title, colormaps[sentimen_pilihan])
