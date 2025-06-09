import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set konfigurasi halaman
st.set_page_config(page_title="Analisis Sentimen Bibit", layout="wide")

# Load model & vectorizer
model = joblib.load("model_sentimen.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load dataset yang sudah mengandung content_bersih
df = pd.read_csv("Bibit_2024_bersih.csv")

# Pastikan kolom content_bersih ada
if 'content_bersih' not in df.columns:
    st.error("Kolom 'content_bersih' tidak ditemukan di dataset.")
    st.stop()

# Label sentimen
df['sentiment'] = df['score'].apply(lambda score: 'negatif' if score <= 2 else 'netral' if score == 3 else 'positif')

# Sidebar Navigasi
st.sidebar.title("Sentimen Bibit 2024")
selected_page = st.sidebar.radio("Pilih Sentimen", ["Home", "Positif", "Netral", "Negatif"])

# Halaman Home
if selected_page == "Home":
    st.title("📊 Ringkasan Sentimen Data Bibit 2024")
    
    sentiment_counts = df['sentiment'].value_counts().reindex(['positif', 'netral', 'negatif'])
    total_data = sentiment_counts.sum()

    st.subheader("Jumlah dan Persentase Sentimen")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="Set2", ax=ax)
        ax.set_title("Distribusi Sentimen")
        ax.set_ylabel("Jumlah")
        ax.set_xlabel("Sentimen")
        st.pyplot(fig)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
        ax2.axis('equal')
        st.pyplot(fig2)

    st.markdown(f"📦 **Total Data:** {total_data}")

    # Tampilkan jumlah per sentimen
    st.markdown(
        f"""
        - 👍 **Positif:** {sentiment_counts['positif']}  
        - 😐 **Netral:** {sentiment_counts['netral']}  
        - 👎 **Negatif:** {sentiment_counts['negatif']}
        """
    )


# Halaman per Sentimen
else:
    sentiment_lower = selected_page.lower()
    st.title(f"Sentimen {selected_page}")

    filtered_df = df[df['sentiment'] == sentiment_lower]

    # WordCloud
    st.subheader("☁️ WordCloud")
    combined_text = " ".join(filtered_df['content_bersih'].astype(str))
    if combined_text.strip():
        cmap = {"positif": "Greens", "netral": "gray", "negatif": "Reds"}[sentiment_lower]
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=cmap).generate(combined_text)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    else:
        st.warning("Tidak ada data untuk ditampilkan.")

    # Tampilkan tabel
    # st.dataframe(filtered_df[['content_bersih']].rename(columns={'content_bersih': 'Ulasan'}), use_container_width=True)
