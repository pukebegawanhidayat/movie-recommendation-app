import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Movie Analytics & Recommendation",
    page_icon="🎬",
    layout="wide",
)


@st.cache_resource
def load_all_assets():
  df = joblib.load("movies_dataframe.pkl")
  cosine_sim = joblib.load("cosine_sim_matrix.pkl")
  rf_model = joblib.load("popularity_rf_model.pkl")
  return df, cosine_sim, rf_model


try:
  df, cosine_sim, rf_model = load_all_assets()
  st.title("🎬 Movie Analytics, Recommendation & Prediction App")
  st.write(
      "Aplikasi End-to-End gabungan antara Sistem Rekomendasi Film dan Prediksi"
      " Popularitas Machine Learning."
  )

  tab1, tab2 = st.tabs(
      ["🎯 Sistem Rekomendasi Film", "🤖 Simulator Prediksi Popularitas"]
  )

  with tab1:
    st.header("Cari & Dapatkan Rekomendasi Film Serupa")
    movie_list = df["Title"].values
    selected_movie = st.selectbox("🔍 Pilih Judul Film:", movie_list)

    indices = pd.Series(df.index, index=df["Title"]).drop_duplicates()

    if st.button("Tampilkan Detail & Rekomendasi"):
      selected_row = df[df["Title"] == selected_movie].iloc[0]
      col1, col2 = st.columns([1, 3])

      with col1:
        if "Poster_Url" in df.columns and pd.notna(selected_row["Poster_Url"]):
          st.image(selected_row["Poster_Url"], width=180)
        else:
          st.write("Poster N/A")

      with col2:
        st.subheader(selected_row["Title"])
        st.write(f"**Genre:** {selected_row['Genre']}")
        st.write(f"**Rating:** ⭐ {selected_row['Vote_Average']} / 10")
        st.write(f"**Popularitas:** 🔥 {selected_row['Popularity']}")
        st.write(f"**Sinopsis:** {selected_row['Overview']}")

      st.markdown("---")
      st.subheader("🍿 Film Lain yang Memiliki Alur & Genre Mirip:")

      idx = indices[selected_movie]
      if isinstance(idx, pd.Series):
        idx = idx.iloc[0]

      sim_scores = list(enumerate(cosine_sim[idx]))
      sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
      movie_indices = [i[0] for i in sim_scores]
      recs = df.iloc[movie_indices]

      for i, row in recs.reset_index(drop=True).iterrows():
        st.write(
            f"**{i+1}. {row['Title']}** ({row['Genre']}) — ⭐"
            f" {row['Vote_Average']}"
        )

  with tab2:
    st.header("Simulator Prediksi Popularitas Film Baru")
    st.write(
        "Masukkan estimasi statistik film Anda untuk memprediksi skor"
        " popularitasnya."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
      input_vote_count = st.number_input(
          "Estimasi Jumlah Vote:", min_value=0, max_value=50000, value=500
      )
    with col_b:
      input_vote_avg = st.slider(
          "Estimasi Rating (0 - 10):",
          min_value=0.0,
          max_value=10.0,
          value=7.5,
          step=0.1,
      )
    with col_c:
      input_release_year = st.number_input(
          "Tahun Rilis:", min_value=1900, max_value=2030, value=2024
      )

    if st.button("Hitung Prediksi Popularitas"):
      input_data = np.array(
          [[input_vote_count, input_vote_avg, input_release_year]]
      )
      predicted_log = rf_model.predict(input_data)[0]
      predicted_pop = np.expm1(predicted_log)
      st.success(f"🔥 **Hasil Prediksi Skor Popularitas:** {predicted_pop:.2f}")

except Exception as e:
  st.error(f"Gagal memuat file model atau dataset: {e}")