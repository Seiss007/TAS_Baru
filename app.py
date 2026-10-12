import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(layout="wide", page_title="AI Model App", page_icon="🌐", initial_sidebar_state="expanded")

# ==========================================
# FACEBOOK-STYLE CSS
# ==========================================
st.markdown("""
<style>
    /* Mengubah background utama menjadi abu-abu khas Facebook */
    .stApp {
        background-color: #F0F2F5;
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar putih bersih */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E4E6EB;
    }
    
    /* Teks Judul Biru Facebook */
    .fb-header {
        color: #1877F2;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 5px;
        font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
    }
    
    /* Style untuk Kartu Metrik (Mirip Post Facebook) */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E4E6EB;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        padding: 15px;
    }
    
    div[data-testid="metric-container"] label {
        color: #65676B !important;
        font-size: 0.95rem !important;
        font-weight: 600;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #050505 !important;
        font-weight: bold;
    }

    /* Style Tombol Biru Facebook */
    .stButton>button {
        background-color: #1877F2 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1rem;
        padding: 0.5rem 1rem;
        transition: background-color 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #166FE5 !important;
    }

    /* Style Tabs Menu */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 1px solid #E4E6EB;
    }
    .stTabs [data-baseweb="tab"] {
        color: #65676B;
        font-weight: 600;
        border-radius: 0;
    }
    .stTabs [aria-selected="true"] {
        color: #1877F2 !important;
        border-bottom: 3px solid #1877F2 !important;
    }
    
    /* Expander / Box Container Background */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #050505 !important;
    }
</style>
""", unsafe_allow_html=True)

# Memastikan matplotlib menggunakan tema terang yang bersih
sns.set_theme(style="whitegrid")

# ==========================================
# CACHE DATASET
# ==========================================
@st.cache_data
def load_data(dataset_name, task_type):
    if task_type == "Regresi":
        if dataset_name == "Diabetes": data = datasets.load_diabetes()
        elif dataset_name == "California Housing": data = datasets.fetch_california_housing()
        else: return None, None
    elif task_type == "Klasifikasi":
        if dataset_name == "Iris": data = datasets.load_iris()
        elif dataset_name == "Breast Cancer": data = datasets.load_breast_cancer()
        elif dataset_name == "Wine": data = datasets.load_wine()
        else: return None, None
            
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['TARGET'] = data.target
    return df, data.target_names if hasattr(data, 'target_names') else None

# ==========================================
# SIDEBAR UI
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color:#050505;'>Pengaturan</h2>", unsafe_allow_html=True)
    st.write("Sesuaikan parameter model di bawah ini.")
    st.divider()
    
    task = st.radio("Jenis Model", ["Regresi", "Klasifikasi"])
    
    if task == "Regresi":
        dataset_options = ["Diabetes", "California Housing"]
        model_options = ["Linear Regression", "Ridge", "Lasso", "Decision Tree", "Random Forest"]
    else:
        dataset_options = ["Iris", "Breast Cancer", "Wine"]
        model_options = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"]
        
    dataset_name = st.selectbox("Pilih Dataset", dataset_options)
    model_name = st.selectbox("Pilih Algoritma", model_options)
    
    st.divider()
    
    test_size = st.slider("Ukuran Data Testing", 0.1, 0.5, 0.2, 0.05, format="%.2f")
    random_state = st.number_input("Random State", value=42, step=1)
    
    st.divider()
    btn_train = st.button("Latih Model")

# ==========================================
# INISIALISASI SESSION STATE
# ==========================================
if "is_trained" not in st.session_state:
    st.session_state.is_trained = False

# ==========================================
# PROSES TRAINING
# ==========================================
if btn_train:
    df, target_names = load_data(dataset_name, task)
    
    if df is not None:
        X = df.drop(columns=['TARGET'])
        y = df['TARGET']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        if task == "Regresi":
            if model_name == "Linear Regression": model = LinearRegression()
            elif model_name == "Ridge": model = Ridge(random_state=random_state)
            elif model_name == "Lasso": model = Lasso(random_state=random_state)
            elif model_name == "Decision Tree": model = DecisionTreeRegressor(random_state=random_state)
            elif model_name == "Random Forest": model = RandomForestRegressor(random_state=random_state)
        else:
            if model_name == "Logistic Regression": model = LogisticRegression(max_iter=2000, random_state=random_state)
            elif model_name == "KNN": model = KNeighborsClassifier()
            elif model_name == "Decision Tree": model = DecisionTreeClassifier(random_state=random_state)
            elif model_name == "Random Forest": model = RandomForestClassifier(random_state=random_state)
            elif model_name == "SVM": model = SVC(random_state=random_state)
            
        with st.spinner("Memproses data..."):
            time.sleep(0.5)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        st.session_state.is_trained = True
        st.session_state.task = task
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.target_names = target_names
        st.session_state.model_name = model_name
        st.session_state.dataset_name = dataset_name


st.markdown('<div class="fb-header">Machine Learning App</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#65676B; margin-top:-10px; margin-bottom:20px;'>Platform analisis Machine Learning sederhana.</p>", unsafe_allow_html=True)

if not st.session_state.is_trained:
    # Tampilan kosong mirip timeline Facebook yang belum ada post
    with st.container():
        st.info("Halo! Silakan atur model di sebelah kiri dan klik 'Latih Model' untuk melihat hasilnya di sini.")
        
        with st.expander("Tentang Aplikasi Ini", expanded=True):
            st.write("""
            **Cara Menggunakan:**
            1. Pilih jenis *Task* (Regresi atau Klasifikasi) di menu samping.
            2. Pilih kombinasi *Dataset* dan *Algoritma* yang ingin digunakan.
            3. Sesuaikan pembagian ukuran data *Testing*.
            4. Klik tombol **Latih Model** warna biru.
            """)
else:
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred
    task = st.session_state.task
    target_names = st.session_state.target_names

    st.success(f"Berhasil melatih {st.session_state.model_name} dengan dataset {st.session_state.dataset_name}.")

    tab1, tab2 = st.tabs(["📊 Visualisasi", "📈 Evaluasi"])
    
    # --- TAB 1: VISUALISASI ---
    with tab1:
        st.markdown("<h4 style='color:#050505;'>Hasil Prediksi Model</h4>", unsafe_allow_html=True)
        
        if task == "Regresi":
            col_chart, col_data = st.columns([2, 1])
            
            with col_chart:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot warna biru Facebook
                ax.scatter(y_test, y_pred, alpha=0.7, color="#1877F2", edgecolors="white", s=60)
                
                min_val = min(np.min(y_test), np.min(y_pred))
                max_val = max(np.max(y_test), np.max(y_pred))
                ax.plot([min_val, max_val], [min_val, max_val], color="#FA383E", linestyle="--", lw=2, label="Perfect Fit")
                
                ax.set_xlabel("Nilai Aktual", color="#65676B", fontweight='bold')
                ax.set_ylabel("Nilai Prediksi", color="#65676B", fontweight='bold')
                
                ax.grid(color='#E4E6EB', linestyle='-', linewidth=1)
                for spine in ax.spines.values():
                    spine.set_color('#E4E6EB')
                    
                ax.legend()
                
                # Atur background figure agar transparan/putih
                fig.patch.set_facecolor('#FFFFFF')
                ax.set_facecolor('#FFFFFF')
                
                st.pyplot(fig)
                
            with col_data:
                st.write("**Data Sampel**")
                df_compare = pd.DataFrame({"Aktual": y_test, "Prediksi": y_pred}).head(15).round(2)
                st.dataframe(df_compare, use_container_width=True)
            
        elif task == "Klasifikasi":
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            
            labels = target_names if target_names is not None else "auto"
            
            # Colormap biru bersih
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                        xticklabels=labels, yticklabels=labels,
                        annot_kws={"size": 12, "weight": "normal"})
            
            ax.set_xlabel("Prediksi", color="#65676B", fontweight='bold', labelpad=10)
            ax.set_ylabel("Aktual", color="#65676B", fontweight='bold', labelpad=10)
            
            fig.patch.set_facecolor('#FFFFFF')
            st.pyplot(fig)

    # --- TAB 2: EVALUATION METRICS ---
    with tab2:
        st.markdown("<h4 style='color:#050505;'>Kinerja Model</h4>", unsafe_allow_html=True)
        
        if task == "Regresi":
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{mae:.4f}")
            col2.metric("MSE", f"{mse:.4f}")
            col3.metric("RMSE", f"{rmse:.4f}")
            col4.metric("R² Score", f"{r2:.4f}")
            
        elif task == "Klasifikasi":
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Akurasi", f"{acc:.4f}")
            col2.metric("Presisi", f"{prec:.4f}")
            col3.metric("Recall", f"{rec:.4f}")
            col4.metric("Skor F1", f"{f1:.4f}")
            
            st.divider()
            st.write("**Laporan Klasifikasi Lengkap**")
            if target_names is not None:
                try:
                    report = classification_report(y_test, y_pred, target_names=target_names)
                except ValueError:
                    report = classification_report(y_test, y_pred)
            else:
                report = classification_report(y_test, y_pred)
            st.code(report, language="text")