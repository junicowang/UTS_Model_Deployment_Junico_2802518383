import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# Page Config
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Models
@st.cache_resource
def load_models():
    base_path = os.path.join(os.path.dirname(__file__), '..', 'models')
    clf_pipeline = joblib.load(os.path.join(base_path, 'classification_pipeline.pkl'))
    reg_pipeline = joblib.load(os.path.join(base_path, 'regression_pipeline.pkl'))
    return clf_pipeline, reg_pipeline

try:
    clf_pipeline, reg_pipeline = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    error_msg = str(e)

# Custom CSS
st.markdown("""
<style>
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .placed {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .not-placed {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .salary-box {
        background-color: #d1ecf1;
        border: 2px solid #17a2b8;
        color: #0c5460;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🎓 Student Placement Predictor")
st.caption("Prediksi status penempatan kerja dan estimasi gaji mahasiswa berdasarkan profil akademik & keterampilan")
st.divider()

if not models_loaded:
    st.error(f"Gagal memuat model: {error_msg}")
    st.info("Pastikan file `classification_pipeline.pkl` dan `regression_pipeline.pkl` ada di folder `models/`.")
    st.stop()

# Sidebar - Input Form
st.sidebar.header("Input Data Mahasiswa")

with st.sidebar.form("prediction_form"):
    st.subheader("Akademik")
    cgpa = st.slider("CGPA", 5.0, 10.0, 8.0, 0.01)
    tenth_percentage = st.slider("Nilai Kelas 10 (%)", 50.0, 100.0, 75.0, 0.1)
    twelfth_percentage = st.slider("Nilai Kelas 12 (%)", 50.0, 100.0, 75.0, 0.1)
    backlogs = st.selectbox("Jumlah Backlogs", [0, 1, 2, 3, 4, 5])
    branch = st.selectbox("Jurusan", ["CSE", "ECE", "IT", "CE", "ME"])

    st.subheader("Keterampilan")
    coding_skill_rating = st.slider("Coding Skill (1-5)", 1, 5, 3)
    communication_skill_rating = st.slider("Communication Skill (1-5)", 1, 5, 3)
    aptitude_skill_rating = st.slider("Aptitude Skill (1-5)", 1, 5, 4)

    st.subheader("Pengalaman")
    projects_completed = st.slider("Projects Completed", 0, 8, 4)
    internships_completed = st.slider("Internships Completed", 0, 4, 2)
    hackathons_participated = st.slider("Hackathons Participated", 0, 6, 2)
    certifications_count = st.slider("Certifications Count", 0, 9, 3)

    st.subheader("Gaya Hidup")
    study_hours_per_day = st.slider("Jam Belajar/Hari", 0.0, 10.0, 4.0, 0.1)
    attendance_percentage = st.slider("Attendance (%)", 44.0, 100.0, 75.0, 0.1)
    sleep_hours = st.slider("Jam Tidur", 4.0, 9.0, 7.0, 0.1)
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)

    st.subheader("Profil")
    gender = st.selectbox("Gender", ["Male", "Female"])
    part_time_job = st.selectbox("Part-time Job?", ["No", "Yes"])
    internet_access = st.selectbox("Internet Access?", ["Yes", "No"])
    family_income_level = st.selectbox("Family Income", ["Low", "Medium", "High"])
    city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    extracurricular_involvement = st.selectbox("Extracurricular", ["Low", "Medium", "High"])

    submitted = st.form_submit_button("🔮 Prediksi!", use_container_width=True)

# Hasil Prediksi
if submitted:
    # Buat DataFrame dari input
    input_data = pd.DataFrame([{
        'cgpa': cgpa,
        'tenth_percentage': tenth_percentage,
        'twelfth_percentage': twelfth_percentage,
        'backlogs': backlogs,
        'study_hours_per_day': study_hours_per_day,
        'attendance_percentage': attendance_percentage,
        'projects_completed': projects_completed,
        'internships_completed': internships_completed,
        'coding_skill_rating': coding_skill_rating,
        'communication_skill_rating': communication_skill_rating,
        'aptitude_skill_rating': aptitude_skill_rating,
        'hackathons_participated': hackathons_participated,
        'certifications_count': certifications_count,
        'sleep_hours': sleep_hours,
        'stress_level': stress_level,
        'gender': gender,
        'part_time_job': part_time_job,
        'internet_access': internet_access,
        'family_income_level': family_income_level,
        'city_tier': city_tier,
        'extracurricular_involvement': extracurricular_involvement,
        'branch': branch,
    }])

    # Prediksi
    placement_pred = clf_pipeline.predict(input_data)[0]
    salary_pred = reg_pipeline.predict(input_data)[0]
    salary_pred = max(0, salary_pred)

    # Tampilkan Hasil
    st.subheader("Hasil Prediksi")

    col1, col2 = st.columns(2)

    with col1:
        if placement_pred == 1:
            st.markdown("""
            <div class='prediction-box placed'>
                <h2>✅ PLACED</h2>
                <p>Mahasiswa diprediksi <b>berhasil ditempatkan</b> di perusahaan.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='prediction-box not-placed'>
                <h2>❌ NOT PLACED</h2>
                <p>Mahasiswa diprediksi <b>belum berhasil ditempatkan</b>.</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='salary-box'>
            <h2>💰 {salary_pred:.2f} LPA</h2>
            <p>Estimasi gaji tahunan mahasiswa.</p>
        </div>
        """, unsafe_allow_html=True)

    # Ringkasan Input
    st.divider()
    st.subheader("Ringkasan Data Input")

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric("CGPA", cgpa)
        st.metric("Kelas 10", f"{tenth_percentage}%")
        st.metric("Kelas 12", f"{twelfth_percentage}%")
        st.metric("Backlogs", backlogs)

    with col_b:
        st.metric("Coding Skill", f"{coding_skill_rating}/5")
        st.metric("Communication", f"{communication_skill_rating}/5")
        st.metric("Aptitude", f"{aptitude_skill_rating}/5")

    with col_c:
        st.metric("Projects", projects_completed)
        st.metric("Internships", internships_completed)
        st.metric("Hackathons", hackathons_participated)
        st.metric("Certifications", certifications_count)

    with col_d:
        st.metric("Study Hours", f"{study_hours_per_day} jam")
        st.metric("Attendance", f"{attendance_percentage}%")
        st.metric("Sleep", f"{sleep_hours} jam")
        st.metric("Stress Level", f"{stress_level}/10")

    # Visualisasi Bar Chart - Profil Skill
    st.divider()
    st.subheader("Profil Keterampilan Mahasiswa")

    skill_names = ['CGPA', 'Coding', 'Communication', 'Aptitude',
                   'Projects', 'Internships', 'Hackathons']
    skill_values = [
        cgpa / 10 * 100,
        coding_skill_rating / 5 * 100,
        communication_skill_rating / 5 * 100,
        aptitude_skill_rating / 5 * 100,
        projects_completed / 8 * 100,
        internships_completed / 4 * 100,
        hackathons_participated / 6 * 100,
    ]

    colors = ['#28a745' if v >= 70 else '#ffc107' if v >= 40 else '#dc3545' for v in skill_values]

    fig = go.Figure(data=go.Bar(
        x=skill_names,
        y=skill_values,
        marker_color=colors,
        text=[f"{v:.0f}%" for v in skill_values],
        textposition='outside'
    ))

    fig.update_layout(
        yaxis=dict(title="Persentase (%)", range=[0, 110]),
        xaxis=dict(title="Kategori"),
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Hijau = Baik (≥70%) | Kuning = Cukup (40-69%) | Merah = Perlu Ditingkatkan (<40%)")

else:
    # Landing Page
    st.info("👈 Isi data mahasiswa di sidebar, lalu klik **Prediksi!** untuk melihat hasil.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🎯 Klasifikasi")
        st.write("Prediksi apakah mahasiswa akan **Placed** atau **Not Placed** berdasarkan profil akademik dan keterampilan.")
    with col2:
        st.subheader("💰 Regresi")
        st.write("Estimasi **besaran gaji (LPA)** yang akan diperoleh mahasiswa setelah lulus.")
    with col3:
        st.subheader("📊 Visualisasi")
        st.write("Lihat **bar chart** profil keterampilan mahasiswa dengan indikator warna.")

# Footer
st.divider()
st.caption("UTS Model Deployment | Junico | 2802518383")
