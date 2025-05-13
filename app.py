import streamlit as st
import pandas as pd
import joblib 
import numpy as np
import re

st.set_page_config(page_title="Academics Insight Jaya jaya Institute", layout="wide")

# Load model Random Forest
@st.cache_resource
def load_rf_model():
    return joblib.load("model/model.pkl")

def normalisasi_dataset(df):
    df = df.copy()
    # Nama kolom sebelumnya
    original_columns = [
        'Marital status', 'Application mode', 'Application order', 'Course',
        'Daytime/evening attendance\t', 'Previous qualification',
        'Previous qualification (grade)', 'Nacionality',
        "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation", 'Admission grade',
        'Displaced', 'Educational special needs', 'Debtor',
        'Tuition fees up to date', 'Gender', 'Scholarship holder',
        'Age at enrollment', 'International',
        'Curricular units 1st sem (credited)',
        'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)',
        'Curricular units 1st sem (approved)',
        'Curricular units 1st sem (grade)',
        'Curricular units 1st sem (without evaluations)',
        'Curricular units 2nd sem (credited)',
        'Curricular units 2nd sem (enrolled)',
        'Curricular units 2nd sem (evaluations)',
        'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)',
        'Curricular units 2nd sem (without evaluations)', 'Unemployment rate',
        'Inflation rate', 'GDP', 'Target'
    ]
    
    # Fungsi pembersih nama kolom
    def clean_column(name):
        # Lowercase
        name = name.lower()
        # Ganti tab dan spasi dengan underscore
        name = re.sub(r'[\t\s]+', '_', name)
        # Hapus karakter selain alfanumerik dan underscore
        name = re.sub(r'[^\w]', '', name)
        # Hindari underscore ganda
        name = re.sub(r'_+', '_', name)
        # Hapus leading/trailing underscore
        name = name.strip('_')
        return name
    
    # Terapkan pada semua kolom
    cleaned_columns = [clean_column(col) for col in original_columns]

    df.columns = cleaned_columns
    float_cols = df.select_dtypes(include='float64').columns
    for col in float_cols:
        if (df[col] % 1 == 0).all():
            df[col] = df[col].astype(int)
            print(f"Kolom '{col}' telah dikonversi dari float ke int.")
        else:
            print(f"Kolom '{col}' tidak dikonversi karena memiliki nilai desimal.")

    # Mapping untuk setiap kolom kategorikal (key disesuaikan dengan nama kolom dataframe)
    mappings = {
        'marital_status': {
            1: 'single',
            2: 'married',
            3: 'widower',
            4: 'divorced',
            5: 'facto union',
            6: 'legally separated'
        },
        'application_mode': {
            1: '1st phase - general contingent',
            2: 'Ordinance No. 612/93',
            5: '1st phase - special contingent (Azores Island)',
            7: 'Holders of other higher courses',
            10: 'Ordinance No. 854-B/99',
            15: 'International student (bachelor)',
            16: '1st phase - special contingent (Madeira Island)',
            17: '2nd phase - general contingent',
            18: '3rd phase - general contingent',
            26: 'Ordinance No. 533-A/99, item b2) (Different Plan)',
            27: 'Ordinance No. 533-A/99, item b3 (Other Institution)',
            39: 'Over 23 years old',
            42: 'Transfer',
            43: 'Change of course',
            44: 'Technological specialization diploma holders',
            51: 'Change of institution/course',
            53: 'Short cycle diploma holders',
            57: 'Change of institution/course (International)'
        },
        'course': {
            33: 'Biofuel Production Technologies',
            171: 'Animation and Multimedia Design',
            8014: 'Social Service (evening attendance)',
            9003: 'Agronomy',
            9070: 'Communication Design',
            9085: 'Veterinary Nursing',
            9119: 'Informatics Engineering',
            9130: 'Equinculture',
            9147: 'Management',
            9238: 'Social Service',
            9254: 'Tourism',
            9500: 'Nursing',
            9556: 'Oral Hygiene',
            9670: 'Advertising and Marketing Management',
            9773: 'Journalism and Communication',
            9853: 'Basic Education',
            9991: 'Management (evening attendance)'
        },
        'daytimeevening_attendance': {
            0: 'evening',
            1: 'daytime'
        },
        'previous_qualification': {
            1: 'Secondary education',
            2: 'Higher education - bachelor\'s degree',
            3: 'Higher education - degree',
            4: 'Higher education - master\'s',
            5: 'Higher education - doctorate',
            6: 'Frequency of higher education',
            9: '12th year of schooling - not completed',
            10: '11th year of schooling - not completed',
            12: 'Other - 11th year of schooling',
            14: '10th year of schooling',
            15: '10th year of schooling - not completed',
            19: 'Basic education 3rd cycle (9th/10th/11th year) or equivalent',
            38: 'Basic education 2nd cycle (6th/7th/8th year) or equivalent',
            39: 'Technological specialization course',
            40: 'Higher education - degree (1st cycle)',
            42: 'Professional higher technical course',
            43: 'Higher education - master (2nd cycle)'
        },
        'nacionality': {
            1: 'Portuguese',
            2: 'German',
            6: 'Spanish',
            11: 'Italian',
            13: 'Dutch',
            14: 'English',
            17: 'Lithuanian',
            21: 'Angolan',
            22: 'Cape Verdean',
            24: 'Guinean',
            25: 'Mozambican',
            26: 'Santomean',
            32: 'Turkish',
            41: 'Brazilian',
            62: 'Romanian',
            100: 'Moldova (Republic of)',
            101: 'Mexican',
            103: 'Ukrainian',
            105: 'Russian',
            108: 'Cuban',
            109: 'Colombian'
        },
        'mothers_qualification': {
            1: 'Secondary Education - 12th Year of Schooling or Equivalent',
            2: 'Higher Education - Bachelor\'s Degree',
            3: 'Higher Education - Degree',
            4: 'Higher Education - Master\'s',
            5: 'Higher Education - Doctorate',
            6: 'Frequency of Higher Education',
            9: '12th Year of Schooling - Not Completed',
            10: '11th Year of Schooling - Not Completed',
            11: '7th Year (Old)',
            12: 'Other - 11th Year of Schooling',
            14: '10th Year of Schooling',
            18: 'General commerce course',
            19: 'Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent',
            22: 'Technical-professional course',
            26: '7th year of schooling',
            27: '2nd cycle of the general high school course',
            29: '9th Year of Schooling - Not Completed',
            30: '8th year of schooling',
            34: 'Unknown',
            35: 'Can\'t read or write',
            36: 'Can read without having a 4th year of schooling',
            37: 'Basic education 1st cycle (4th/5th year) or equivalent',
            38: 'Basic Education 2nd Cycle (6th/7th/8th Year) or Equivalent',
            39: 'Technological specialization course',
            40: 'Higher education - degree (1st cycle)',
            41: 'Specialized higher studies course',
            42: 'Professional higher technical course',
            43: 'Higher Education - Master (2nd cycle)',
            44: 'Higher Education - Doctorate (3rd cycle)'        
        },
        'fathers_qualification': {
            1: 'Secondary Education - 12th Year of Schooling or Equivalent',
            2: 'Higher Education - Bachelor\'s Degree',
            3: 'Higher Education - Degree',
            4: 'Higher Education - Master\'s',
            5: 'Higher Education - Doctorate',
            6: 'Frequency of Higher Education',
            9: '12th Year of Schooling - Not Completed',
            10: '11th Year of Schooling - Not Completed',
            11: '7th Year (Old)',
            12: 'Other - 11th Year of Schooling',
            13: '2nd year complementary high school course',
            14: '10th Year of Schooling',
            18: 'General commerce course',
            19: 'Basic Education 3rd Cycle (9th/10th/11th Year) or Equivalent',
            20: 'Complementary High School Course',
            22: 'Technical-professional course',
            25: 'Complementary High School Course - not concluded',
            26: '7th year of schooling',
            27: '2nd cycle of the general high school course',
            29: '9th Year of Schooling - Not Completed',
            30: '8th year of schooling',
            31: 'General Course of Administration and Commerce',
            33: 'Supplementary Accounting and Administration',
            34: 'Unknown',
            35: 'Can\'t read or write',
            36: 'Can read without having a 4th year of schooling',
            37: 'Basic education 1st cycle (4th/5th year) or equivalent',
            38: 'Basic Education 2nd Cycle (6th/7th/8th Year) or Equivalent',
            39: 'Technological specialization course',
            40: 'Higher education - degree (1st cycle)',
            41: 'Specialized higher studies course',
            42: 'Professional higher technical course',
            43: 'Higher Education - Master (2nd cycle)',
            44: 'Higher Education - Doctorate (3rd cycle)'        
        },
        'mothers_occupation': {
            0: 'Student',
            1: 'Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers',
            2: 'Specialists in Intellectual and Scientific Activities',
            3: 'Intermediate Level Technicians and Professions',
            4: 'Administrative staff',
            5: 'Personal Services, Security and Safety Workers and Sellers',
            6: 'Farmers and Skilled Workers in Agriculture, Fisheries and Forestry',
            7: 'Skilled Workers in Industry, Construction and Craftsmen',
            8: 'Installation and Machine Operators and Assembly Workers',
            9: 'Unskilled Workers',
            10: 'Armed Forces Professions',
            90: 'Other Situation',
            99: '(blank)',
            122: 'Health professionals',
            123: 'teachers',
            125: 'Specialists in information and communication technologies (ICT)',
            131: 'Intermediate level science and engineering technicians and professions',
            132: 'Technicians and professionals, of intermediate level of health',
            134: 'Intermediate level technicians from legal, social, sports, cultural and similar services',
            141: 'Office workers, secretaries in general and data processing operators',
            143: 'Data, accounting, statistical, financial services and registry-related operators',
            144: 'Other administrative support staff',
            151: 'personal service workers',
            152: 'sellers',
            153: 'Personal care workers and the like',
            171: 'Skilled construction workers and the like, except electricians',
            173: 'Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like',
            175: 'Workers in food processing, woodworking, clothing and other industries and crafts',
            191: 'cleaning workers',
            192: 'Unskilled workers in agriculture, animal production, fisheries and forestry',
            193: 'Unskilled workers in extractive industry, construction, manufacturing and transport',
            194: 'Meal preparation assistants'        
        },
        'fathers_occupation': {
            0: 'Student',
            1: 'Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers',
            2: 'Specialists in Intellectual and Scientific Activities',
            3: 'Intermediate Level Technicians and Professions',
            4: 'Administrative staff',
            5: 'Personal Services, Security and Safety Workers and Sellers',
            6: 'Farmers and Skilled Workers in Agriculture, Fisheries and Forestry',
            7: 'Skilled Workers in Industry, Construction and Craftsmen',
            8: 'Installation and Machine Operators and Assembly Workers',
            9: 'Unskilled Workers',
            10: 'Armed Forces Professions',
            90: 'Other Situation',
            99: '(blank)',
            101: 'Armed Forces Officers',
            102: 'Armed Forces Sergeants',
            103: 'Other Armed Forces personnel',
            112: 'Directors of administrative and commercial services',
            114: 'Hotel, catering, trade and other services directors',
            121: 'Specialists in the physical sciences, mathematics, engineering and related techniques',
            122: 'Health professionals',
            123: 'teachers',
            124: 'Specialists in finance, accounting, administrative organization, public and commercial relations',
            131: 'Intermediate level science and engineering technicians and professions',
            132: 'Technicians and professionals, of intermediate level of health',
            134: 'Intermediate level technicians from legal, social, sports, cultural and similar services',
            135: 'Information and communication technology technicians',
            141: 'Office workers, secretaries in general and data processing operators',
            143: 'Data, accounting, statistical, financial services and registry-related operators',
            144: 'Other administrative support staff',
            151: 'personal service workers',
            152: 'sellers',
            153: 'Personal care workers and the like',
            154: 'Protection and security services personnel',
            161: 'Market-oriented farmers and skilled agricultural and animal production workers',
            163: 'Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence',
            171: 'Skilled construction workers and the like, except electricians',
            172: 'Skilled workers in metallurgy, metalworking and similar',
            174: 'Skilled workers in electricity and electronics',
            175: 'Workers in food processing, woodworking, clothing and other industries and crafts',
            181: 'Fixed plant and machine operators',
            182: 'assembly workers',
            183: 'Vehicle drivers and mobile equipment operators',
            192: 'Unskilled workers in agriculture, animal production, fisheries and forestry',
            193: 'Unskilled workers in extractive industry, construction, manufacturing and transport',
            194: 'Meal preparation assistants',
            195: 'Street vendors (except food) and street service providers'        
        },
        'displaced': {
            0: 'no',
            1: 'yes'
        },
        'educational_special_needs': {
            0: 'no',
            1: 'yes'
        },
        'debtor': {
            0: 'no',
            1: 'yes'
        },
        'tuition_fees_up_to_date': {
            0: 'no',
            1: 'yes'
        },
        'gender': {
            0: 'female',
            1: 'male'
        },
        'scholarship_holder': {
            0: 'no',
            1: 'yes'
        },
        'international': {
            0: 'no',
            1: 'yes'
        },
        'target': {
            'Dropout': 'Dropout',
            'Enrolled': 'Enrolled',
            'Graduate': 'Graduate'
        }
    }
        
    for col, map_dict in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(map_dict).astype('object').fillna(df[col])

    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip().str.lower()

    df = df.drop(["target","educational_special_needs","international"], axis=1)

    return df

# Preprocessing
def preprocess_data(df):
    df = df.copy()
    
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

# Fungsi halaman prediksi
def predict_page():
    model = load_rf_model()

    st.markdown("<h1 style='text-align: center;'>📂 Prediksi Massal Dropout Mahasiswa</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload file CSV berisi data Mahasiswa", type=["csv"])
    
    if uploaded_file is not None:
        uploaded_file.seek(0)
        head = uploaded_file.read(1024).decode('utf-8')
        uploaded_file.seek(0)
        
        if ';' in head and ',' not in head:
            delimiter = ';'
        elif ',' in head and ';' not in head:
            delimiter = ','
        else:
            delimiter = ','

        raw_df = pd.read_csv(uploaded_file, sep=delimiter)
        st.subheader("📋 Data Mentah yang Diupload")
        st.dataframe(raw_df.head())
        normal_df = normalisasi_dataset(raw_df)
        try:
            X_processed = preprocess_data(normal_df)
            preds = model.predict(X_processed)
            probs = model.predict_proba(X_processed)
            preds = np.argmax(probs, axis=1)
            max_probs = np.max(probs, axis=1)

            label_map = {0: "graduate", 1: "enrolled", 2: "dropout"}
            normal_df['Predicted'] = [label_map.get(pred, "unknown") for pred in preds]
            normal_df['Probability'] = np.round(max_probs, 4)

            st.subheader("🎯 Hasil Prediksi")
            st.dataframe(normal_df[["marital_status","application_mode","application_order","course","daytimeevening_attendance","previous_qualification","previous_qualification_grade","nacionality","mothers_qualification","fathers_qualification","mothers_occupation","fathers_occupation","admission_grade","displaced","debtor","tuition_fees_up_to_date","gender","scholarship_holder","age_at_enrollment","curricular_units_1st_sem_credited","curricular_units_1st_sem_enrolled","curricular_units_1st_sem_evaluations","curricular_units_1st_sem_approved","curricular_units_1st_sem_grade","curricular_units_1st_sem_without_evaluations","curricular_units_2nd_sem_credited","curricular_units_2nd_sem_enrolled","curricular_units_2nd_sem_evaluations","curricular_units_2nd_sem_approved","curricular_units_2nd_sem_grade","curricular_units_2nd_sem_without_evaluations","unemployment_rate","inflation_rate","gdp","Predicted", "Probability"]])

            csv = normal_df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Unduh Hasil Prediksi", data=csv, file_name='hasil_prediksi.csv', mime='text/csv')

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")

# Fungsi halaman utama
def home_page():
    st.title('👥 Academics Insight - Jaya Jaya Institute')
    st.write('Selamat datang di homepage!')
    st.markdown("""
Aplikasi internal milik Jaya Jaya Institut yang dirancang khusus untuk membantu tim akademik dan manajemen dalam menganalisis performa mahasiswa secara lebih efektif dan berbasis teknologi.

Dengan alat ini, Anda dapat memprediksi kemungkinan seorang mahasiswa mengalami dropout berdasarkan berbagai faktor seperti status pembayaran, kepemilikan beasiswa, jurusan, serta kondisi akademik lainnya.
Model prediktif yang digunakan berbasis algoritma machine learning dan telah dilatih menggunakan data historis mahasiswa, sehingga mampu memberikan hasil prediksi yang akurat untuk mendukung intervensi dan bimbingan yang lebih tepat sasaran.

Gunakan menu di sisi kiri untuk mengunggah data mahasiswa dalam format CSV dan lihat hasil prediksi secara massal hanya dalam hitungan detik.
Prediksi ini dapat membantu Anda merancang strategi pendampingan, mengidentifikasi risiko dropout lebih awal, serta meningkatkan tingkat kelulusan institusi secara keseluruhan.

Terima kasih telah menggunakan Academic Insights — mari bersama wujudkan lingkungan belajar yang lebih inklusif, suportif, dan berkelanjutan.
    """)

# Sidebar navigasi
def toggle_burger():
    pages = {
        'Home': home_page,
        'Prediksi Massal': predict_page,
    }
    st.sidebar.title("Jaya jaya Institute")
    st.sidebar.image("logo.jpg", width=200)
    page = st.sidebar.selectbox('Pilih Halaman', list(pages.keys()))
    pages[page]()

if __name__ == '__main__':
    toggle_burger()
