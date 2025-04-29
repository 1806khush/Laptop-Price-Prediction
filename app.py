import streamlit as st
import numpy as np
import pickle
import joblib
from sklearn.ensemble import GradientBoostingRegressor

# This must be the first Streamlit command
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
)

# Background image function - add after set_page_config
def add_bg_from_url():
    background_image_url = "https://w0.peakpx.com/wallpaper/544/540/HD-wallpaper-macbook-pro-apple-technology.jpg"
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{background_image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.85);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function after set_page_config
add_bg_from_url()

st.markdown("""
<style>
    /* Global Styles */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #2c3e50;
        background-color: #f9fafc;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin: 1.5rem 0 2.5rem 0;
        padding-bottom: 1.2rem;
        border-bottom: 3px solid #3498db;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1.8rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e8eaed;
        position: relative;
    }
    
    .section-header::after {
        content: "";
        position: absolute;
        left: 0;
        bottom: -2px;
        width: 80px;
        height: 2px;
        background-color: #3498db;
    }
    
    /* Prediction Styles */
    .prediction-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
        font-weight: 600;
        text-align: center;
    }
    
    .prediction-result {
        font-size: 2.2rem;
        font-weight: bold;
        color: #27ae60;
        background: linear-gradient(145deg, #eafaf1, #e0f5e9);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.2rem 0;
        box-shadow: 0 4px 10px rgba(39, 174, 96, 0.1);
        border-left: 5px solid #27ae60;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.7rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2980b9, #2573a7);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    .stButton>button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Info Boxes */
    .info-box {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .info-box::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 5px;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        opacity: 0.7;
    }
    
    /* Success box variant */
    .success-box {
        background-color: #f0fff4;
        border-left: 5px solid #2ecc71;
    }
    
    .success-box::before {
        background: linear-gradient(90deg, #2ecc71, #27ae60);
    }
    
    /* Warning box variant */
    .warning-box {
        background-color: #fffaed;
        border-left: 5px solid #f39c12;
    }
    
    .warning-box::before {
        background: linear-gradient(90deg, #f39c12, #e67e22);
    }
    
    /* Card Layout */
    .card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border-top: 4px solid #3498db;
    }
    
    /* Input Styling */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 0.6rem 1rem;
        font-size: 1rem;
        transition: all 0.3s;
    }
    
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
    }
    
    /* Select Box Styling */
    .stSelectbox>div>div>div {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    /* Checkbox Styling */
    .stCheckbox>div>div>label>div {
        background-color: #3498db;
    }
    
    /* Metric Styling */
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Table Styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .stDataFrame table {
        border-collapse: separate;
        border-spacing: 0;
    }
    
    .stDataFrame th {
        background-color: #f4f6f9;
        padding: 0.8rem 1rem;
        text-align: left;
        font-weight: 600;
        border-top: 1px solid #e0e0e0;
        border-bottom: 1px solid #e0e0e0;
    }
    
    .stDataFrame td {
        padding: 0.8rem 1rem;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .stDataFrame tr:nth-child(even) {
        background-color: #f9fafc;
    }
    
    /* Custom Progress Bar */
    .progress-container {
        width: 100%;
        height: 12px;
        background-color: #e0e0e0;
        border-radius: 6px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        border-radius: 6px;
        transition: width 0.5s ease-in-out;
    }
    
    /* Responsive adjustments */
    @media screen and (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .prediction-header {
            font-size: 1.5rem;
        }
        
        .prediction-result {
            font-size: 1.8rem;
            padding: 1rem;
        }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .info-box, .card {
            background-color: #1f2937;
            color: #e5e7eb;
        }
        
        .section-header {
            border-bottom-color: #374151;
        }
        
        .stDataFrame th {
            background-color: #1f2937;
        }
        
        .stDataFrame tr:nth-child(even) {
            background-color: #262f3c;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_data():
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    dataset = pickle.load(open('dataset.pkl', 'rb'))
    return pipe, dataset

try:
    pipe, dataset = load_data()
    load_success = True
except Exception as e:
    st.error(f"Error loading model files: {e}")
    load_success = False

# Main Application
st.markdown("<h1 class='main-header'>💻 Laptop Price Predictor</h1>", unsafe_allow_html=True)

# Brief Introduction
st.markdown("""
<div class="info-box">
    <p>This tool helps you predict laptop prices based on specifications. Enter the details below to get an estimated price.</p>
</div>
""", unsafe_allow_html=True)

if load_success:
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<p class='section-header'>Brand & Type</p>", unsafe_allow_html=True)
        company = st.selectbox('Brand', dataset['Company'].unique(), help="Select the laptop manufacturer")
        type = st.selectbox('Type', dataset['TypeName'].unique(), help="Select the type of laptop")
        
        st.markdown("<p class='section-header'>Performance</p>", unsafe_allow_html=True)
        ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64], help="Select the amount of RAM")
        cpu = st.selectbox('CPU', dataset['Cpu brand'].unique(), help="Select the CPU manufacturer and model")
        gpu = st.selectbox('GPU', dataset['Gpu brand'].unique(), help="Select the graphics card")
        
        st.markdown("<p class='section-header'>Operating System</p>", unsafe_allow_html=True)
        os = st.selectbox('OS', dataset['os'].unique(), help="Select the operating system")
        
    with col2:
        st.markdown("<p class='section-header'>Physical Features</p>", unsafe_allow_html=True)
        weight = st.number_input('Weight of the Laptop (kg)', min_value=0.5, max_value=5.0, value=1.5, step=0.1, 
                                help="Enter the weight of the laptop in kilograms")
        touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'], help="Select whether the laptop has a touchscreen")
        
        st.markdown("<p class='section-header'>Display</p>", unsafe_allow_html=True)
        screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.0, 0.1, 
                              help="Select the screen size in inches")
        resolution = st.selectbox('Screen Resolution', 
                               ['1366x768', '1600x900', '1920x1080', '2304x1440', 
                                '2560x1440', '2560x1600', '2880x1800', '3200x1800', '3840x2160'],
                               help="Select the screen resolution")
        
        st.markdown("<p class='section-header'>Storage</p>", unsafe_allow_html=True)
        col_hdd, col_ssd = st.columns(2)
        with col_hdd:
            hdd = st.selectbox('HDD (GB)', [0, 128, 256, 512, 1024, 2048], help="Select the HDD capacity")
        with col_ssd:
            ssd = st.selectbox('SSD (GB)', [0, 8, 128, 256, 512, 1024], help="Select the SSD capacity")
    
    # Summary box before prediction
    st.markdown("""
    <div class="info-box">
        <p><strong>Configuration Summary:</strong> {0} {1} with {2}GB RAM, {3} processor, {4} graphics, 
        {5}GB HDD + {6}GB SSD, {7}" display ({8}), {9} OS {10}</p>
    </div>
    """.format(
        company, type, ram, cpu, gpu, hdd, ssd, screen_size, resolution, os,
        ", with touchscreen" if touchscreen == "Yes" else ""
    ), unsafe_allow_html=True)
    
    # Prediction button with centering
    col_empty1, col_button, col_empty2 = st.columns([1, 2, 1])
    with col_button:
        predict_button = st.button('Predict Price')
    
    # When Predict button is clicked
    if predict_button:
        with st.spinner('Calculating price...'):
            # Process the data for prediction
            if touchscreen == 'Yes':
                touchscreen = 1
            else:
                touchscreen = 0
                
            X_res = int(resolution.split('x')[0])
            Y_res = int(resolution.split('x')[1])
            ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
            
            # Create the query and predict
            query = np.array([company, type, ram, weight, touchscreen, ppi, cpu, hdd, ssd, gpu, os])
            query = query.reshape(1, 11)
            
            # Get prediction and convert to currency format
            predicted_price = int(np.exp(pipe.predict(query)[0]))
            formatted_price = "₹{:,}".format(predicted_price)
            
            # Display the prediction with styling
            st.markdown(f"<h2 class='prediction-header'>Predicted Price:</h2>", unsafe_allow_html=True)
            st.markdown(f"<div class='prediction-result'>{formatted_price}</div>", unsafe_allow_html=True)
            
            # Provide some context for the prediction
            if predicted_price < 20000:
                price_category = "budget"
            elif predicted_price < 40000:
                price_category = "entry-level"
            elif predicted_price < 50000:
                price_category = "mid-range"
            elif predicted_price < 80000:
                price_category = "high-end"
            else:
                price_category = "premium"
                
            st.info(f"This configuration falls into the {price_category} category of laptops.")
else:
    st.warning("Please make sure the model files 'pipe.pkl' and 'dataset.pkl' are available in the same directory as this script.")
    
# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #ddd; color: #777;">
    <p>Laptop Price Predictor | Machine Learning Model</p>
</div>
""", unsafe_allow_html=True)