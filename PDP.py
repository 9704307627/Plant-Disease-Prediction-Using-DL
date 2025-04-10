import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import google.generativeai as genai
from streamlit_option_menu import option_menu
import os

# Page icon
icon = Image.open(r"C:\Users\K. Lathika\Downloads\Plant Diesease Prediction\Logo.jpg")

# Page configuration
st.set_page_config(
    page_title="AI Based Plant Health Monitoring System",
    page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("<h2 style='text-align: center; color: #000080;'>Ramachandra College of Engineering</h2>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #BDB76B;'>Department of Artificial Intelligence & Data Science</h2>", unsafe_allow_html=True)
st.text("")
st.text("")

# Page Styling
background_image_path = r"C:\Users\K. Lathika\Downloads\Plant Diesease Prediction\Logo.jpg"

st.markdown(
    f"""
    <style>
    body {{
        background-image: url('{background_image_path}');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    .header-title {{
        font-size: 35px;
        font-weight: medium;
        color: #708090;
        text-align: left;
        margin-bottom: 30px;
    }}
    .emotion-text {{
        font-size: 24px;
        font-weight: bold;
        color: #4169e1;
        text-align: center;
        margin-bottom: 20px;
    }}
    .song-info {{
        font-size: 18px;
        color: #008080;
        text-align: center;
        margin-bottom: 20px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

#st.snow()

with st.sidebar:
    st.sidebar.image(icon, use_container_width=True)  # Updated parameter
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Project Details", "Contact"],
        icons=["house", "book", "envelope"],
        menu_icon="cast",
        default_index=0,
    )

# Home Section
if selected == "Home":
    # Language selection dropdown
    language = st.selectbox(
        label="Select Language",
        options=["English", "తెలుగు", "हिंदी"],
        index=0
    )

    # Translate static text based on selected language
    def get_translated_text(language):
        translations = {
            "తెలుగు": {
                "title": "AI ఆధారిత మొక్కల ఆరోగ్య పర్యవేక్షణ వ్యవస్థ 🌱",  # Updated title
                "description": "ఆకు యొక్క చిత్రాన్ని అప్‌లోడ్ చేయండి, మరియు మోడల్ రోగాన్ని (ఏదైనా ఉంటే) అంచనా వేస్తుంది లేదా మొక్క ఆరోగ్యంగా ఉందని నిర్ధారిస్తుంది.",
                "upload_text": "ఆకు యొక్క చిత్రాన్ని అప్‌లోడ్ చేయండి...",
                "classifying_text": "వర్గీకరిస్తోంది...",
                "fetching_text": "సిఫార్సులను తెస్తోంది...",
                "recommendations_title": "వివరణాత్మక సిఫార్సులు:",
                "prediction_text": "అంచనా: {} (నమ్మకం: {:.2f}%)",
            },
            "हिंदी": {
                "title": "AI आधारित पौध स्वास्थ्य निगरानी प्रणाली 🌱",  # Updated title
                "description": "पत्ते की एक छवि अपलोड करें, और मॉडल रोग (यदि कोई हो) की भविष्यवाणी करेगा या पुष्टि करेगा कि पौधा स्वस्थ है।",
                "upload_text": "पत्ते की एक छवि अपलोड करें...",
                "classifying_text": "वर्गीकरण कर रहा है...",
                "fetching_text": "सिफारिशें लाई जा रही हैं...",
                "recommendations_title": "विस्तृत सिफारिशें:",
                "prediction_text": "भविष्यवाणी: {} (आत्मविश्वास: {:.2f}%)",
            },
            "English": {
                "title": "AI Based Plant Health Monitoring System 🌱",  # Updated title
                "description": "Upload an image of a leaf, and the model will predict the disease (if any) or confirm that the plant is healthy.",
                "upload_text": "Upload an image of a leaf...",
                "classifying_text": "Classifying...",
                "fetching_text": "Fetching recommendations...",
                "recommendations_title": "Detailed Recommendations:",
                "prediction_text": "Prediction: {} (Confidence: {:.2f}%)",
            }
        }
        return translations[language]

    # Function to translate text using Gemini
    def translate_text(text, target_language):
        try:
            prompt = f"Translate the following text into {target_language}: {text}"
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.warning(f"Translation failed: {e}. Using original text.")
            return text

    # Get translated text based on selected language
    translated_text = get_translated_text(language)

    # Set the title based on the selected language
    st.title(translated_text["title"])
    st.write(translated_text["description"])

    # Initialize Gemini
    try:
        genai.configure(api_key="AIzaSyCrDTBASuyHG5qujYGiVe15dKrg2F1v04U")  # Replace with your Gemini API key
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-002')
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {e}")
        st.stop()

    # Load the saved model
    @st.cache_resource  # Cache the model for faster loading
    def load_tomato_model():
        try:
            return load_model('tomato_disease_model.h5')
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

    model = load_tomato_model()

    # Define class indices (replace with your actual class indices)
    class_indices = {
        0: 'Tomato___Bacterial_spot',
        1: 'Tomato___Early_blight',
        2: 'Tomato___Late_blight',
        3: 'Tomato___Leaf_Mold',
        4: 'Tomato___Septoria_leaf_spot',
        5: 'Tomato___Spider_mites Two-spotted_spider_mite',
        6: 'Tomato___Target_Spot',
        7: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        8: 'Tomato___Tomato_mosaic_virus',
        9: 'Tomato___healthy'
    }

    # Function to load and preprocess an image
    def load_and_preprocess_image(image_path, target_size=(224, 224)):
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array

    # Function to predict the class of an image
    def predict_image_class(model, image_path, class_indices):
        preprocessed_img = load_and_preprocess_image(image_path)
        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices[predicted_class_index]
        confidence = np.max(predictions) * 100  # Confidence level in percentage
        return predicted_class_name, confidence

    # Function to get detailed recommendations from Gemini
    def get_detailed_recommendations(disease, language):
        try:
            prompt = f"""
            Provide detailed information about {disease} in tomato plants, including:
            1. About the Disease
            2. Causes of the Disease
            3. Preventions
            4. Pesticides (Inorganic and organic pesticides, including quantities and application methods)
            5. Precautions (How to prevent the disease from occurring)
            Translate the response into {language}.
            """
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Failed to fetch recommendations: {e}")
            return "Recommendations not available."

    # File uploader
    uploaded_file = st.file_uploader(translated_text["upload_text"], type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Save the uploaded file temporarily
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Make a prediction
            st.write(translated_text["classifying_text"])
            predicted_class, confidence = predict_image_class(model, "temp_image.jpg", class_indices)
            
            # Translate the predicted class name into the selected language
            translated_class = translate_text(predicted_class, language)
            st.success(translated_text["prediction_text"].format(translated_class, confidence))

            # Get detailed recommendations from Gemini
            st.write(translated_text["fetching_text"])
            recommendations = get_detailed_recommendations(predicted_class, language)
            st.subheader(translated_text["recommendations_title"])
            st.write(recommendations)

        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists("temp_image.jpg"):
                os.remove("temp_image.jpg")

# Project Details Section
elif selected == "Project Details":
    st.markdown("<h2 class='sider-title' style='color: SlateGray;'>Project Details</h2>", unsafe_allow_html=True)
    st.write("")
    st.markdown("<h3 class='sider-title' style='color: black;'>Title:</h3>", unsafe_allow_html=True)
    st.write("")
    st.write("AI-Based Plant Health Monitoring System")
    st.markdown("<h3 class='sider-title' style='color: black;'>About the Project:</h3>", unsafe_allow_html=True)
    st.write("")
    st.write("""
    The *AI-Based Plant Health Monitoring System* is an innovative solution designed to revolutionize agriculture by leveraging artificial intelligence (AI) and computer vision technologies. 
    The system aims to detect and diagnose plant health issues, such as diseases, nutrient deficiencies, and environmental stresses, through advanced image recognition techniques. 
    By analyzing visual characteristics of plants—such as color, texture, and leaf structure—the system provides real-time insights into crop health, enabling farmers and agronomists to take proactive measures to ensure optimal crop yield and sustainability.
    """)
    st.write("")
    st.write("""
    ### Key Features:
    1. *Disease Detection*: Identifies common plant diseases like blight, bacterial spot, and powdery mildew.
    2. *Nutrient Deficiency Detection*: Detects symptoms of nutrient imbalances, such as yellowing leaves or stunted growth.
    3. *Environmental Stress Detection*: Identifies signs of stress caused by factors like drought or excessive sunlight.
    4. *Real-Time Alerts and Recommendations*: Provides actionable advice to address detected issues.
    5. *User-Friendly Interface*: Accessible on web and mobile platforms for easy use by farmers and researchers.
    """)
    st.write("")
    st.write("""
    ### Why It Matters:
    - *Improves Crop Yield*: Early detection of issues helps farmers take timely action, preventing crop losses.
    - *Promotes Sustainability*: Reduces the need for excessive pesticides and fertilizers, minimizing environmental impact.
    - *Empowers Farmers*: Provides farmers with the tools to make informed decisions and improve their livelihoods.
    """)
    st.write("")
    st.write("""
    ### How It Works:
    1. *Upload an Image*: Farmers upload a picture of a plant’s leaves.
    2. *AI Analysis*: The system uses AI to analyze the image and detect health issues.
    3. *Get Results*: Farmers receive a diagnosis and actionable recommendations in real time.
    """)
    st.write("")
    st.write("""
    ### Future Goals:
    - Expand to support more crops and languages.
    - Integrate with IoT devices for automated field monitoring.
    - Provide advanced analytics for long-term crop management.
    """)
    st.write("")
    image = r"C:\Users\K. Lathika\Downloads\Plant Diesease Prediction\Logo.jpg"
    image = Image.open(image)
    st.image(image, caption="AI-Based Plant Health Monitoring System", width=500, use_container_width=True)  # Updated parameter

# Contact Section
elif selected == "Contact":
    st.markdown("<h2 class='sider-title' style='color: SlateGray;'>Project Team</h2>", unsafe_allow_html=True)
    st.text("")

    # Team member details
    team_members = [
        {
            "name": "Challagolla Ramya Sree",
            "Roll_Number": "21ME1A5407",
            "Contact": "8919123783",
            "Mail": "ramyachallagolla79@gmail.com",
            "image_path": r"C:\Users\K. Lathika\Downloads\Plant Diesease Prediction\ramya.jpg",
            "role": "Team Lead"
        },
        {
            "name": "Kanakam Lathika",
            "Roll_Number": "21ME1A5423",
            "Contact": "9704307627",
            "Mail": "lathikalilly5459@gmail.com",
            "image_path": r"C:\Users\K. Lathika\Downloads\Plant Diesease Prediction\lilly.jpg",
            "role": "Member"
        },
        {
            "name": "Dulla Ravi Tejasri",
            "Roll_Number": "21ME1A5415",
            "Contact": "9182360110",
            "Mail": "satishd11223@gmail.com",
            "image_path": r"C:\Users\K. Lathika\Downloads\Plant Diesease Prediction\ravi.jpg",
            "role": "Member"
        },
        {
            "name": "Yadlapalli Vyshnavi",
            "Roll_Number": "21ME1A5465",
            "Contact": "9014660399",
            "Mail": "chowdaryvyshnavi285@gmail.com",
            "image_path": r"C:\Users\K. Lathika\Downloads\Plant Diesease Prediction\vysh.jpg",
            "role": "Member"
        }
    ]
        # Display each team member's details horizontally
    for member in team_members:
        # Create columns for image and details
        col1, col2 = st.columns([1, 3])  # Adjust the ratio as needed

        with col1:
            # Display the resized image
            image = Image.open(member["image_path"]).resize((150, 200))  # Resize to 150x200 pixels
            st.image(image, caption=member["name"], width=150)  # Set a fixed width for the image

        with col2:
            # Display the details
            st.markdown(f"*Role:* {member['role']}")
            st.markdown(f"*Name:* {member['name']}")
            st.markdown(f"*Roll Number:* {member['Roll_Number']}")
            st.markdown(f"*Contact:* {member['Contact']}")
            st.markdown(f"*Mail:* {member['Mail']}")

        # Add a divider for better spacing
        st.markdown("---")

    # Display team member details with images side by side
    #col1, col2, col3, col4 = st.columns(4)

    #for i, member in enumerate(team_members):
      #  with locals()[f"col{i+1}"]:
       #     st.write(f"*Name:* {member['name']}")
        #    st.write(f"*Roll Number:* {member['Roll_Number']}")
         #   st.write(f"*Contact:* {member['Contact']}")
          #  st.write(f"*Mail:* {member['Mail']}")