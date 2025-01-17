import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np



st.markdown(
    """
    <style>
    body {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🍊CitNet7 - Citrus Plant Disease Detection ")

def model_prediction(test_image):
    model = tf.keras.models.load_model('CitNet7.h5')
    image = keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
    input_arr = keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    result = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return result, confidence


st.sidebar.title('CitNet7 - Dashboard')
app_mode = st.sidebar.selectbox('Please select a page', ['Home', 'Disease Detection'])

if app_mode == "Home":
    # st.title("Welcome to CitNet7 - Citrus Plant Disease Detection")
    st.image("citrus.jpeg", 
             caption="Citrus Plant", use_container_width=True)
    st.write("""
        This application helps detect diseases in citrus plants using a Convolutional Neural Network (CNN).
        Upload an image of a citrus plant leaf to classify it as healthy or infected with Citrus Canker, Citrus Greening, or Leaf Miner.
    """)
        st.write("""
______________________________________________________________________________________________________
### How to use the application

1. Navigate to the **Disease Detection** page.
2. Upload an image of the leaf from the citrus plant.
3. Wait for the model to classify the disease and display the result.


        """)




    st.write("""
______________________________________________________________________________________________________
### Usage/Examples

Use infected Images of:
- Citrus - Canker
- Citus - Leaf-Minor
- Citus - Greening (HLB)
- Healthy Citrus Plant
______________________________________________________________________________________________________
        """)

    st.markdown("""
        ### Contact Us:
        - **Email:** [ranasheraz.202101902@gcuf.edu.pk](mailto:ranasheraz.202101902@gcuf.edu.pk)
        - **LinkedIn:** [Rana Sheraz Ahmad](https://www.linkedin.com/in/sherazahmadd/)
        - **GitHub:** [Rana Sheraz Ahmad](https://github.com/SherazAhmadd)
             __________________________________________________________
        - **Email:** [m.tahirulqamar@hotmail.com](mailto:m.tahirulqamar@hotmail.com)
        - **LinkedIn:** [Muhammad Tahir ul Qamar](https://www.linkedin.com/in/muhammad-tahir-ul-qamar-9999057a/)
        - **GitHub:** [Muhammad Tahir ul Qamar](https://github.com/tahirulqamar)
        ______________________________________________________________________________________________________
        """)
# Define the class names
class_name = {0: "Canker", 1: "Greening", 2: "Healthy", 3: "Leaf Miner"}

# for predicting the class of disease

                

if app_mode == 'Disease Detection':
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=160)  # Resize the image using the `width` parameter

        # if(st.button("show image")):
        #     st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        if(st.button("Predict")):
            with st.spinner('Model is Analyzing the disease...'):
                st.write("The prediction is:")
                result, confidence = model_prediction(uploaded_file)
                # define class
                st.success("🟢 Predicted Disease: {}".format(class_name[int(result)]))
                # Display additional suggestions or actions based on the prediction
                st.write("### Recommended Actions:")
                if class_name[result] == "Canker":
                    st.warning("🔴 **Canker Detected**: Remove infected leaves and apply a copper-based fungicide.")
                elif class_name[result] == "Greening":
                    st.warning("🔴 **Greening Detected**: Remove the infected plant to prevent spreading.")
                elif class_name[result] == "Healthy":
                    st.success("🟢 **Healthy Plant**: Keep monitoring and maintain good care.")
                elif class_name[result] == "Leaf Miner":
                    st.warning("🟠 **Leaf Miner Detected**: Prune infected leaves and use organic sprays.")
    else:
        st.info("Please upload an image to start the detection process.")


