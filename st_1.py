import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('cat_dog.h5')

def classify_image(image_path):

    #img = image.load_img(image_path,target_size=(160,160))
    img_array = image.img_to_array(image_path)
    img_array = preprocess_input(img_array)
    img_array = tf.expand_dims(img_array,0)
    pred = model.predict(img_array)
    if pred[0] > 0.5:
        return 'dog'
    else:
        return 'cat'

    

def main():
    st.title("Cat & Dog Classifier")

    input_image = st.file_uploader("Select Input Image", type=['jpg','jpeg','png'])

    if input_image is not None:

        image = Image.open(input_image)
        image = image.resize((160,160))

        # Display the original input image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        label = classify_image(image)

        st.write(f'Prediction : {label}')
        

    
if __name__ == "__main__":
    main()