from transformers import BlipProcessor , BlipForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PIL import Image
import streamlit as st
import os

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-base")
lang_model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")

# models

def caption_generation(processor , model , img):
    image_processor =  processor(images=img, return_tensors="pt", padding=True, truncation=True)
    output = model.generate(**image_processor , max_length = 100 , num_beams = 5 , temperature = 1 , top_k = 50 , top_p=0.9)
    caption_generation =  processor.decode(output[0], skip_special_tokens=True)
    return caption_generation

    
def language_translation(caption):
    translations = {}  # This should be a dictionary to store translations
    for lang in ['French', 'German']:
        text = f'translate English to {lang}: {caption}'
        input_tokens = tokenizer(text, return_tensors="pt").input_ids
        output_ids = lang_model.generate(input_tokens)  # Call the model's generate method
        translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        translations[lang] = translated_text  # Store translations in the dictionary
    return translations
    
    
# Streamlit
st.title("Let's caption your image")

st.sidebar.header('Upload the image and choose the language')

image_file = st.sidebar.file_uploader('Enter your image here', type=['jpg', 'jpeg', 'png'])

translations = {}

if image_file is not None:  # Check if an image file has been uploaded
    img = Image.open(image_file)  # Open the uploaded image
    
    img = img.resize((500,500))
    
    st.image(img)
    
    lang_options = ['','French', 'German', 'English']    # Select a language for caption generation
    caption_language = st.sidebar.selectbox('Select a language to generate captions in:', lang_options)
    
    if st.sidebar.button('Generate'):
        with st.spinner('generating your captions...'):
            if caption_language:
            
                caption = caption_generation(processor, model, img)  # Generate caption for the uploaded image
                
                translations = language_translation(caption)    # Generate translations

                translations['English'] = caption     # Include the English caption as well

                # Display the selected translation
                st.write(f'Captions in {caption_language} : {translations[caption_language].upper()}')
            else:
                st.warning('Please select a valid language')

else:
    st.warning("Please upload an image file.")

