# Image_Captioning
This project aims to analyze the content of images to generate accurate captions using a Transformer-based model.It is deployed on Streamlit, it provides a straightforward, user-friendly interface for captioning images in real time, making it available to all users.

## Technologies Used

- **Python**: The programming language used for building the application.
- **Streamlit**: A framework for creating web applications in Python.
- **Transformers Library**: A library by Hugging Face for working with pre-trained models for natural language processing and computer vision tasks.
- **PIL (Pillow)**: A Python Imaging Library used for opening, manipulating, and saving image files.

## Models and Architecture

This application utilizes two pre-trained models:

1. **BLIP (Bootstrapping Language-Image Pre-training)**:
   - **Model**: `Salesforce/blip-image-captioning-base`
   - **Architecture**: This model is based on the vision-language pre-training paradigm, which combines visual and textual representations, serves as the foundation for this model. By processing the input image and predicting descriptive text with a transformer-based architecture, it creates image captions.

2. **T5 (Text-to-Text Transfer Transformer)**:
   - **Model**: `google-t5/t5-base`
   - **Architecture**: T5 is a versatile text generation model that treats every NLP task as a text-to-text problem. In this project, it is employed for translating the generated captions into various languages.

## Project Details

The application is structured to provide an interactive user experience, featuring the following functionalities:

- **Image Upload**: Users can upload an image in JPG, JPEG, or PNG formats using a sidebar file uploader.
  
- **Caption Generation**: Upon uploading an image, the application generates a descriptive caption using the BLIP model. The captioning process involves:
  - Preprocessing the uploaded image.
  - Using the BLIP model to generate a caption based on the image content.
  
- **Display of Results**: The generated caption and its translations are displayed in the main application interface, providing users with clear and accessible output.

## Unit testing
- The file Caption_unit_test.py is created to perform unit test for the code. This unit test has been implemented to ensure the robustness of the image captioning feature. The test checks whether the caption generation process returns a valid output when provided with an image.

## Enhancements

- **Language Selection**: Users can select a target language (French, German, or English) for caption translation. The translation process involves:
  - Sending the generated English caption to the T5 model for translation into the selected language.This enhances the usability by supporting multiple languages , making captions available for wider audience.
  

## The decision to use multimodal pretrained models

Although the Flickr30k dataset is a well-known benchmark for image captioning tasks, I opted to utilize pre-trained models instead of training a model from scratch on this dataset. The rationale for this decision includes:

1. **Leverage State-of-the-Art Technology**: Pre-trained models like BLIP and T5 provide high-quality outputs that have been fine-tuned on extensive datasets. This approach allows for superior performance compared to training on a smaller dataset.

3. **Resource Efficiency**: Training a model on a dataset such as Flickr30k would require significant computational resources and time. Utilizing existing pre-trained models enabled rapid development and deployment, making the application efficient and accessible.
