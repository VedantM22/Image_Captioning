import unittest
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from App import caption_generation

processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def test_caption():
    img = Image.new('RGB',(150,150))
    caption = caption_generation(processor , model , img)
    assert isinstance(caption, str), "Caption should be a string."
    assert len(caption) >0, 'Caption should not be empty'
    

if __name__ == '__main__':
    test_caption()
    print('Passed the caption generation test')