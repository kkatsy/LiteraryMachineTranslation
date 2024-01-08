# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image
import pytesseract


filename = "/Users/kk/Desktop/test7.png"
im = Image.open(filename)
new_filename = 'test7.tiff'
im.save(new_filename)
im = Image.open(new_filename)
extracted_text = pytesseract.image_to_string(im)
print(extracted_text)