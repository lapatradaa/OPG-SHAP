
from PIL import Image

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

image_path = 'static/uploads/shap2nd/cropped_shap_image_plot.png'
width, height = get_image_dimensions(image_path)
print(f"Width: {width}, Height: {height}")