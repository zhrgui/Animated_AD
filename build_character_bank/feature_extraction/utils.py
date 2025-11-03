import json
import numpy as np
import PIL.Image as Image
from scipy.ndimage import binary_fill_holes, binary_erosion

def load_json(file_path):
    with open(file_path, 'r') as infile:
        return json.load(infile)

def crop_image(image, bbox, mask=None, output_filename=None):
    x_min, y_min, x_max, y_max = map(int, bbox)
    width, height = image.size
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)

    if mask is None:
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
    else:
        np_image = np.array(image)
        cropped_image = np_image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]
        white_background = np.full_like(cropped_image, 255)

        mask_bool = cropped_mask.astype(bool)
        mask_bool = binary_erosion(mask_bool, iterations=2)
        mask_bool = binary_fill_holes(mask_bool)

        if cropped_image.ndim == 3 and cropped_image.shape[2] > 1:
            mask_bool = mask_bool[..., None]

        result_image = np.where(mask_bool, cropped_image, white_background)
        result_image_uint8 = result_image.astype(np.uint8)
        cropped_image = Image.fromarray(result_image_uint8)

    if output_filename is not None:
        cropped_image.save(output_filename)
    return cropped_image

def rgba_to_rgb_alpha_to_white(image):
    if image.mode != 'RGBA':
        return image.convert("RGB")

    image_array = np.array(image)
    rgb_array = image_array[:, :, :3]
    alpha_channel = image_array[:, :, 3]

    alpha_mask = alpha_channel != 255
    rgb_array[alpha_mask] = [255, 255, 255]
    return Image.fromarray(rgb_array)
