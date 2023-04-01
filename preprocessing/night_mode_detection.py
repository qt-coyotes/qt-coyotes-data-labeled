from PIL import Image, ImageStat
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import as_completed, ProcessPoolExecutor

# Based on: https://stackoverflow.com/questions/20068945/detect-if-image-is-color-grayscale-or-black-and-white-using-python
def detect_image_color(image, data_path, size=200, MSE_cutoff=7, adjust_bias=True):
    filename = str(data_path / image["file_name"])
    try:
        pil_img = Image.open(filename)
    except FileNotFoundError as e:
        image["is_color"] = None
        return image, 0.0
    bands = pil_img.getbands()

    if bands == ('R','G','B') or bands== ('R','G','B','A'):
        small_img = pil_img.resize((size, size))
        SSE = 0
        bias = [0, 0, 0]

        if adjust_bias:
            bias = ImageStat.Stat(small_img).mean[:3]
            bias = [b - sum(bias) / 3 for b in bias]

        for pixel in small_img.getdata():
            mu = sum(pixel) / 3 # Mean of all color channels of this pixel

            # Sum across all color channels in this pixel and across all pixels 
            # For each color channel in this pixel, find the squared difference between the value, the mean of all color channels, and the bias.
            SSE += sum((pixel[i] - mu - bias[i]) * (pixel[i] - mu - bias[i]) for i in [0,1,2])
        
        # Find mean squared error across entire image by dividing SSE by size of image
        MSE = float(SSE) / (size * size)

        # A few very dark day time photos in this range are marked as grayscale and need special handling
        # NOTE: If you change the size or adjust_bias the values of these photos will change and they may no longer fall in this range
        # which could cause some false classifications.
        if (MSE > 2 and MSE < 6): 
            image["is_color"] = True
            return image, MSE

        if MSE <= MSE_cutoff:
            image["is_color"] = False
            return image, MSE
        else:
            image["is_color"] = True
            return image, MSE


def main():

    with open(Path(MERGED_PATH), "r") as f:
        coco = json.load(f)

    images = coco["images"]

    classified_images = []
    MSEs = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for image in images:
            future = executor.submit(
                detect_image_color, image, DATA_ROOT_PATH
            )
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures)):
            classified_image, MSE = future.result()
            classified_images.append(classified_image)
            MSEs.append(MSE)

    coco["images"] = classified_images

    with open(MERGED_PATH, "w") as f:
        json.dump(coco, f, indent=4)
        

if __name__ == '__main__':
    MERGED_PATH = Path("../mange-classifier/data/qt-coyotes-merged.json")
    DATA_ROOT_PATH = Path("../mange-classifier/data")

    main()