from PIL import Image
import numpy as np

def draw_image(data):
    # reform data into an array
    drawdata = np.reshape(data, (28,28))

    im = Image.fromarray(drawdata).convert('RGB')
    im.save('my.png')
