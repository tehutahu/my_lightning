import io

import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    Source: (https://www.tensorflow.org/tensorboard/image_summaries)."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    figure.savefig(buf, format="png")
    buf.seek(0)
    image = Image.open(buf)
    img_ar = np.array(image)
    image = ToTensor()(img_ar)
    return image
