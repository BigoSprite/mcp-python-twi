from PIL import Image as PILImage
import io
import base64


def create_image_from_bytes(image_bytes, mode='RGB', size=(358, 441)):
    """
    从字节数据创建PIL图像

    Args:
        image_bytes: 图像的字节数据
        mode: 图像模式，如 'RGB', 'RGBA', 'L' (灰度)
        size: 图像尺寸 (width, height)
    """
    image = PILImage.frombytes(mode, size, image_bytes)
    return image


def create_image_from_base64(base64_string):
    """
    从base64字符串创建PIL图像
    """

    # 解码base64
    image_bytes = base64.b64decode(base64_string)

    # 使用BytesIO创建图像
    image_data = io.BytesIO(image_bytes)
    image = PILImage.open(image_data)

    return image


def create_image_from_numpy(numpy_array, mode='RGB'):
    """
    从numpy数组创建PIL图像
    """
    import numpy as np
    image = PILImage.fromarray(numpy_array.astype('uint8'), mode)
    return image