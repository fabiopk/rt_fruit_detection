import numpy as np
from PIL import Image
import torchvision.transforms as T
import random
import os


class AddRandomBackground(object):
    """ Transform compatible with torchvision that adds a random backgrounds

    Arguments:
        threshold {[int, tuple]} -- threshold of average white (0 to 255) that will be replaced
        background_path {string} -- path of the possible backgrounds
    """

    def __init__(self, threshold, background_path='bgs'):

        assert isinstance(threshold, (int, tuple))
        if isinstance(threshold, int):
            self.threshold = (threshold, threshold)
        else:
            assert len(threshold) == 2
            self.threshold = threshold

    def __call__(self, sample):

        bg_file = random.choice(os.listdir(self.background_path))
        bg = Image.open(os.path.join(self.background_path, bg_file))
        h, w = sample.size[:2]
        bg = T.RandomCrop((h, w))(bg)

        img = np.array(sample)
        bgg = np.array(bg)

        random_threshold = random.randint(*self.threshold)

        mask = (img.mean(axis=2) <= random_threshold).reshape(h, w, 1)

        new_img = sample * mask + bgg * np.invert(mask)

        return Image.fromarray(new_img)


class RandomZoom(object):

    """ Transform compatile with torchvision to zoom in or out

        Arguments:
        zoom_range {[float]} -- multiplier to the original size (<1 = zoom out; >1 = zoom in)

    """

    def __init__(self, zoom_range):
        assert isinstance(zoom_range, (int, tuple))
        if isinstance(zoom_range, int):
            self.zoom_range = (-zoom_range, zoom_range)
        else:
            assert len(zoom_range) == 2
            self.zoom_range = zoom_range

    def __call__(self, sample):

        amount = random.randint(a=self.zoom_range[0], b=self.zoom_range[1])
        amount = 1 + (amount / 100)  # percentage
        h, w = sample.size[:2]
        nh, nw = int(h*amount), int(w*amount)
        sample = T.Resize((nh, nw))(sample)
        if amount > 1:
            sample = T.CenterCrop((h, w))(sample)
        else:
            canvas = np.ones((h, w, 3), dtype='uint8') * 255
            h_bound = (h - nh) // 2
            w_bound = (w - nw) // 2
            h_end = h_bound + nh
            w_end = w_bound + nw
            canvas[h_bound:h_end, w_bound:w_end] = np.array(sample)
            sample = Image.fromarray(canvas)

        return sample
