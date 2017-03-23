from unittest import TestCase

from src.data import Data

"""

Class that contains tests for a
"""

class TestData(TestCase):
    def test_train_test_split(self):
        pass

    def test_resize_img(self):
        data = Data("this")
        data.load_images()
        data.resize_img([100,100])
        data.save_pickle()

    def test_load_pickle(self):
        data = Data("this")
        data.load_pickle("data.pickle")
        self.assertEqual(data.class_names, ["Vanilla Ice Cream", "Pizza", "Salad", "Fish and Chips"])

    def test_load_images(self):
        data = Data("this")
        data.load_images()
        print(data.class_names)

    def test_extract(self):
        data = Data("this")
        data.try_download_imagenet("laurynas", "**")
        data.extract()

    def test_try_download_imagenet(self):
        data = Data("this")
        data.try_download_imagenet("laurynas", "**")
