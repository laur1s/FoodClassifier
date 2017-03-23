import os
import urllib.request
import pickle
import tarfile
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.misc import imresize
from sklearn import preprocessing


class Data(object):
    def __init__(self, directory):
        self.directory = os.path.abspath(directory)
        self.img_directory = os.path.join(self.directory, "images")
        self.tar_directory = os.path.join(self.directory, "tar")
        self.pickle_directory = os.path.join(self.directory, "pickle")
        self.class_names = []
        self.num_of_classes = 0
        self.images = []
        self.labels = []
        self.img_size = 0

    def one_hot(self, y_train, y_test):
        """ Transforms the categorical class names into one hot encoded"""
        le = preprocessing.LabelEncoder()
        le.fit(y_train)
        numeric_labels = le.transform(y_train)
        numeric_test_labels = le.transform(y_test)
        n_classes = 4
        y_one_hot = np.eye(n_classes)[numeric_labels]
        y_test_code = np.eye(n_classes)[numeric_test_labels]

        return y_one_hot, y_test_code

    def train_test_split(self, t_size=0.1):
        (x_train, x_test, y_train, y_test) = train_test_split(self.images, self.labels,
                                                              test_size=t_size, random_state=42)
        return x_train, x_test, y_train, y_test

    def to_gray_scale(self):
        grays = []
        for image in self.images:
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            grays.append(gray)

            self.images = grays
        return grays

    def resize_img(self, size):
        """
        :param images: list of images
        :param size: desired size of images formatted as [h, w]
        :return: list of images resized  to the desired size
        """
        resized = []
        for image in self.images:
            resized.append(imresize(image, size, 'bilinear'))
        self.images = resized
        self.img_size = len(resized[0])
        return resized

    def save_pickle(self, name):
        """
        Save the image data to python pickle file
        """
        if not os.path.exists(self.pickle_directory):
            os.makedirs(self.pickle_directory)
        print(self.pickle_directory + '\\' + name)
        with open(self.pickle_directory + '\\' + name, 'wb') as f:
            pickle.dump(self.labels, f)
            pickle.dump(self.images, f)

    def load_pickle(self, name):
        """
        loads the data set from the pickle file to memory
        """

        file = os.path.join(self.pickle_directory, name)
        with open(file, "rb") as f:
            self.labels = pickle.load(f)
            self.images = pickle.load(f)
        classes = set(self.labels)
        self.class_names = list(classes)

    def load_images(self):
        """
        loads the data set from the images folder to memory
        :return: img_clases - list with name of  every image, images- list of images
        """
        for name in os.listdir(self.img_directory):
            self.class_names.append(name)
            class_dir = os.path.join(self.img_directory, name)
            for file in os.listdir(class_dir):
                if os.path.splitext(file)[-1].lower() == (".jpeg" or ".jpg"):
                    self.labels.append(name)
                    self.images.append(plt.imread(os.path.join(class_dir, file)))
            return self.labels, self.images

    def extract(self):
        """Extracts the images from the tar directory to the images category"""
        for file in os.listdir(self.tar_directory):
            class_name = os.path.splitext(file)[0]  # get a class name from a folder
            img_class_path = os.path.join(self.img_directory, class_name)
            if not os.path.exists(img_class_path):
                tar_dir = os.path.join(self.tar_directory, file)
                t_file = tarfile.open(name=tar_dir)
                print(t_file)
                t_file.extractall(img_class_path)
                t_file.close()

    def try_download_imagenet(self, user_name, access_key):
        """Download the archived image files from ImageNet of selected classes,
        after checking if these classes are not existing
        """
        classes = ["Vanilla Ice Cream", "Pizza", "Salad", "Fish and Chips"]
        self.num_of_classes = len(classes)
        self.class_names = classes
        links = {
            "Vanilla Ice Cream": "http://www.image-net.org/download/synset?wnid=n07615671&username=" + user_name + "&accesskey=" + access_key + "&release=latest&src=stanford",
            "Pizza": "http://image-net.org/download/synset?wnid=n07873807&username=" + user_name + "&accesskey=" + access_key + "&release=latest&src=stanford",
            "Salad": "http://image-net.org/download/synset?wnid=n07806221&username=" + user_name + "&accesskey=" + access_key + "&release=latest&src=stanford",
            "Fish and Chips": "http://image-net.org/download/synset?wnid=n07867324&username=" + user_name + "&accesskey=" + access_key + "&release=latest&src=stanford"
        }
        if not os.path.exists(self.tar_directory):
            os.makedirs(self.tar_directory)
        for img_class in classes:
            archive = os.path.join(img_class + ".tar")
            tar_dir = os.path.join(self.tar_directory, archive)
            if not os.path.isfile(tar_dir):
                print("Downloading....")
                tar = urllib.request.urlretrieve(links[img_class], filename=tar_dir)
