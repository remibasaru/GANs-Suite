# --------------------------------------------------------
# Onaria technologies
# Written by: Rilwan Remilekun Basaru
# --------------------------------------------------------

from mnist import MNIST
import os
import numpy as np
import random
from lib.database_util import Dataset

# MNIST Hand-written dataset http://yann.lecun.com/exdb/mnist/


def load_mnist():
	dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'Data', 'mnist')
	mndata = MNIST(os.path.join(dir))

	images, labels = mndata.load_training()
	train_images = np.reshape(np.array(images), (-1, 28, 28, 1))
	train_labels = np.array(labels)
	images, labels = mndata.load_testing()
	test_images = np.reshape(np.array(images), (-1, 28, 28, 1))
	test_labels = np.array(labels)

	return train_images, train_labels, test_images, test_labels


class MNISTDatahandler(Dataset):
	def __init__(self, batch_size):
		train_images, train_targets, test_images, test_targets = load_mnist()
		self.train_images = train_images
		self.train_targets = train_targets
		self.test_images = test_images
		self.test_targets = test_targets
		self.images = None
		self.labels = None
		self.num_classes = np.unique(self.train_targets).max() + 1
		super().__init__(batch_size)
		self.sample_size = (None, 28, 28, 1)
		self.size_fourth = 7

	def get_next_batch(self):

		indices = self._get_next_minibatch_inds()
		target = self.labels[indices]
		target = np.eye(self.num_classes)[target.reshape(-1)]
		input = self.images[indices, :, :, :]

		blob = {
			"input_img": input.astype(np.float32),
			"target_img": target.astype(np.float32)
		}
		return blob

	def reset(self, mode):
		if mode.lower() == 'train':
			self.data_indices = list(range(self.train_images.shape[0]))
			random.shuffle(self.data_indices)

			self.images = self.train_images
			self.labels = self.train_targets
		else:
			self.data_indices = list(range(self.test_images.shape[0]))
			random.shuffle(self.data_indices)
			self.images = self.test_images
			self.labels = self.test_targets
		self.data_indices = np.array(self.data_indices)
		self.ite = 0


if __name__ == "__main__":
	mnist_dataset_handler = MNISTDatahandler(20)
	mnist_dataset_handler.reset("TEST")
	mnist_dataset_handler.get_next_batch()
