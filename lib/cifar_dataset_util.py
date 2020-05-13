# --------------------------------------------------------
# Onaria technologies
# Written by: Rilwan Remilekun Basaru
# --------------------------------------------------------
import pickle
import random

import numpy as np
from lib.database_util import Dataset
import os

data_meta_info = {
	"TRAIN": ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'],
	"VAL": ['test_batch'],
	"meta": 'batches.meta'
}


def load_data():

	path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'Data', 'cifar')
	images = []
	targets = []
	for i in range(5):
		filename = os.path.join(path, "data_batch_%d" % (i + 1))

		with open(filename, 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
			image = np.reshape(dict[b'data'], (-1, 32, 32, 3))
			labels = np.reshape(np.array(dict[b'labels']), (-1, 1))
			images.append(image)
			targets.append(labels)
	train_images = np.concatenate(images, axis=0)
	train_targets = np.concatenate(targets, axis=0)

	filename = os.path.join(path, "test_batch")
	with open(filename, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
		test_images = np.reshape(dict[b'data'], (-1, 32, 32, 3))
		test_targets = np.reshape(np.array(dict[b'labels']), (-1, 1))

	filename = os.path.join(path, "batches.meta")
	with open(filename, 'rb') as fo:
		label_meta = pickle.load(fo, encoding='bytes')

	return train_images, train_targets, test_images, test_targets, label_meta


# CIFAR-10 Dataset http://www.cs.utoronto.ca/~kriz/cifar.html

class CIFARDatahandler(Dataset):
	def __init__(self, batch_size):
		train_images, train_targets, test_images, test_targets, label_meta = load_data()
		self.train_images = train_images
		self.train_targets = train_targets
		self.test_images = test_images
		self.test_targets = test_targets
		self.images = None
		self.labels = None
		self.num_classes = np.unique(self.train_targets).max() + 1
		super().__init__(batch_size)
		self.sample_shape = (None, 32, 32, 3)
		self.size_fourth = 8

	def get_next_batch(self):

		indices = self._get_next_minibatch_inds()
		target = self.labels[indices, :]
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
	cifar_dataset_handler = CIFARDatahandler(20)
	cifar_dataset_handler.reset("TRAIN")
	cifar_dataset_handler.get_next_batch()