# --------------------------------------------------------
# Onaria technologies
# Written by: Rilwan Remilekun Basaru
# --------------------------------------------------------

import numpy as np
import random


def generate_input_signal(size=(5, 40, 40, 3)):
	return np.random.rand(size[0], size[1], size[2], size[3])


class Dataset:
	def __init__(self, batch_size, length=20000, sample=None):
		if sample is not None:
			self.indices = random.sample(range(length), sample)
		else:
			self.indices = list(range(length))
			random.shuffle(self.indices)
		self.data_indices = None
		self.ite = 0
		self.batch_size = batch_size
		self.sample_shape = None

	def augment_blob(self, blob, batch_size):
		sample_shape = [None, 1, 1, 100]
		sample_shape[0] = batch_size
		input_signal = generate_input_signal(sample_shape)
		blob["input_signal"] = input_signal
		return blob

	def get_next_augmented_batch(self):
		blob = self.get_next_batch()
		blob = self.augment_blob(blob, blob["input_img"].shape[0])
		return blob

	def get_next_batch(self):
		raise NotImplementedError

	def _get_next_minibatch_inds(self):

		db_inds = self.data_indices[self.ite: min(self.data_indices.size, self.ite + self.batch_size)]
		self.ite += self.batch_size

		return db_inds

	def ite_count(self, multiplier=1):
		return np.ceil(self.data_indices.size / (self.batch_size * multiplier))

	def is_next(self):
		return self.ite < self.data_indices.size

	def get_size(self):
		return self.data_indices.size

	def reset(self, mode):
		train_count = round(0.8 * float(len(self.indices)))
		if mode.lower() == 'train':
			self.data_indices = np.array(self.indices[:train_count])
		else:
			self.data_indices = np.array(self.indices[train_count:])
		self.ite = 0

