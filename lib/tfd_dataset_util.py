# --------------------------------------------------------
# Onaria technologies
# Written by: Rilwan Remilekun Basaru
# --------------------------------------------------------
import numpy as np
from lib.database_util import Dataset
import os
import scipy.io as sio


# Toronto Face Dataset https://vjywpjj.tk/sports/toronto-face-dataset.php
class TFDDataHandler(Dataset):
	def __init__(self, batch_size):
		self.absolute_base_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'Data', 'tfd')
		tfd_96x96 = sio.loadmat(os.path.join(self.absolute_base_directory, "TFD_96x96.mat"))

		self.labels = tfd_96x96['labs_ex']
		self.images = tfd_96x96["images"]
		self.num_classes = self.labels.max() - self.labels.min() + 1
		self.labels = self.labels - self.labels.min()
		# self.folds = tfd_96x96['folds']
		# self.length = 40
		super().__init__(batch_size, length=self.images.shape[0])
		# super().__init__(batch_size, length=self.images.shape[0], sample=self.length)
		self.sample_shape = (None, 96, 96, 1)
		self.size_fourth = 24

	def get_next_batch(self):
		indices = self._get_next_minibatch_inds()
		target = self.labels[indices, :]
		input = self.images[indices, :, :]
		target = np.eye(self.num_classes)[target.reshape(-1)]

		blob = {
			"input_img": np.expand_dims(input, axis=3).astype(np.float32),
			"target_img": target.astype(np.float32)
		}
		return blob


if __name__ == "__main__":
	tfd_dataset_handler = TFDDataHandler(20)
	tfd_dataset_handler.reset("TRAIN")
	tfd_dataset_handler.get_next_batch()