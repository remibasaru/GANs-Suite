# --------------------------------------------------------
# Onaria Technologies
# Written by: Rilwan Remilekun Basaru
# --------------------------------------------------------
import errno
import os
import pickle
import random

from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re

import time


class Timer(object):
	"""A simple timer."""

	def __init__(self):
		self.total_time = 0.
		self.calls = 0
		self.start_time = 0.
		self.diff = 0.
		self.average_time = 0.

	def tic(self):
		# using time.time instead of time.clock because time time.clock
		# does not normalize for multithreading
		self.start_time = time.time()

	def toc(self, average=True):
		self.diff = time.time() - self.start_time
		self.total_time += self.diff
		self.calls += 1
		self.average_time = self.total_time / self.calls
		if average:
			return self.average_time
		else:
			return self.diff


class Trainer(object):

	def __init__(self, _expDir):
		self.saver = None
		self.expDir = _expDir
		self.model_figure_path = None

	def update_opts(self):
		pass

	def model_path(self, ep):
		return os.path.join(self.expDir, 'net-epoch-' + str(ep), 'model.ckpt')

	def model_folder_path(self, ep):
		return os.path.join(self.expDir, 'net-epoch-' + str(ep))

	def load_state(self, prev_pos_1, sess):
		model_folder_path = self.model_folder_path(prev_pos_1)
		model_path = self.model_path(prev_pos_1)
		self.saver.restore(sess, model_path)
		stats = None
		with open(os.path.join(model_folder_path, 'stats.pickle'), 'rb') as handle:
			stats = pickle.load(handle)

		return stats, sess

	def save_state(self, epoch, state):
		save_path_folder = self.model_folder_path(epoch)
		save_path = self.model_path(epoch)
		try:
			os.mkdir(save_path_folder)
		except OSError as exc:
			if exc.errno != errno.EEXIST:
				raise
			pass
		_sess = state['sess']
		save_path = self.saver.save(_sess, save_path)
		print("Model saved in path: %s" % save_path)
		return True

	def save_stats(self, epoch, stats):
		stats_path = self.model_folder_path(epoch)
		with open(os.path.join(stats_path, 'stats.pickle'), 'wb') as handle:
			pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
		return None

	def find_last_checkpoint(self):
		epoch = 0
		for f in os.listdir(os.path.join('.', self.expDir)):
			if re.match(r'net-epoch-\d+', f):
				tmp_epoch = int((re.search(r'\d+', f)).group(0))
				if tmp_epoch > epoch:
					epoch = tmp_epoch
		return epoch