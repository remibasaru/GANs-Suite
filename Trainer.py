

import os
import numpy as np
import cv2

from GANs_util import GANsNetwork
from lib.trainer_util import Trainer, Timer
import tensorflow as tf


class GANsTrainer(Trainer):

	def __init__(self, param):

		super().__init__(param["expDir"])
		self.opts = param
		self.batch_size = 20
		self.data_loader = param["data_loader"](self.batch_size)
		self.sess = None
		if not os.path.isdir(param['expDir']):
			os.mkdir(param['expDir'])
		self.model_figure_path = os.path.join(param['expDir'], 'net-train.pdf')

		if param['use_gpu']:
			device_str = "/device:GPU:0"
		else:
			device_str = "/cpu:0"
		self.net = GANsNetwork(device_str, self.data_loader.size_fourth)
		self.discriminator_training_steps = param["innerTrainSteps"]
		self.num_train_ite = param["numTrainIte"]

	def setup(self):
		tfconfig = tf.ConfigProto(allow_soft_placement=True)
		tfconfig.gpu_options.allow_growth = True
		self.data_loader.reset("TRAIN")

		sess = tf.Session(config=tfconfig)
		with sess.graph.as_default():
			self.net.build_model()
			self.net.add_optimizers(self.opts)
			variables = tf.global_variables()
			# Initialize all variables first
			sess.run(tf.variables_initializer(variables))
			self.saver = tf.train.Saver()

		self.sess = sess

	def process_discriminator_training_round(self):

		for k in range(self.discriminator_training_steps):
			blob = self.data_loader.get_next_augmented_batch()
			curr_loss = self.net.discriminator_train_step(self.sess, blob)
			# _, curr_loss = self.net.discriminator_test_step(self.sess, blob)
			# print("\t loss: ", curr_loss)
			if not self.data_loader.is_next():
				self.data_loader.reset("TRAIN")
				return True
		return False

	def process_epoch(self, state, params, timer, display=False):

		epoch = params['epoch']
		stats = dict()

		if not state:
			state['stats'] = dict()
		losses_name = ["generative_loss"]
		for l in losses_name:
			stats[l] = {'count': 0, 'average': 0}
		ite = 1
		exhausted_dataset = False
		# for ite in range(self.num_train_ite):
		while not exhausted_dataset:
			exhausted_dataset = self.process_discriminator_training_round()
			blob = self.data_loader.augment_blob({}, self.batch_size)
			# print("Discriminator Loss (Forward Phase):")
			timer.tic()
			stats = self.net.generator_train_step(self.sess, blob, stats)
			timer.toc()
			if display:
				image = self.net.generator_test_step(self.sess, blob)
				img = self.get_pretty_image(image)
				cv2.imshow("test", img)

			# Display training information
			print('train: epoch %d:\t %d/%d: (%.2fs)\t generative_loss: %.6f ' %
				  (epoch, ite, self.data_loader.ite_count(self.discriminator_training_steps),
				   timer.average_time, stats["generative_loss"]['average']))
			ite = ite + 1
		# Save back to state
		state['stats'] = stats
		state['sess'] = self.sess
		return state

	@staticmethod
	def get_pretty_image(img):
		cut = np.power(np.ceil(np.power(img.shape[0], .5)), 2)
		padding = np.zeros((int(cut - img.shape[0]), img.shape[1], img.shape[2], img.shape[3]))
		img = np.concatenate((img, padding), axis=0)
		img = np.reshape(img, (int(np.power(cut, .5)), int(np.power(cut, .5)), img.shape[-3], img.shape[-2], img.shape[-1]))
		img = np.transpose(img, (0, 2, 1, 3, 4))
		img = np.reshape(img, (img.shape[0] * img.shape[1], -1, img.shape[-1]))
		return img.astype(np.uint8)

	def train_model(self, opts):
		timer = Timer()

		if opts['continue'] is not None:
			prev_pos_1 = max(0, min(opts['continue'], self.find_last_checkpoint()))
		else:
			prev_pos_1 = max(0, self.find_last_checkpoint())

		start_1 = prev_pos_1 + 1
		if prev_pos_1 >= 1:
			print('Resuming by loading epoch', str(prev_pos_1))
			stats, self.sess = self.load_state(prev_pos_1, self.sess)

			if self.sess is None or stats is None:
				stats = dict()
				stats['train'] = []
				stats['val'] = []
				print('Failed to load. Starting with epoch ', str(start_1), '\n')
			else:
				print('Continuing at epoch ', str(start_1), '\n')
		else:
			stats = []
			print('Starting at epoch ', str(start_1), '\n')

		state = dict()
		for ep in range(start_1 - 1, opts['numEpochs']):
			epoch = ep + 1
			params = opts
			params['epoch'] = epoch

			state = self.process_epoch(state, params, timer)
			self.save_state(epoch, state)
			last_stats = state['stats']
			stats.append(last_stats)
			self.save_stats(epoch, stats)

		return stats


if __name__ == "__main__":

	opts = {}
	opts['learningRate'] = 0.0001
	opts['weightDecay'] = 0.005
	opts['momentum'] = 0.09
	opts['use_gpu'] = True

	gans_trainer = GANsTrainer(opts)
	gans_trainer.setup()
	gans_trainer.process_discriminator_training_round()
