import tensorflow as tf

from model.network_util import conv2d_block, conv2d, conv_1x1, Network, deconv2d_block
import numpy as np


class GANsNetwork(Network):
	def __init__(self, device, num_filters_root):
		super().__init__(device)
		self.input_signal = None
		self.train_flag = None
		self.real_input_images = None
		self.num_channels = 1
		self.discriminator_train_op = None
		self.generator_train_op = None
		self.discriminator_model_scope_name = "Disc_Model"
		self.generator_model_scope_name = "Gen_Model"
		self.train_discriminator = None
		self.num_filters_root = num_filters_root

	def handle_admin(self):
		self.input_signal = tf.placeholder(dtype=tf.float32, shape=(None, 1, 1, 100), name="input_signal")
		self.real_input_images = tf.placeholder(dtype=tf.float32, shape=(None, None, None, None), name="input_images")

		self.train_flag = tf.Variable(False, trainable=False, name='train_mode')
		self.train_discriminator = tf.Variable(True, trainable=False, name='train_discriminator')

	def get_discriminator_train_mode_target(self):
		generated_images = self.layers["generated_images"]
		with tf.variable_scope("G_target"):
			real_shape = (tf.shape(self.real_input_images)[0], 1)
			fake_shape = (tf.shape(generated_images)[0], 1)
			real_gt = tf.concat((tf.ones(real_shape, dtype=self.real_input_images.dtype),
								tf.zeros(real_shape, dtype=self.real_input_images.dtype)), axis=-1)
			fake_gt = tf.concat((tf.zeros(fake_shape, dtype=generated_images.dtype),
								tf.ones(fake_shape, dtype=generated_images.dtype)), axis=-1)

			target = tf.concat((real_gt, fake_gt), axis=0)
		return tf.stop_gradient(target)

	def get_generator_train_mode_target(self):
		generated_images = self.layers["generated_images"]
		with tf.variable_scope("D_target"):

			shape = (tf.shape(generated_images)[0], 1)
			target = tf.concat((tf.ones(shape, dtype=generated_images.dtype), tf.zeros(shape, dtype=generated_images.dtype)), axis=-1)

		return tf.stop_gradient(target)

	def build_model(self):
		self.handle_admin()
		with tf.device(self._device):
			# build generator
			self.build_generator()
			with tf.variable_scope("switch"):
				#  when training the discriminator we need to feed both real and generated(fake) images otherwise we only pass in the generated (fake)images
				input = tf.cond(self.train_discriminator,
								lambda: tf.concat((self.real_input_images, self.layers["generated_images"]), axis=0),
								lambda: tf.identity(self.layers["generated_images"]))
			# build discriminator
			self.build_discriminator(input)
			self.add_discriminator_max()
			self.add_loss()
			self.write_graph_to_tensorboard(tf.get_default_graph())

	def build_generator(self):
		with tf.variable_scope(self.generator_model_scope_name):
			depth_multiplier = 1

			net = conv_1x1(self.input_signal, np.power(self.num_filters_root, 2), name='fc1_1')
			net = tf.reshape(net, (-1, self.num_filters_root, self.num_filters_root, 1))
			net = conv_1x1(net, 256, name='fc1_2')

			net = deconv2d_block(net, int(128/depth_multiplier), 3, 2, name='deconv2')
			net = conv2d_block(net, 128/depth_multiplier, 3, 1, self.train_flag, name='conv2_1')
			net = conv2d_block(net, 128/depth_multiplier, 3, 1, self.train_flag, name='conv2_2')

			net = conv2d_block(net, 256/depth_multiplier, 3, 1, self.train_flag, name='conv3_1')
			net = conv2d_block(net, 256/depth_multiplier, 3, 1, self.train_flag, name='conv3_2')
			net = conv2d_block(net, 256/depth_multiplier, 3, 1, self.train_flag, name='conv3_3')

			net = deconv2d_block(net, int(256/depth_multiplier), 3, 2, name='deconv3')
			net = conv2d_block(net, 512/depth_multiplier, 3, 1, self.train_flag, name='conv4_1')
			net = conv2d_block(net, 512/depth_multiplier, 3, 1, self.train_flag, name='conv4_2')

			net = conv2d_block(net, 128/depth_multiplier, 3, 1, self.train_flag, name='conv5_1')
			net = conv2d_block(net, 128/depth_multiplier, 3, 1, self.train_flag, name='conv5_2')

			net = conv2d_block(net, 64/depth_multiplier, 3, 1, self.train_flag, name='conv6_1')
			net = conv2d(net, self.num_channels, 3, 3, 1, 1,  name='conv6_2', bias=False)
			self.layers["generated_images"] = net

	def build_discriminator(self, input):
		with tf.variable_scope(self.discriminator_model_scope_name):
			depth_multiplier = 2

			net = conv2d_block(input, 64 / depth_multiplier, 3, 1, self.train_flag, name='conv1_1')

			net = conv2d_block(net, 128 / depth_multiplier, 3, 1, self.train_flag, name='conv2_1')
			net = conv2d_block(net, 128, 3, 2, self.train_flag, name='conv2_2')

			net = conv2d_block(net, 256 / depth_multiplier, 3, 1, self.train_flag, name='conv3_1')
			net = conv2d_block(net, 256, 3, 1, self.train_flag, name='conv3_2')

			net = conv2d_block(net, 256 / depth_multiplier, 3, 1, self.train_flag, name='conv4_1')
			net = conv2d_block(net, 256, 3, 2, self.train_flag, name='conv4_2')

			net = conv2d_block(net, 128 / depth_multiplier, 3, 1, self.train_flag, name='conv5_1')
			net = conv2d_block(net, 128 / depth_multiplier, 3, 2, self.train_flag, name='conv5_2')

			net = conv2d_block(net, 64 / depth_multiplier, 3, 1, self.train_flag, name='conv6_1')

			flattened = tf.reshape(net,  [tf.shape(net)[0], 1, 1, net.shape[-1] * net.shape[-2] * net.shape[-3]])
			fc6 = conv_1x1(flattened, 4096, name='fc6')

			fc7 = conv_1x1(fc6, 4096, name='fc7')

			logits = tf.contrib.layers.flatten(conv_1x1(fc7, 2, name='logits'))
			self.layers["discriminator_output"] = logits

	def add_discriminator_max(self):
		with tf.variable_scope("arg_max"):
			ix = tf.argmax(self.layers["discriminator_output"], axis=-1)
			self.layers["discriminator_classification"] = ix

	def add_loss(self):

		with tf.variable_scope("loss"):
			target = tf.cond(self.train_discriminator,
							lambda: self.get_discriminator_train_mode_target(),
							lambda: self.get_generator_train_mode_target())
			loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.layers["discriminator_output"], labels=target, name="loss")
			# loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.get_discriminator_train_mode_target(), logits=self.layers["discriminator_output"], name="loss")
			self.losses["per_image_loss"] = loss
			mean_entropy_loss = tf.reduce_mean(loss, name="mean")

		self.losses["discriminator_loss"] = mean_entropy_loss

	def add_optimizers(self, opts):
		self._add_generator_optimizer(opts)
		self._add_discriminator_optimizer(opts)

	def _add_discriminator_optimizer(self, opts):
		lr = opts["discriminatorLearningRate"]
		loss = self.losses['discriminator_loss']

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			optimizer = tf.train.AdamOptimizer(learning_rate=lr)

			gvs = optimizer.compute_gradients(loss)
			final_gvs = []

			with tf.variable_scope('Disc_filter'):
				for grad, var in gvs:
					if self.discriminator_model_scope_name not in var.name:
						grad = tf.multiply(grad, 0)
					final_gvs.append((grad, var))

			self.discriminator_train_op = optimizer.apply_gradients(final_gvs)

	def _add_generator_optimizer(self, opts):
		lr = opts["generatorLearningRate"]
		loss = self.losses['discriminator_loss']

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			optimizer = tf.train.AdamOptimizer(learning_rate=lr)

			gvs = optimizer.compute_gradients(loss)
			final_gvs = []

			with tf.variable_scope('Gen_filter'):
				for grad, var in gvs:
					if self.generator_model_scope_name not in var.name:
						grad = tf.multiply(grad, 0)
					final_gvs.append((grad, var))

			self.generator_train_op = optimizer.apply_gradients(final_gvs)

	def generator_test_step(self, sess, blob):
		feed_dict = {
			self.input_signal: blob["input_signal"],
			self.real_input_images: np.zeros_like(blob["input_signal"]),
			self.train_flag: False,
			self.train_discriminator: False
		}
		generated_images = sess.run(self.layers["generated_images"], feed_dict=feed_dict)
		return generated_images

	def discriminator_test_step(self, sess, blob):
		feed_dict = {
			self.real_input_images: blob["input_img"],
			self.input_signal: blob["input_signal"],
			self.train_flag: False,
			self.train_discriminator: True
		}

		discriminator_class, discriminator_loss = sess.run([self.layers["discriminator_classification"],
										self.losses["discriminator_loss"]], feed_dict=feed_dict)
		return discriminator_class, discriminator_loss

	def discriminator_train_step(self, sess, blob, mode="TRAIN"):
		feed_dict = {
			self.real_input_images: blob["input_img"],
			self.input_signal: blob["input_signal"],
			self.train_flag: False,
			self.train_discriminator: True
		}

		if mode == 'TRAIN':
			feed_dict[self.train_flag] = True
			sess.run([self.discriminator_train_op], feed_dict=feed_dict)

		discriminator_loss = sess.run(self.losses["discriminator_loss"], feed_dict=feed_dict)
		return discriminator_loss

	def generator_train_step(self, sess, blob, stats, mode="TRAIN"):
		# 1, self.real_input_images.shape[1], self.real_input_images.shape[2], self.real_input_images.shape[3]
		feed_dict = {
			self.input_signal: blob["input_signal"],
			self.real_input_images: np.zeros_like(blob["input_signal"]),
			self.train_flag: False,
			self.train_discriminator: False
		}
		if mode == 'TRAIN':
			feed_dict[self.train_flag] = True
			sess.run([self.generator_train_op], feed_dict=feed_dict)

		generated_images, discriminator_loss = sess.run([self.layers["generated_images"],
														 self.losses["discriminator_loss"]], feed_dict=feed_dict)
		self.update_stats("generative_loss", discriminator_loss, stats)
		return stats

	def update_stats(self, name, out_measure, stats):

		loss_val = np.mean(out_measure)
		cur_stats = stats[name]
		tot_loss = cur_stats['average'] * cur_stats['count'] + loss_val
		cur_stats['count'] = cur_stats['count'] + 1
		cur_stats['average'] = tot_loss / cur_stats['count']
		stats[name] = cur_stats
		return stats