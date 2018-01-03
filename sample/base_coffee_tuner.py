#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import tensorflow as tf



class BaseCoffeeTuner(abc.ABC):
	"""Abstract base class for coffee-tuner. The `self.model` is to be
	implemented in any class inherits this."""

	dtype = 'float32'


	def __init__(self, n_features):
		
		self.n_features = n_features

		self.data = []
		self.graph = tf.Graph()
		self.posterior = self.get_prior()


	def train(self):
		pass


	def sugguests(self):
		pass


	def add_data(self):
		pass


	def compile(self):

		def log_g(x, y, param):
			""":math:`\ln g(y, f(x, w))` in the documentation.
			Args:
				x:
					Tensor of the shape [self.n_features].
				y:
					Scalar.
				param:
					The same as the argument `param` in `self.model`.
			Returns:
				Scalar.
			"""
			g = tf.cond(y>0, self.model(x, param), 1-self.model(x, param))
			return tf.log(g)
		
	    with self.graph.as_defaut():

	    	# shape: `[len(self.data), n_features]`
	    	self.x_tensor = tf.placeholder(
	    		shape=[None, self.n_features], dtype=self.dtype)
	    	# shape: `[len(self.data)]`
	    	self.y_tensor = tf.placeholder(shape=[None], dtype=self.dtype)

	    	# XXX: TBC




	@abc.abstractmethod
	def model(self, input, param):
		"""Shall be implemented by TensorFlow. In the documentation, this is
		the :math:`f`.

		Args:
			input:
				Tensor as input-value.

			param:
				Dictionary with keys as `str`s as parameter-name and values as
				`Tensor`s, as the parameter-value of the model.

		Returns:
			Scalar, as the parameter of the Bernoulli distribution of "taste".
			In the documentation, this is the :math:`\psi`.
		"""
		assert param.keys() == self.posterior.keys()
		pass


	@abc.abstractmethod
	def get_prior(self):
		"""Shall be implemented by TensorFlow. In the documentation, this is
		the :math:`p_1(W)`.

		Returns:
			Dictionary with keys as `str`s as parameter-name in the argument
			`param` in `self.model`, and values `tf.distributions.Distribution`
			instances, as the prior distribution.
		"""
		pass