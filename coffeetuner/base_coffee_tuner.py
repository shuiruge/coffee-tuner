#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import tensorflow as tf

from nn4post import build_nn4post
from nn4post.utils import get_param_shape, get_parse_param, get_trained_q
from nn4post.utils.tf_trainer import SimpleTrainer




class BaseCoffeeTuner(abc.ABC):
    """Abstract base class for coffee-tuner. The `self.model` is to be
    implemented in any class inherits this."""

    def __init__(self, n_features, dir_to_ckpt,
                 data=None, n_c=1, dtype='float32'):
		
        self.n_features = n_features
        self.dir_to_ckpt = dir_to_ckpt

        self.data = [] if data is None else data
        self.n_c = n_c
        self.dtype = dtype

        self.graph = tf.Graph()
        self.prior = self.get_prior()

        self.x, self.y, self.collection, self.grads_and_vars = self.compile()

        self.loss = self.collection['loss']
        self.trainer = SimpleTrainer(loss=self.loss,
            gvs=self.grads_and_vars, dir_to_ckpt=self.dir_to_ckpt)
        # TODO: self.argmaxer = SimpleTrainer(loss=XXX)

    
    def train(self, n_iters=1000):
        """If employ mini-batch, then `scale` argument in the employed
        `build_nn4post` shall be assigned by the mini-batch scale.

        Args:
            n_iters:
                `int` as the number of iterations in the training. optional.

        Returns:
            An instance of `tf.distributions.Distribution`, as the trained
            inference distribution which fits the posterior, and from which
            will the parameters be sampled.
        """
        
        def get_feed_dict_generator():

            x = np.array([x for x, y in self.data], dtype=self.dtype)
            y = np.array([y for x, y in self.data], dtype=self.dtype)

            while True:
                feed_dict = {self.x: x, self.y: y}
                yield feed_dict

        feed_dict_generator = get_feed_dict_generator()

        self.trainer.train(n_iters, feed_dict_generator)

        trained_var = {}
        for var_name in ['a', 'mu', 'zeta']:
            var_value = self.trainer.sess.run(self.ops[var_name])
            trained_var[var_name] = var_value

        trained_q = get_trained_q(trained_var)
        return trained_q

    
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

            with tf.name_scope('data'):

                # shape: `[len(self.data), n_features]`
                x_tensor = tf.placeholder(
                    shape=[None, self.n_features], dtype=self.dtype)
                # shape: `[len(self.data)]`
                y_tensor = tf.placeholder(shape=[None], dtype=self.dtype)

            with tf.name_scope('log_joint'):

                def log_joint(param):
                    """:math:`ln p ( data, param )`, which, in documentation,
                    is :math:`\sum_i ln g( y_i, f(x_i, w) )`.

                    Args:
                        param:
                            The same as the argument `param` in `self.model`.
                        Returns:
                            Scalar.
                    """

                    log_joint_list = tf.map_fn(
                        lambda x, y: log_g(x, y, param),
                        (x_tensor, y_tensor))

                    return tf.reduce_sum(log_joint_list)

            with tf.name_scope('log_prior'):

                def log_prior(param):
                    """:math:`ln p ( param )`.

                    Args:
                        param:
                            The same as the argument `param` in `self.model`.
                        Returns:
                            Scalar.
                    """
                    log_prior_list = [
                        self.prior[param_name].log_prob(param_val)
                        for param_name, param_val in param.items()
                    ]
                    return tf.reduce_sum(log_prior_list)

            with tf.name_scope('log_posterior'):

                @vectorize(get_param_shape(self.prior))
                def log_posterior(param):
                    return log_joint(param) + log_prior(param)


            with tf.name_scope('variational_inference'):

                collection, grads_and_vars = build_nn4post(self.n_c,
                    param_space_dim, log_posterior, dtype=self.dtype)

            return (x_tensor, y_tensor, collection, grads_and_vars)


    @abc.abstractmethod
    def model(self, input, param):
        """Shall be implemented by TensorFlow. In the documentation, this is
        the :math:`f`.

        CAUTION:
            For numerical stability, ensure that `model` is asymptotic at
            (image) values `0` and `1`, like using sigmoid. This is because
            `model` will be placed into logorithm as `model(input, param)` or
            `1 - model(input, param)`.

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