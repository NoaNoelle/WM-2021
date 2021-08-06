#imports
from __future__ import division

import logging
import warnings

import numpy as np
import nengo

from nengo.exceptions import SimulationError, ValidationError, BuildError
import nengo.learning_rules
from nengo.builder import Operator, Signal
from nengo.utils.neurons import settled_firingrate
from nengo.neurons import AdaptiveLIFRate, LIF, LIFRate
from nengo.config import SupportDefaultsMixin
from nengo.params import (Default, IntParam, FrozenObject, NumberParam,
                          Parameter, Unconfigurable)
from nengo.synapses import Lowpass, SynapseParam
from nengo.utils.compat import is_iterable, is_string, itervalues, range
from nengo.learning_rules import *
from nengo.builder.neurons import SimNeurons
from nengo.builder.operator import DotInc, ElementwiseInc, Copy, Reset
from nengo.connection import LearningRule
from nengo.ensemble import Ensemble, Neurons
from nengo.node import Node
from nengo.builder.learning_rules import *
from nengo.dists import Uniform
from nengo.processes import Piecewise
#import nengo_spa as spa

from nengo_ocl import Simulator
from nengo_ocl.utils import as_ascii #, indent, round_up
from mako.template import Template
import pyopencl as cl
from nengo_ocl.plan import Plan
from nengo_ocl.clra_nonlinearities import _plan_template

from collections import OrderedDict

import pyopencl as cl
from mako.template import Template
import nengo.dists as nengod
from nengo.utils.compat import is_number, itervalues, range

from nengo_ocl.raggedarray import RaggedArray
from nengo_ocl.clraggedarray import CLRaggedArray, to_device
from nengo_ocl.plan import Plan
from nengo_ocl.utils import as_ascii, indent, round_up

import tensorflow as tf

from nengo_dl.compat import tf_compat, tf_math


#create new neuron type stpLIF with resources (x) and calcium (u)

class stpLIF(LIF):
    probeable = ('spikes', 'resources', 'voltage', 'refractory_time', 'calcium')

    tau_x = NumberParam('tau_x', low=0, low_open=True)
    tau_u = NumberParam('tau_u', low=0, low_open=True)
    U = NumberParam('U', low=0, low_open=True)

    def __init__(self, tau_x=0.2, tau_u=1.5, U=0.2, **lif_args):
        super(stpLIF, self).__init__(**lif_args)
        self.tau_x = tau_x
        self.tau_u = tau_u
        self.U = U

    @property
    def _argreprs(self):
        args = super(stpLIF, self)._argreprs # used to be LIFRate
        if self.tau_x != 0.2:
            args.append("tau_x=%s" % self.tau_x)
        if self.tau_u != 1.5:
            args.append("tau_u=%s" % self.tau_u)
        if self.U!= 0.2:
            args.append("U=%s" % self.U)
        return args

    def step_math(self, dt, J, output, voltage, ref, resources, calcium):
        """Implement the u and x parameters """
        x = resources
        u = calcium
        LIF.step_math(self, dt, J, output, voltage, ref)

        #calculate u and x
        dx=dt * ( (1-x)/self.tau_x - u*x*output )
        du=dt * ( (self.U-u)/self.tau_u + self.U*(1-u)*output )

        x += dx
        u += du

#add builder for stpLIF

from nengo.builder import Operator, Signal
# from nengo.builder import Builder
from nengo_dl.builder import NengoBuilder
from nengo.builder.neurons import SimNeurons
@NengoBuilder.register(stpLIF)
def build_stpLIF(model, stplif, neurons):

    model.sig[neurons]['voltage'] = Signal(
        np.zeros(neurons.size_in), name="%s.voltage" % neurons)
    model.sig[neurons]['refractory_time'] = Signal(
        np.zeros(neurons.size_in), name="%s.refractory_time" % neurons)
    model.sig[neurons]['resources'] = Signal(
        np.ones(neurons.size_in), name="%s.resources" % neurons)
    model.sig[neurons]['calcium'] = Signal(
        np.full(neurons.size_in, stplif.U), name="%s.calcium" % neurons)
    model.add_op(SimNeurons(neurons=stplif,
                            J=model.sig[neurons]['in'],
                            output=model.sig[neurons]['out'],
                            states=[model.sig[neurons]['voltage'],
                                    model.sig[neurons]['refractory_time'],
                                    model.sig[neurons]['resources'],
                                    model.sig[neurons]['calcium']]))


#create new learning rule to model short term plasticity (only works if pre-ensemble has neuron type StpLIF)
class STP(LearningRuleType):
    """STP learning rule.
    Modifies connection weights according to the calcium and resources of the presynapse
    Parameters
    ----------
    learning_rate : float, optional (Default: 1)
        A scalar indicating the rate at which weights will be adjusted (exponential).
    Attributes
    ----------
    learning_rate : float
    """

    modifies = 'weights'
    probeable = ('delta', 'calcium', 'resources')

    learning_rate = NumberParam(
        'learning_rate', low=0, readonly=True, default=1)

    def __init__(self, learning_rate=Default):
        super(STP, self).__init__(learning_rate, size_in=0)

    @property
    def _argdefaults(self):
        return (('learning_rate', STP.learning_rate.default))

#builders for STP
class SimSTP(Operator):
    r"""Calculate connection weight change according to the STP rule.
    Implements the STP learning rule of the form:
    .. math:: omega_{ij} = .....
    where
    * :math:`\omega_{ij}` is the connection weight between the two neurons.
    Parameters
    ----------
    weights : Signal
        The connection weight matrix, :math:`\omega_{ij}`.
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    tag : str, optional (Default: None)
        A label associated with the operator, for debugging purposes.
    Attributes
    ----------
    delta : Signal
        The synaptic weight change to be applied, :math:`\Delta \omega_{ij}`.
    learning_rate : float
        The scalar learning rate, :math:`\kappa`.
    tag : str or None
        A label associated with the operator, for debugging purposes.
    weights : Signal
        The connection weight matrix, :math:`\omega_{ij}`.
    Notes
    -----
    1. sets ``[]``
    2. incs ``[]``
    3. reads ``[pre_filtered, post_filtered, weights]``
    4. updates ``[delta]``
    """

    def __init__(self, calcium, resources, weights, delta,
                 learning_rate, tag=None):
        super(SimSTP, self).__init__(tag=tag)
        self.learning_rate = learning_rate
       # self.init_weights=init_weights
        self.sets = []
        self.incs = []
        self.reads = [weights, calcium, resources]
        self.updates = [delta]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def weights(self):
        return self.reads[0]

    @property
    def calcium(self):
        return self.reads[1]

    @property
    def resources(self):
        return self.reads[2]

    @property
    def initial_calcium(self):
        return self.reads[1].initial_value

    @property
    def initial_weights(self):
        return self.reads[0].initial_value

    def _descstr(self):
        return '%s' % (self.delta)

    def make_step(self, signals, dt, rng):
        weights = signals[self.weights]
        delta = signals[self.delta]
        alpha = self.learning_rate #* dt
        init_weights = self.weights.initial_value
        calcium = signals[self.calcium]
        resources = signals[self.resources]
        U=self.calcium.initial_value
        def step_simstp():
            # perform update
                delta[...] = ((calcium * resources)/U) * init_weights - weights

        return step_simstp

@NengoBuilder.register(STP)
def build_stp(model, stp, rule):
    """Builds a `.STP` object into a model.

    Parameters
    ----------
    model : Model
        The model to build into.
    stp : STP
        Learning rule type to build.
    rule : LearningRule
        The learning rule object corresponding to the neuron type.
    Notes
    -----
    Does not modify ``model.params[]`` and can therefore be called
    more than once with the same `.Stp` instance.
    """

    conn = rule.connection
    calcium = model.sig[get_pre_ens(conn).neurons]['calcium']
    resources = model.sig[get_pre_ens(conn).neurons]['resources']

    ##added for nengo DL
    # U = model.sig[get_pre_ens(conn).neurons]['calcium'].initial_value


    model.add_op(SimSTP(calcium,
                        resources,
                        model.sig[conn]['weights'],
                        model.sig[rule]['delta'],
                        learning_rate=stp.learning_rate,
                        ))

    # expose these for probes
    model.sig[rule]['calcium'] = calcium
    model.sig[rule]['resources'] = resources
    # model.sig[rule]['U'] = U

#----- Nengo DL implementation of STP and StpLIF -------################################################
#-------------------------------------------------------################################################

###  NEURON  ###

from nengo_dl.builder import Builder, OpBuilder

from nengo_dl.neuron_builders import LIFRateBuilder, SimNeuronsBuilder

class stpLIFBuilder(OpBuilder): #### tensorflow implementation
    """Build a group of `.stpLIF` neuron operators."""

    def __init__(self, ops, signals, config):

        super(stpLIFBuilder, self).__init__(ops, signals, config)

        ### LIF ###

        self.tau_ref = signals.op_constant(
            [op.neurons for op in ops], [op.J.shape[0] for op in ops],
            "tau_ref", signals.dtype)

        self.tau_rc = signals.op_constant(
            [op.neurons for op in ops], [op.J.shape[0] for op in ops],
            "tau_rc", signals.dtype)

        self.amplitude = signals.op_constant(
            [op.neurons for op in ops], [op.J.shape[0] for op in ops],
            "amplitude", signals.dtype)

        self.J_data = signals.combine([op.J for op in ops])
        self.output_data = signals.combine([op.output for op in ops])
        self.zeros = tf.zeros(self.J_data.shape + (signals.minibatch_size,),
                              signals.dtype)

        self.epsilon = tf.constant(1e-15, dtype=signals.dtype)

        # copy these so that they're easily accessible in the _step functions
        self.zero = signals.zero
        self.one = signals.one

        self.min_voltage = signals.op_constant(
            [op.neurons for op in ops], [op.J.shape[0] for op in ops],
            "min_voltage", signals.dtype)

        self.alpha = self.amplitude / signals.dt

        self.voltage_data = signals.combine([op.states[0] for op in ops])
        self.refractory_data = signals.combine([op.states[1] for op in ops])

        ### stp ###

        self.tau_x = signals.op_constant(
            [op.neurons for op in ops],
            [op.J.shape[0] for op in ops],
            "tau_x",
            signals.dtype,
        )

        self.tau_u = signals.op_constant(
            [op.neurons for op in ops],
            [op.J.shape[0] for op in ops],
            "tau_u",
            signals.dtype,
        )

        self.U = signals.op_constant(
            [op.neurons for op in ops],
            [op.J.shape[0] for op in ops],
            "U",
            signals.dtype,
        )

        self.resources_data = signals.combine([op.states[2] for op in ops])

        self.calcium_data = signals.combine([op.states[3] for op in ops])

    def _step(self, J, voltage, refractory, dt, resources, calcium):

        ### LIF ###

        delta_t = tf.clip_by_value(dt - refractory, self.zero, dt)

        dV = (voltage - J) * tf_math.expm1(-delta_t / self.tau_rc)
        voltage += dV

        spiked = voltage > self.one
        spikes = tf.cast(spiked, J.dtype) * self.alpha

        partial_ref = -self.tau_rc * tf_math.log1p(
            (self.one - voltage) / (J - self.one)
        )

        refractory = tf.where(spiked, self.tau_ref - partial_ref, refractory - dt)

        voltage = tf.where(spiked, self.zeros, tf.maximum(voltage, self.min_voltage))

        ### stp ###

        x = resources
        u = calcium

        dx = delta_t * ( (self.one-x)/self.tau_x - u*x*spikes )
        du = delta_t * ( (self.U-u)/self.tau_u + self.U*(self.one-u)*spikes )

        x += dx
        u += du

        return (
            tf.stop_gradient(spikes),
            tf.stop_gradient(voltage),
            tf.stop_gradient(refractory),
            tf.stop_gradient(x),
            tf.stop_gradient(u)
        )

    def build_step(self, signals):

        J = signals.gather(self.J_data)
        voltage = signals.gather(self.voltage_data)
        refractory = signals.gather(self.refractory_data)
        resources = signals.gather(self.resources_data)
        calcium = signals.gather(self.calcium_data)

        spikes, voltage, refractory , resources, calcium = self._step(
            J, voltage, refractory, signals.dt, resources, calcium
        )

        signals.scatter(self.output_data, spikes)
        signals.mark_gather(self.J_data)
        signals.scatter(self.refractory_data, refractory)
        signals.scatter(self.voltage_data, voltage)
        signals.scatter(self.resources_data, resources)
        signals.scatter(self.calcium_data, calcium)

SimNeuronsBuilder.TF_NEURON_IMPL[stpLIF] = stpLIFBuilder

###  LEARNING RULE  ###

@Builder.register(SimSTP)
class SimSTPBuilder(OpBuilder):
    """Build a group of `.STP` operators."""

    def __init__(self, ops, signals, config):
        super(SimSTPBuilder, self).__init__(ops, signals, config)

        self.weights_data = signals.combine([op.weights for op in ops])
        self.output_data = signals.combine([op.delta for op in ops])

        self.resources_data = signals.combine([op.resources for op in ops])
        # self.resources_data = self.resources_data.reshape(self.resources_data.shape + (1,))


        self.calcium_data = signals.combine([op.calcium for op in ops])
        # self.calcium_data = self.calcium_data.reshape(self.calcium_data.shape + (1,))


        self.initial_weights = tf.constant(
            np.concatenate([op.initial_weights[:,:,None] for op in ops], axis = 1),
            signals.dtype
        )

        #
        self.initial_calcium = tf.constant(
            np.concatenate([op.initial_calcium[:,None] for op in ops], axis = 1), #removed extra Non
            signals.dtype
        )


    def build_step(self, signals):

        resources = signals.gather(self.resources_data)
        # print('resources shape', resources.shape)
        # print('resources type', resources.dtype)

        calcium = signals.gather(self.calcium_data)
        # print('calcium shape', calcium.shape)
        # print('calcium type', calcium.dtype)

        weights = signals.gather(self.weights_data)
        # print('weights shape', weights.shape)
        # print('weight type', weights.dtype)


        # print('U shape', self.initial_calcium.shape)
        # print('U type', self.initial_calcium.dtype)

        # print('init_weights shape', self.initial_weights.shape)
        # print('init weights type', self.initial_weights.dtype)

        # print('assignments complete')

        # delta = ((calcium * resources)/U) * init_weights - weights

        # tf.print(self.initial_calcium)
        # tf.get_static_value

        # print('delta = calcium * resources')
        delta = calcium * resources
        # print('calculation succesfull')

        # print('delta /= U')
        delta /= self.initial_calcium
        # print('calculation succesfull')

        # print('delta *= init_weights')
        delta *= self.initial_weights
        # print('calculation succesfull')

        # print('delta -= weights')
        delta -= weights


        # print('calculation succesfull')
        # print('delta shape', delta.shape)

        signals.scatter(self.output_data, delta)
