import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from nengo.dists import Uniform, Gaussian
import nengo
import math
from stp_dl_implementation import *
import os, inspect
from nengo_extras.vision import Gabor, Mask
from random import randint
import nengo.spa as spa
import os.path
import nengo_dl
from timeit import default_timer as timer

"""
Integrated Model 1: activity-silent model with content-specific 
reactivation to maintain information in one population and generic
reactivation to delete the information maintained in the primary module
following the first probe.

Memory population contains STP neurons, the recurrent connection on the 
memory population contains the STP learning rule and weights solver. 

Content-specific reactivation is achieved through the sensory maintenance
mechanism on the primary sensory population; the generic reactivation is 
fed to the memory population of the primary module after the first probe.
"""

#SIMULATION CONTROL for GUI
load_gabors_svd=True #set to false if you want to generate new ones
store_representations = False #store representations of model runs (for Fig 3 & 4)
store_decisions = False #store decision ensemble (for Fig 5 & 6)
store_spikes_and_resources = False #store spikes, calcium etc. (Fig 3 & 4)
store_memory = False #store neural output for Cross Temporal Analysis

#specify here which sim you want to run if you do not use the nengo GUI
#1 = representations & spikes
#2 = performance, decision signal
sim_to_run = 2
sim_no="2"      #simulation number (used in the names of the outputfiles)

# tf.config.list_physical_devices('GPU')
##batch processing did not work in exp1 for sim1, so check whether this works here
if sim_to_run==1:
    batch_processing=False
elif sim_to_run==2:
    batch_processing=True

#set this for use with nengo DL
if batch_processing:
    device="/gpu:2" #select GPU, use 0 (Nvidia 1) or 1 (Nvidia 3) for regular simulator, and 2 (Titan Xp 2) or 3 (Titan X (Pascal)) for batch processing
if not batch_processing:
    device="/gpu:1"


#MODEL PARAMETERS
D = 24  #dimensions of representations
Ns = 1000 #number of neurons in sensory layer
Nm = 1500 #number of neurons in memory layer
Nc = 1500 #number of neurons in comparison
Nd = 1000 #number of neurons in decision


#LOAD INPUT STIMULI (images created using the psychopy package)
#(Stimuli should be in a subfolder named 'Stimuli')

#width and height of images
diameter=col=row=128

#load grating stimuli
angles=np.arange(-90,90,1)  #rotation
phases=np.arange(0,1,0.1)   #phase

#stim2003: 60% grey
try:
    imagearr = np.load('Stimuli/all_stims_exp2.npy') #load stims if previously generated
except FileNotFoundError: #or generate
    imagearr=np.zeros((0,diameter**2))
    for phase in phases:
        for angle in angles:
            name="Stimuli/stim"+str(angle)+"_"+str(round(phase,1))+".png"
            img=Image.open(name)
            img=np.array(img.convert('L'))
            imagearr=np.vstack((imagearr,img.ravel()))

    #also load the  bull's eye 'impulse stimulus'
    name="Stimuli/stim2003.png"
    img=Image.open(name)
    img=np.array(img.convert('L'))
    imagearr=np.vstack((imagearr,img.ravel()))

    #normalize to be between -1 and 1
    imagearr=imagearr/255
    imagearr=2 * imagearr - 1

    #imagearr is a (1801, 16384) np array containing all stimuli + the impulse
    np.save('Stimuli/all_stims_exp2.npy',imagearr)


#INPUT FUNCTIONS

#set default input
memory_item_first = 0
probe_first = 0
memory_item_second = 0
probe_second = 0

#input stimuli 1
#250 ms memory items | 0-250
#950 ms fixation | 250-1200
#100 ms impulse | 1200-1300
#500 ms fixation | 1300-1800
#250 ms probe | 1800-2050
#1750 fixation | 2050-3800
#100 ms impulse | 3800-3900
#400 ms fixation | 3900-4300
#250 ms probe | 4300-4550
def input_func_first(t):
    if t > 0 and t < 0.25:
        return (imagearr[memory_item_first,:]/100) * 1.0
    elif t > 1.2 and t < 1.3:
        return imagearr[-1,:]/50 #impulse, twice the contrast of other items
    elif t > 1.8 and t < 2.05:
        return imagearr[probe_first,:]/100
    elif t > 3.8 and t < 3.9:
        return imagearr[-1,:]/50 #impulse, twice the contrast of other items
    #elif t > 4.3 and t < 4.55:
    #    return imagearr[probe_second,:]/100
    else:
        return np.zeros(128*128) #blank screen

def input_func_second(t):
    if t > 0 and t < 0.25:
        return (imagearr[memory_item_second,:]/100) * .9 #slightly lower input for secondary item
    elif t > 1.2 and t < 1.3:
        return imagearr[-1,:]/50 #impulse, twice the contrast of other items
    #elif t > 1.8 and t < 2.05:
    #    return imagearr[probe_first,:]/100
    elif t > 3.8 and t < 3.9:
        return imagearr[-1,:]/50 #impulse, twice the contrast of other items
    elif t > 4.3 and t < 4.55:
        return imagearr[probe_second,:]/100
    else:
        return np.zeros(128*128) #blank screen

#reactivate second ensemble based on lateralization
def reactivate_func(t):
    if t>2.250 and t<2.270:
        return np.ones(Nm)*0.0200
    else:
        return np.zeros(Nm)

#sine wave for the gated mechanism
f = 30.
w = 2. * np.pi * f
def gate_func(t):
    if t > 0.300 and t < 1.8:
        return np.ones(Ns)*np.sin(t*w)
    else:
        return np.ones(Ns)*(-1)

def clear_func(t):
    if t > 2.250 and t < 2.270:
        return np.ones(Nm)*0.02
    if t > 2.350 and t < 2.370:
        return np.ones(Nm)*0.02
    elif t > 2.450 and t < 2.470:
        return np.ones(Nm)*0.02
    elif t > 2.550 and t < 2.570:
        return np.ones(Nm)*0.02
    elif t > 2.650 and t < 2.670:
        return np.ones(Nm)*0.02
    elif t > 2.750 and t < 2.770:
        return np.ones(Nm)*0.02
    elif t > 2.850 and t < 2.870:
        return np.ones(Nm)*0.02
    elif t > 2.950 and t < 2.970:
        return np.ones(Nm)*0.02
    elif t > 3.050 and t < 3.070:
        return np.ones(Nm)*0.02
    elif t > 3.150 and t < 3.170:
        return np.ones(Nm)*0.02
    elif t > 3.250 and t < 3.270:
        return np.ones(Nm)*0.02
    elif t > 2.350 and t < 3.370:
        return np.ones(Nm)*0.02
    elif t > 3.450 and t < 3.470:
        return np.ones(Nm)*0.02
    elif t > 3.550 and t < 3.570:
        return np.ones(Nm)*0.02
    elif t > 3.650 and t < 3.670:
        return np.ones(Nm)*0.02
    elif t > 3.750 and t < 3.770:
        return np.ones(Nm)*0.02
    elif t > 3.850 and t < 3.870:
        return np.ones(Nm)*0.02
    elif t > 3.950 and t < 3.970:
        return np.ones(Nm)*0.02
    elif t > 4.050 and t < 4.070:
        return np.ones(Nm)*0.02
    elif t > 4.150 and t < 4.170:
        return np.ones(Nm)*0.02
    elif t > 4.250 and t < 4.270:
        return np.ones(Nm)*0.02
    else:
        return np.zeros(Nm)

#create data for batch processing using the Nengo DL simulator
def get_data(inputs_first, inputs_second, inputs_reactivate, inputs_gate, inputs_clear, initialangles, n_trials, probelist, res):
    start = timer()

    #trials come in sets of 12, which we call a run (all possible orientation differences between memory and probe),
    runs = int(n_trials / 12)

    for run in range(runs):

        #run a trial with each possible orientation difference
        for cnt_in_run, anglediff in enumerate(probelist):

            i = run * 12 + cnt_in_run

            #set probe and stim
            memory_item_first=randint(0, 179) #random memory
            probe_first=memory_item_first+anglediff #probe based on that
            probe_first=norm_p(probe_first) #normalise probe

            #random phase
            or_memory_item_first=memory_item_first #original
            memory_item_first=memory_item_first+(180*randint(0, 9))
            probe_first=probe_first+(180*randint(0, 9))

            #same for secondary item
            memory_item_second = memory_item_first
            probe_second = probe_first

            initialangles[i] = or_memory_item_first

            inputs_first[i,0:int(250/res),:]=(imagearr[memory_item_first,:]/100) * 1.0
            inputs_first[i,int(1800/res):int(2050/res),:]=imagearr[probe_first,:]/100

            inputs_second[i,0:int(250/res),:]=(imagearr[memory_item_second,:]/100) * 0.9
            inputs_second[i,int(4300/res):int(4550/res),:]=imagearr[probe_second,:]/100

    #the impulses are the same for each trial
    inputs_first[:,int(1200/res):int(1300/res),:]=imagearr[-1,:]/50
    inputs_first[:,int(3800/res):int(3900/res),:]=imagearr[-1,:]/50

    inputs_second[:,int(1200/res):int(1300/res),:]=imagearr[-1,:]/50
    inputs_second[:,int(3800/res):int(3900/res),:]=imagearr[-1,:]/50

    #reactivation is the same for each trial
    inputs_reactivate[:,int(2250/res):int(2270/res),:] = np.ones(Nm)*0.0200

    #inputs gate is the same for each trial 
    Fs = 750
    f = 30
    x = np.arange(Fs)
    sine_input = np.sin(2 * np.pi * f * x / Fs)

    for neuron in range(Ns):
        inputs_gate[:,int(300/res):int(1800/res),neuron] = sine_input

    #clear_func is the same on each trial
    clear_start = range(2250, 4300, 100)
    clear_stop = range(2270, 4320, 100)
    for start_, stop_ in zip(clear_start, clear_stop):
        inputs_clear[:,int(start_/res):int(stop_/res),:] = np.ones(Nm)*0.02

    end = timer()

    print("Data generation for %d trials lasted: %d s" % (n_trials, end-start) )

#Create matrix of sine and cosine values associated with the stimuli
#so that we can later specify a transform from stimuli to rotation
Fa = np.tile(angles,phases.size) #want to do this for each phase
Frad = (Fa/90) * math.pi #make radians
Sin = np.sin(Frad)
Cos = np.cos(Frad)
sincos = np.vstack((Sin,Cos)) #sincos

#Create eval points so that we can go from sine and cosine of theta in sensory and memory layer
#to the difference in theta between the two
samples = 10000
sinAcosA = nengo.dists.UniformHypersphere(surface=True).sample(samples,2)
thetaA = np.arctan2(sinAcosA[:,0],sinAcosA[:,1])
thetaDiff = (90*np.random.random(samples)-45)/180*np.pi
thetaB = thetaA + thetaDiff

sinBcosB = np.vstack((np.sin(thetaB),np.cos(thetaB)))
scale = np.random.random(samples)*0.9+0.1
sinBcosB = sinBcosB * scale
ep = np.hstack((sinAcosA,sinBcosB.T))


#continuous variant of arctan(a,b)-arctan(c,d)
def arctan_func(v):
    yA, xA, yB, xB = v
    z = np.arctan2(yA, xA) - np.arctan2(yB, xB)
    pos_ans = [z, z+2*np.pi, z-2*np.pi]
    i = np.argmin(np.abs(pos_ans))
    return pos_ans[i]*90/math.pi

#MODEL

#gabor generation for a particular model-participant
def generate_gabors():

    global e_first
    global U_first
    global compressed_im_first

    global e_second
    global U_second
    global compressed_im_second

    #to speed things up, load previously generated ones
    if load_gabors_svd & os.path.isfile('Stimuli/gabors_svd_first_exp2.npz'):
        gabors_svd_first = np.load('Stimuli/gabors_svd_first_exp2.npz') #load stims if previously generated
        e_first = gabors_svd_first['e_first']
        U_first = gabors_svd_first['U_first']
        compressed_im_first = gabors_svd_first['compressed_im_first']
        print("SVD first loaded")

    else: #or generate and save

        #cued module
        #for each neuron in the sensory layer, generate a Gabor of 1/3 of the image size
        gabors_first = Gabor().generate(Ns, (int(col/3), int(row/3)))
        #put gabors on image and make them the same shape as the stimuli
        gabors_first = Mask((col, row)).populate(gabors_first, flatten=True).reshape(Ns, -1)
        #normalize
        gabors_first=gabors_first/abs(max(np.amax(gabors_first),abs(np.amin(gabors_first))))
        #gabors are added to imagearr for SVD
        x_first=np.vstack((imagearr,gabors_first))

        #SVD
        print("SVD first started...")
        U_first, S_first, V_first = np.linalg.svd(x_first.T)
        print("SVD first done")

        #Use result of SVD to create encoders
        e_first = np.dot(gabors_first, U_first[:,:D]) #encoders
        compressed_im_first = np.dot(imagearr[:1800,:]/100, U_first[:,:D]) #D-dimensional vector reps of the images
        compressed_im_first = np.vstack((compressed_im_first, np.dot(imagearr[-1,:]/50, U_first[:,:D])))

        np.savez('Stimuli/gabors_svd_first_exp2.npz', e_first=e_first, U_first=U_first, compressed_im_first=compressed_im_first)

    #same for secondary module

    if load_gabors_svd & os.path.isfile('Stimuli/gabors_svd_second_exp2.npz'):
        gabors_svd_second = np.load('Stimuli/gabors_svd_second_exp2.npz') #load stims if previously generated
        e_second = gabors_svd_second['e_second']
        U_second = gabors_svd_second['U_second']
        compressed_im_second = gabors_svd_second['compressed_im_second']
        print("SVD second loaded")
    else:
        gabors_second = Gabor().generate(Ns, (int(col/3), int(row/3)))#.reshape(N, -1)
        gabors_second = Mask((col, row)).populate(gabors_second, flatten=True).reshape(Ns, -1)
        gabors_second=gabors_second/abs(max(np.amax(gabors_second),abs(np.amin(gabors_second))))
        x_second=np.vstack((imagearr,gabors_second))

        print("SVD second started...")
        U_second, S_second, V_second = np.linalg.svd(x_second.T)
        print("SVD second done")
        e_second = np.dot(gabors_second, U_second[:,:D])
        compressed_im_second=np.dot(imagearr[:1800,:]/100, U_second[:,:D])
        compressed_im_second = np.vstack((compressed_im_second, np.dot(imagearr[-1,:]/50, U_second[:,:D])))

        np.savez('Stimuli/gabors_svd_second_exp2.npz', e_second=e_second, U_second=U_second, compressed_im_second=compressed_im_second)

nengo_gui_on = __name__ == 'builtins' #python3

def create_model(seed=None):

    global model 

    #create vocabulary to show representations in gui
    if nengo_gui_on:
        vocab_angles = spa.Vocabulary(D)
        for name in [0, 5, 10, 16, 24, 32, 40]:
            #vocab_angles.add('D' + str(name), np.linalg.norm(compressed_im_first[name+90])) #take mean across phases
            v = compressed_im_first[name+90]
            nrm = np.linalg.norm(v)
            if nrm > 0:
                v /= nrm
            vocab_angles.add('D' + str(name), v) #take mean across phases

        v = np.dot(imagearr[-1,:]/50, U_first[:,:D])
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v /= nrm
        vocab_angles.add('Impulse', v)

    #model = nengo.Network(seed=seed)
    model = spa.SPA(seed=seed)
    with model:

        #input nodes
        if not batch_processing:
            model.inputNode_first=nengo.Node(input_func_first,label='input_first')
            model.gateNode=nengo.Node(gate_func,label="gate")
            model.clearNode=nengo.Node(clear_func,label="clearance")
        elif batch_processing:
            model.inputNode_first=nengo.Node(np.zeros(128*128))
            model.gateNode=nengo.Node(np.zeros(Ns))
            model.clearNode=nengo.Node(np.zeros(Nm))

        #eye ensemble
        eye_first = nengo.Ensemble(Ns, D, encoders=e_first, intercepts=Uniform(0.01, 0.1),radius=1, label='eye_first', seed=seed)
        nengo.Connection(model.inputNode_first, eye_first, transform=U_first[:,:D].T)

        #sensory ensemble
        sensory_first = nengo.Ensemble(Ns, D, encoders=e_first, intercepts=Uniform(0.01, .1),radius=1,label='sensory_first')

        conn_first = nengo.Connection(eye_first, sensory_first, function=lambda x: x) #default: identity function

        if nengo_gui_on:
            with nengo.simulator.Simulator(network=model, progress_bar=False) as sim_first:
                weights_first = sim_first.data[conn_first].weights

            model.connections.remove(conn_first)

        elif not(nengo_gui_on):
            with nengo_dl.Simulator(network=model, progress_bar=False, device=device) as sim_first: #, device=device
                weights_first = sim_first.data[conn_first].weights

            model.connections.remove(conn_first)

        Ncluster=1 # number of neurons per cluster
        clusters=np.arange(0, Ns, Ncluster)

        #parameters for gamma distribution where k=shape, theta=scale and mean=k*theta -> theta=mean/k
        # wanted distribution: lowest value 51ms and mean value 141 ms (Lamme & Roelfsema (2000))
        
        k, mean = 2, 0.090    #(mean_synapse - lowest_synapse) #subtract lowest synapse from mean so distribution starts at 0
        theta = mean / k
        shift = 0.051

        weights_trans = 1.50
        rec_trans = 0.30

        noise_sd = 0.010

        if(True):
            # Build second simulation to fetch weights before making the large amount of connections between eye_first and sensory_first, place all
            # of second model inside this block statement to keep code close together
            if not batch_processing:
                model.inputNode_second=nengo.Node(input_func_second,label='input_second')
                model.reactivate=nengo.Node(reactivate_func,label='reactivate')
            elif batch_processing:
                model.inputNode_second=nengo.Node(np.zeros(128*128))
                model.reactivate=nengo.Node(np.zeros(Nm))

            eye_second = nengo.Ensemble(Ns, D, encoders=e_second, intercepts=Uniform(0.01, 0.1),radius=1, label='eye_second', seed=seed)
            nengo.Connection(model.inputNode_second, eye_second, transform=U_second[:,:D].T)

            sensory_second = nengo.Ensemble(Ns, D, encoders=e_second, intercepts=Uniform(0.01, .1),radius=1,label='sensory_second')

            conn_second = nengo.Connection(eye_second, sensory_second, function=lambda x: x) #default: identity function

            if nengo_gui_on:
                with nengo.simulator.Simulator(network=model, progress_bar=False) as sim_second:
                    weights_second = sim_second.data[conn_second].weights

                model.connections.remove(conn_second)

            elif not(nengo_gui_on):
                with nengo_dl.Simulator(network=model, progress_bar=False, device=device) as sim_second: #device=device
                    weights_second = sim_second.data[conn_second].weights

                model.connections.remove(conn_second)

            synapse_second = np.random.gamma(k, theta, clusters.size) + shift

            for i in range(clusters.size):

                begin=clusters[i]
                end=(begin+Ncluster)

                nengo.Connection(eye_second.neurons[begin:end], sensory_second, transform=weights_second[:,begin:end]*weights_trans, synapse=synapse_second[i])

            nengo.Connection(sensory_second, eye_second, transform = rec_trans, solver=nengo.solvers.LstsqL2(weights=True), synapse=0.200)

            noise_second= nengo.processes.WhiteNoise(dist=Gaussian(0,noise_sd))

            memory_second = nengo.Ensemble(Nm, D,neuron_type=stpLIF(), intercepts=Uniform(0.01, 0.1), noise=noise_second, radius=1,label='memory_second')
            nengo.Connection(sensory_second, memory_second, transform=0.1)
            nengo.Connection(model.reactivate,memory_second.neurons) #potential reactivation

            nengo.Connection(memory_second, memory_second,transform=1,learning_rule_type=STP(),solver=nengo.solvers.LstsqL2(weights=True))

            comparison_second = nengo.Ensemble(Nd, dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1),label='comparison_second')

            nengo.Connection(memory_second, comparison_second[2:],eval_points=compressed_im_second[0:-1],function=sincos.T)
            nengo.Connection(sensory_second, comparison_second[:2],eval_points=compressed_im_second[0:-1],function=sincos.T)

            decision_second = nengo.Ensemble(n_neurons=Nd,  dimensions=1,radius=45,label='decision_second')
            nengo.Connection(comparison_second, decision_second, eval_points=ep, scale_eval_points=False, function=arctan_func)

        ### first ###

        synapse_first = np.random.gamma(k, theta, clusters.size) + shift

        for i in range(clusters.size):

            begin=clusters[i]
            end=(begin+Ncluster)

            nengo.Connection(eye_first.neurons[begin:end], sensory_first, transform=weights_first[:,begin:end]*weights_trans, synapse=synapse_first[i])

        nengo.Connection(sensory_first, eye_first, transform = rec_trans, solver=nengo.solvers.LstsqL2(weights=True), synapse=0.200)

        #gating mechanism
        gate = nengo.Ensemble(Ns, D, intercepts=nengo.dists.Uniform(0.01, 0.1),radius=1, label="gate")
        nengo.Connection(model.gateNode, gate.neurons)#,transform=np.ones((Ns,1)))
        nengo.Connection(sensory_first, gate, synapse=.05, transform = 1.0) #play with these transform values (eg 1.2/1.2 with gatefunc3)
        nengo.Connection(gate, sensory_first, synapse=.05, transform = 0.8) #play with these transform values or 1.0/.8 with gatefunc4

        #memory ensemble
        noise_first= nengo.processes.WhiteNoise(dist=Gaussian(0,noise_sd))

        memory_first = nengo.Ensemble(Nm, D,neuron_type=stpLIF(), intercepts=Uniform(0.01, 0.1), noise=noise_first, radius=1,label='memory_first')
        nengo.Connection(sensory_first, memory_first, transform=0.1) #.1)
        nengo.Connection(model.clearNode, memory_first.neurons)

        #recurrent STSP connection
        nengo.Connection(memory_first, memory_first,transform=1, learning_rule_type=STP(), solver=nengo.solvers.LstsqL2(weights=True))

        #comparison represents sin, cosine of theta of both sensory and memory ensemble
        comparison_first = nengo.Ensemble(Nc, dimensions=4,radius=math.sqrt(2),intercepts=Uniform(.01, 1),label='comparison_first')
        nengo.Connection(sensory_first, comparison_first[:2],eval_points=compressed_im_first[0:-1],function=sincos.T)
        nengo.Connection(memory_first, comparison_first[2:],eval_points=compressed_im_first[0:-1],function=sincos.T)

        #decision represents the difference in theta decoded from the sensory and memory ensembles
        decision_first = nengo.Ensemble(n_neurons=Nd,  dimensions=1,radius=45,label='decision_first')
        nengo.Connection(comparison_first, decision_first, eval_points=ep, scale_eval_points=False, function=arctan_func)


        #decode for gui
        if nengo_gui_on:
            model.sensory_decode = spa.State(D, vocab=vocab_angles, subdimensions=12, label='sensory_decode')
            for ens in model.sensory_decode.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(sensory_first, model.sensory_decode.input,synapse=None)

            model.memory_decode = spa.State(D, vocab=vocab_angles, subdimensions=12, label='memory_decode')
            for ens in model.memory_decode.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(memory_first, model.memory_decode.input,synapse=None)

        #probes
        if not(nengo_gui_on):
            if store_representations: #sim 1 trials 1-100

                model.p_mem_first=nengo.Probe(memory_first, synapse=0.01)
                model.p_mem_second=nengo.Probe(memory_second, synapse=0.01)

            if store_spikes_and_resources: #sim 1 trial 1
                model.p_spikes_mem_first=nengo.Probe(memory_first.neurons, 'spikes')
                model.p_res_first=nengo.Probe(memory_first.neurons, 'resources')
                model.p_cal_first=nengo.Probe(memory_first.neurons, 'calcium')

                model.p_spikes_mem_second=nengo.Probe(memory_second.neurons, 'spikes')
                model.p_res_second=nengo.Probe(memory_second.neurons, 'resources')
                model.p_cal_second=nengo.Probe(memory_second.neurons, 'calcium')

            if store_decisions: #sim 2
                model.p_dec_first=nengo.Probe(decision_first, synapse=0.01)
                model.p_dec_second=nengo.Probe(decision_second, synapse=0.01)

            if store_memory:
                model.p_mem_first_raw=nengo.Probe(memory_first.neurons, 'output', synapse=0.01)
                model.p_mem_second_raw=nengo.Probe(memory_second.neurons, 'output', synapse=0.01)

    return model

#PLOTTING CODE
from nengo.utils.matplotlib import rasterplot
from matplotlib import style
from plotnine import *
theme = theme_classic()
plt.style.use('default')

def plot_sim_1(sp_1,sp_2,res_1,res_2,cal_1,cal_2, mem_1, mem_2):

    figure_name = 'Sim_1/full_model/exp2_integrated_model/exp2_figure'

    #representations  & spikes
    with plt.rc_context():
        plt.rcParams.update(theme.rcParams)

        fig, axes, = plt.subplots(2,2,squeeze=True)
        theme.setup_figure(fig)
        t = sim.trange()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        #spikes, calcium, resources first
        ax1=axes[0,0]
        ax1.set_title("First Module")
        ax1.set_ylabel('# cell', color='black')
        ax1.set_yticks(np.arange(0,Nm,500))
        ax1.tick_params('y')#, colors='black')
        rasterplot(sim.trange(), sp_1,ax1,colors=['black']*sp_1.shape[0])
        ax1.set_xticklabels([])
        ax1.set_xticks([])
        ax1.set_xlim(0,4.6)
        ax2 = ax1.twinx()
        ax2.plot(t, res_1, "#00bfc4",linewidth=2)
        ax2.plot(t, cal_1, "#e38900",linewidth=2)
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        ax2.set_ylim(0,1.1)

        #spikes, calcium, resources second
        ax3=axes[0,1]
        ax3.set_title("Second Module")
        rasterplot(sim.trange(), sp_2,ax3,colors=['black']*sp_2.shape[0])
        ax3.set_xticklabels([])
        ax3.set_xticks([])
        ax3.set_yticklabels([])
        ax3.set_yticks([])
        ax3.set_xlim(0,4.6)
        ax4 = ax3.twinx()
        ax4.plot(t, res_2, "#00bfc4",linewidth=2)
        ax4.plot(t, cal_2, "#e38900",linewidth=2)
        ax4.set_ylabel('synaptic variables', color="black",size=11)
        ax4.tick_params('y', labelcolor='#333333',labelsize=9,color='#333333')
        ax4.set_ylim(0,1.1)

        #representations first
        plot_mc=axes[1,0]
        plot_mc.plot(sim.trange(),(mem_1));

        plot_mc.set_ylabel("Cosine similarity")
        plot_mc.set_xticks(np.arange(0.0,4.6,0.5))
        plot_mc.set_xticklabels(np.arange(0,4600,500).tolist())
        plot_mc.set_xlabel('time (ms)')
        plot_mc.set_xlim(0,4.6)
        colors=["#00c094","#00bfc4","#00b6eb","#06a4ff","#a58aff","#df70f8","#fb61d7", "#c49a00"]
        for i,j in enumerate(plot_mc.lines):
            j.set_color(colors[i])

        #representations uncued
        plot_mu=axes[1,1]

        plot_mu.plot(sim.trange(),(mem_2));
        plot_mu.set_xticks(np.arange(0.0,4.6,0.5))
        plot_mu.set_xticklabels(np.arange(0,4600,500).tolist())
        plot_mu.set_xlabel('time (ms)')
        plot_mu.set_yticks([])
        plot_mu.set_yticklabels([])
        plot_mu.set_xlim(0,4.6)
        for i,j in enumerate(plot_mu.lines):
            j.set_color(colors[i])
        plot_mu.legend(["0°","5°","10°","16°","24°","32°","40°", "Impulse"], title="Stimulus", bbox_to_anchor=(1.02, -0.25, .30, 0.8), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)

        fig.set_size_inches(11, 5)
        theme.apply(plt.gcf().axes[0])
        theme.apply(plt.gcf().axes[1])
        theme.apply(plt.gcf().axes[2])
        theme.apply(plt.gcf().axes[3])
        plt.savefig(figure_name + '_3.eps', format='eps', dpi=1000)
        plt.show()


    # impulse 1
    with plt.rc_context():
        plt.rcParams.update(theme.rcParams)

        fig, axes, = plt.subplots(1,2,squeeze=True)
        theme.setup_figure(fig)
        t = sim.trange()
        plt.subplots_adjust(wspace=0.1, hspace=0.05)

        plot_mc=axes[0]
        plot_mc.set_title("First Module")
        plot_mc.plot(sim.trange(),(mem_1));
        plot_mc.set_ylabel("Cosine similarity")
        plot_mc.set_xticks(np.arange(1.2,1.4,0.05))
        plot_mc.set_xticklabels(np.arange(0,200,50).tolist())
        plot_mc.set_xlabel('time after onset impulse (ms)')
        plot_mc.set_xlim(1.2,1.35)
        plot_mc.set_ylim(0,0.9)
        colors=["#00c094","#00bfc4","#00b6eb","#06a4ff","#a58aff","#df70f8","#fb61d7", "#c49a00"]
        for i,j in enumerate(plot_mc.lines):
            j.set_color(colors[i])

        plot_mu=axes[1]
        plot_mu.set_title("Second Module")
        plot_mu.plot(sim.trange(),(mem_2));
        plot_mu.set_xticks(np.arange(1.2,1.4,0.05))
        plot_mu.set_xticklabels(np.arange(0,200,50).tolist())
        plot_mu.set_xlabel('time after onset impulse (ms)')
        plot_mu.set_yticks([])
        plot_mu.set_yticklabels([])
        plot_mu.set_xlim(1.2,1.35)
        plot_mu.set_ylim(0,0.9)
        for i,j in enumerate(plot_mu.lines):
            j.set_color(colors[i])
        plot_mu.legend(["0°","5°","10°","16°","24°","32°","40°", "Impulse"], title="Stimulus", bbox_to_anchor=(0.85, 0.25, .55, 0.8), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)

        fig.set_size_inches(6, 4)

        theme.apply(plt.gcf().axes[0])
        theme.apply(plt.gcf().axes[1])
        plt.savefig(figure_name + '_Impulse1_4.eps', format='eps', dpi=1000)
        plt.show()

    # impulse 2
    with plt.rc_context():
        plt.rcParams.update(theme.rcParams)

        fig, axes, = plt.subplots(1,2,squeeze=True)
        theme.setup_figure(fig)
        t = sim.trange()
        plt.subplots_adjust(wspace=0.1, hspace=0.05)

        plot_mc=axes[0]
        plot_mc.set_title("First Module")
        plot_mc.plot(sim.trange(),(mem_1));
        plot_mc.set_ylabel("Cosine similarity")
        plot_mc.set_xticks(np.arange(3.8,3.95,0.05))
        plot_mc.set_xticklabels(np.arange(0,200,50).tolist())
        plot_mc.set_xlabel('time after onset impulse (ms)')
        plot_mc.set_xlim(3.8,3.95)
        plot_mc.set_ylim(0,0.9)
        colors=["#00c094","#00bfc4","#00b6eb","#06a4ff","#a58aff","#df70f8","#fb61d7", "#c49a00"]
        for i,j in enumerate(plot_mc.lines):
            j.set_color(colors[i])

        plot_mu=axes[1]
        plot_mu.set_title("Second Module")
        plot_mu.plot(sim.trange(),(mem_2));
        plot_mu.set_xticks(np.arange(3.8,3.95,0.05))
        plot_mu.set_xticklabels(np.arange(0,200,50).tolist())
        plot_mu.set_xlabel('time after onset impulse (ms)')
        plot_mu.set_yticks([])
        plot_mu.set_yticklabels([])
        plot_mu.set_xlim(3.8,3.95)
        plot_mu.set_ylim(0,0.9)
        for i,j in enumerate(plot_mu.lines):
            j.set_color(colors[i])
        plot_mu.legend(["0°","5°","10°","16°","24°","32°","40°", "Impulse"], title="Stimulus", bbox_to_anchor=(0.85, 0.25, .55, 0.8), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)

        fig.set_size_inches(6, 4)

        theme.apply(plt.gcf().axes[0])
        theme.apply(plt.gcf().axes[1])
        plt.savefig(figure_name + '_Impulse2_4.eps', format='eps', dpi=1000)
        plt.show()



#SIMULATION
#note that this is split for running a single trial in the nengo gui, and a full simulation

#normalise stimuli to be between 0 and 180 degrees orientation
def norm_p(p):
    if p<0:
        return 180+p
    if p>180:
        return p-180
    else:
        return p

#Calculate normalised cosine similarity and avoid divide by 0 errors
def cosine_sim(a,b):
    out=np.zeros(a.shape[0])
    for i in range(0,  a.shape[0]):
        if abs(np.linalg.norm(a[i])) > 0.05:
            out[i]=abs(np.dot(a[i], b)/(np.linalg.norm(a[i])*np.linalg.norm(b)))
    return out

if nengo_gui_on:
    generate_gabors() #generate gabors

    with nengo.Network() as model:
        create_model(seed=0) #build model

    memory_item_first = 0 + 90
    probe_first = 40 + 90
    memory_item_second = 0 + 90
    probe_second = 40 + 90

else: #no gui

    #path
    cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+'/data_exp2/' #store output in data subfolder

    #simulation 1
    if sim_to_run == 1:

        print('Running simulation 1')
        print('')

        batch_processing = False

        load_gabors_svd = False#no need to randomize this

        ntrials = 10
        store_representations = True
        store_decisions = False

        #store results
        templates = np.array([0, 5, 10, 16, 24, 32, 40]) + 90
        mem_1 = np.zeros((4600,len(templates)+1)) #keep cosine sim for 9 items
        mem_2 = np.zeros((4600,len(templates)+1))

        #first, run 100 trials to get average cosine sim
        for run in range(ntrials):

            print('Run ' + str(run+1))

            #stimuli

            phase = 180*(run % 10)
            memory_item_first = 0 + 90 + phase
            probe_first = 40 + 90 + phase
            memory_item_second = 0 + 90 + phase
            probe_second = 40 + 90 + phase

            #create new gabor filters every 10 trials
            if run % 10 == 0:
                generate_gabors()

            model = create_model(seed=run)
            sim = nengo_dl.Simulator(network=model, seed=run, progress_bar=False, device=device)

            #run simulation
            sim.run(4.6)

            #reset simulator, clean probes thoroughly
            #print(sim.data[model.p_mem_cued].shape)
            #calc cosine sim with templates
            temp_phase = list(templates + phase) + [1800]
            for cnt, templ in enumerate(temp_phase):
                mem_1[:,cnt] += cosine_sim(sim.data[model.p_mem_first][:,:,],compressed_im_first[templ,:])
                mem_2[:,cnt] += cosine_sim(sim.data[model.p_mem_second][:,:,],compressed_im_second[templ,:])

            sim.reset()
            sim.close()

        #average
        mem_1 /= ntrials
        mem_2 /= ntrials

        #second, run 1 trial to get calcium and spikes
        store_spikes_and_resources = True
        store_representations = False
        model = create_model(seed=0) #recreate model to change probes
        sim = nengo_dl.Simulator(network=model, seed=0, progress_bar=False, device=device)

        print('Run ' + str(ntrials+1))
        sim.run(4.6)

        #store spikes and calcium
        sp_1 = sim.data[model.p_spikes_mem_first]
        res_1=np.mean(sim.data[model.p_res_first][:,:,],1) #take mean over neurons
        cal_1=np.mean(sim.data[model.p_cal_first][:,:,],1) #take mean over neurons

        sp_2=sim.data[model.p_spikes_mem_second]
        res_2=np.mean(sim.data[model.p_res_second][:,:,],1)
        cal_2=np.mean(sim.data[model.p_cal_second][:,:,],1)

        sim.close()

        #plot
        plot_sim_1(sp_1,sp_2,res_1,res_2,cal_1,cal_2, mem_1, mem_2)


    #simulation 2: collect data for fig 5 & 6. 1728 trials for 19 subjects
    if sim_to_run == 2:

        load_gabors_svd = False #set to false for real simulation

        data_path = "/Users/s3344282/"

        #set to 1 for (default) simulator dt of 0.001, set to 2 for simulator dt of 0.002, and so on. the number of time steps should be divisible by this number to prevent errors.
        res = 2 #resolution

        n_subj =  19 #19
        trials_per_subj = 1728
        store_representations = False
        store_decisions = True
        store_memory = True

        split = 6 #we split the neuron data into this many separate files to decrease the cpu memory use, set to 1 for no split

        #division in the form of n_trials/minibatch_size/split should yield a whole integer, otherwise indexing errors will occur and/or the number of trials performed will not be as specified.

        minibatch_size=12 #12 takes about 5-6 gb of GPU, lowest batch possible as probelist contains 12 items, higher values should be multiples of this
        n_steps=int(4600/res)

        #orientation differences between probe and memory item for each run
        probelist=[-40, -32, -24, -16, -10, -5, 5, 10, 16, 24, 32, 40]

        #arrays for storing input data
        input_data_first = np.zeros((minibatch_size, n_steps, 128*128))
        input_data_second = np.zeros((minibatch_size, n_steps, 128*128))
        input_data_reactivate = np.zeros((minibatch_size, n_steps, Nm))
        input_data_gate = np.ones((minibatch_size, n_steps, Ns))*(-1)
        input_data_clear = np.zeros((minibatch_size, n_steps, Nm))
        data_initialangles = np.zeros(minibatch_size)

        if store_memory:
            initialangles = np.zeros(int(trials_per_subj/split))
            neuro_mem_first = np.zeros((int(trials_per_subj/split), n_steps, Nm))
            neuro_mem_second = np.zeros((int(trials_per_subj/split), n_steps, Nm))

        for subj in range(n_subj):

            split_index=0

            #create new gabor filters and model for each new participant
            generate_gabors()
            model = create_model(seed=subj)

            #use Nengo DL simulator to make use of the Nengo DL implementation of STSP and batch processing, set dt in accordance with resolution.
            sim = nengo_dl.Simulator(network=model, minibatch_size=minibatch_size, seed=subj, device=device, progress_bar=False, dt=0.002)

            n_batches = int(trials_per_subj/minibatch_size)

            for batch in range(n_batches):

                print("Subject " + str(subj + 1) + "/" + str(n_subj) + ": batch " + str(batch+1) + "/" + str(n_batches) + ": " + str(minibatch_size) + " trials in batch")

                get_data(input_data_first, input_data_second, input_data_reactivate, input_data_gate, input_data_clear, data_initialangles, minibatch_size, probelist, res)

                #run simulation
                sim.run_steps(
                    n_steps,
                    data = {
                        model.inputNode_first: input_data_first,
                        model.inputNode_second: input_data_second,
                        model.reactivate: input_data_reactivate,
                        model.gateNode: input_data_gate,
                        model.clearNode: input_data_clear
                    },
                )

                #store performance data (sim 2)
                if store_decisions:
                    for in_batch_trial in range(minibatch_size):

                        anglediff = probelist[in_batch_trial%12]

                        trial = batch * minibatch_size + in_batch_trial + 1

                        np.savetxt(data_path+"Performance/full_model/exp2_integrated_model/"+sim_no+"_Diff_Theta_%i_subj_%i_trial_%i_probe1.csv" % (anglediff, subj+1, trial),
                            sim.data[model.p_dec_first][in_batch_trial,int((1800)/res):int((2100)/res),:], delimiter=",")

                        np.savetxt(data_path+"Performance/full_model/exp2_integrated_model/"+sim_no+"_Diff_Theta_%i_subj_%i_trial_%i_probe2.csv" % (anglediff, subj+1, trial),
                            sim.data[model.p_dec_second][in_batch_trial,int((4300)/res):int((4600)/res),:], delimiter=",")

                #store neuron data (sim 3)
                if store_memory:
                    trial_index_start=int((batch*minibatch_size)%(trials_per_subj/split))
                    trial_index_end=trial_index_start+minibatch_size

                    initialangles[trial_index_start:trial_index_end]=data_initialangles

                    neuro_mem_first[trial_index_start:trial_index_end, :, :] = sim.data[model.p_mem_first_raw]
                    neuro_mem_second[trial_index_start:trial_index_end, :, :] = sim.data[model.p_mem_second_raw]

                    #use this for-loop is you want to split the neuron data to separate files, otherwise use the commented out code above above sim.close()
                    if (batch+1)%(n_batches/split)==0:

                        np.save(data_path+"Decoding/full_model/exp2_integrated_model/subj_%i_mem_neuron_data_first_%i.npy" % (subj+1, split_index), neuro_mem_first)
                        np.save(data_path+"Decoding/full_model/exp2_integrated_model/subj_%i_mem_neuron_data_second_%i.npy" % (subj+1, split_index), neuro_mem_second)
                        np.save(data_path+"Decoding/full_model/exp2_integrated_model/subj_%i_initial_angles_%i.npy" % (subj+1, split_index), initialangles)

                        initialangles[:]=0
                        neuro_mem_first[:,:,:]=0
                        neuro_mem_second[:,:,:]=0

                        split_index+=1

                        print("Neuron dataset %i/%i stored" % (split_index, split))

                #reset simulator
                sim.reset()

            sim.close()

