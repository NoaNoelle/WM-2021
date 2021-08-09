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
Model 4: Alternative maintenance mechanism in which a recurrent connection on the
sensory population maintains the information, but the pattern is 'gated'
by a gate node that suppresses the pattern on the basis of a 30Hz sine wave.
"""

#SIMULATION CONTROL for GUI
uncued=False #set if you want to run both the cued and uncued model
load_gabors_svd=False #set to false if you want to generate new ones
store_representations = False #store representations of model runs (for Fig 3 & 4)
store_decisions = False #store decision ensemble (for Fig 5 & 6)
store_spikes_and_resources = False #store spikes, calcium etc. (Fig 3 & 4)

store_sensory = False
store_memory = False

#specify here which sim you want to run if you do not use the nengo GUI
#1 = simulation to generate Fig 3 & 4
#2 = simulation to generate Fig 5 & 6
sim_to_run = 2
sim_no="2"      #simulation number (used in the names of the outputfiles)

#tf.config.list_physical_devices('GPU')
if sim_to_run==1:
    batch_processing=False
elif sim_to_run==2:
    batch_processing=True

#set this for use with nengo DL
if batch_processing:
    device="/gpu:1" #select GPU, use 0 (Nvidia 1) or 1 (Nvidia 3) for regular simulator, and 2 (Titan Xp 2) or 3 (Titan X (Pascal)) for batch processing
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

try:
    imagearr = np.load('Stimuli/all_stims.npy') #load stims if previously generated
except FileNotFoundError: #or generate
    imagearr=np.zeros((0,diameter**2))
    for phase in phases:
        for angle in angles:
            name="Stimuli/stim"+str(angle)+"_"+str(round(phase,1))+".png"
            img=Image.open(name)
            img=np.array(img.convert('L'))
            imagearr=np.vstack((imagearr,img.ravel()))

    #also load the  bull's eye 'impulse stimulus'
    name="Stimuli/stim999.png"
    img=Image.open(name)
    img=np.array(img.convert('L'))
    imagearr=np.vstack((imagearr,img.ravel()))

    #normalize to be between -1 and 1
    imagearr=imagearr/255
    imagearr=2 * imagearr - 1

    #imagearr is a (1801, 16384) np array containing all stimuli + the impulse
    np.save('Stimuli/all_stims.npy',imagearr)


#INPUT FUNCTIONS

#set default input
memory_item_cued = 0
probe_cued = 0
memory_item_uncued = 0
probe_uncued = 0

#input stimuli
#250 ms memory items | 0-250
#800 ms fixation | 250-1050
#20 ms reactivation | 1050-1070
#1080 ms fixation | 1070-2150
#100 ms impulse | 2150-2250
#400 ms fixation | 2250-2650
#250 ms probe | 2650-2900
def input_func_cued(t):
    if t > 0 and t < 0.25:
        return imagearr[memory_item_cued,:]/100
    elif t > 2.15 and t < 2.25:
        return imagearr[-1,:]/50 #impulse, twice the contrast of other items
    elif t > 2.65 and t < 2.90:
        return imagearr[probe_cued,:]/100
    else:
        return np.zeros(128*128) #blank screen

def input_func_uncued(t):
    if t > 0 and t < 0.25:
        return imagearr[memory_item_uncued,:]/100
    elif t > 2.15 and t < 2.25:
        return imagearr[-1,:]/50 #impulse, twice the contrast of other items
    elif t > 2.65 and t < 2.90:
        return imagearr[probe_uncued,:]/100
    else:
        return np.zeros(128*128) #blank screen

#reactivate memory cued ensemble with nonspecific signal
def reactivate_func(t):
    if t>1.050 and t<1.070:
        return np.ones(Nm)*0.0200
    else:
        return np.zeros(Nm)

f = 30.
w = 2. * np.pi * f
def gate_func(t):
    if t > 0.300 and t < 2.65:
        return np.ones(Ns)*np.sin(t*w)
    else:
        return np.ones(Ns)*(-1)

#create data for batch processing using the Nengo DL simulator
def get_data(inputs_cued, inputs_uncued, inputs_reactivate, inputs_gate, initialangles, n_trials, probelist, res):
    start = timer()

    #trials come in sets of 14, which we call a run (all possible orientation differences between memory and probe),
    runs = int(n_trials / 14)

    for run in range(runs):

        #run a trial with each possible orientation difference
        for cnt_in_run, anglediff in enumerate(probelist):

            i = run * 14 + cnt_in_run

            #set probe and stim
            memory_item_cued=randint(0, 179) #random memory
            probe_cued=memory_item_cued+anglediff #probe based on that
            probe_cued=norm_p(probe_cued) #normalise probe

            #random phase
            or_memory_item_cued=memory_item_cued #original
            memory_item_cued=memory_item_cued+(180*randint(0, 9))
            probe_cued=probe_cued+(180*randint(0, 9))

            #same for uncuedary item
            memory_item_uncued = memory_item_cued
            probe_uncued = probe_cued

            initialangles[i] = or_memory_item_cued

            inputs_cued[i,0:int(250/res),:]=imagearr[memory_item_cued,:]/100
            inputs_cued[i,int(2650/res):int(2900/res),:]=imagearr[probe_cued,:]/100

            inputs_uncued[i,0:int(250/res),:]=imagearr[memory_item_uncued,:]/100
            inputs_uncued[i,int(2650/res):int(2900/res),:]=imagearr[probe_uncued,:]/100

    #the impulses are the same for each trial
    inputs_cued[:,int(2150/res):int(2250/res),:]=imagearr[-1,:]/50
    inputs_uncued[:,int(2150/res):int(2250/res),:]=imagearr[-1,:]/50

    #reactivation is the same for each trial
    inputs_reactivate[:,int(1050/res):int(1070/res),:] = np.ones(Nm)*0.0200

    #gateNode input is the same for each trial
    Fs = 1175
    f = 30
    x = np.arange(Fs)
    y = np.sin(2 * np.pi * f * x / Fs)

    for i in range(Ns):
        inputs_gate[:,int(300/res):int(2650/res),i] = y

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

    global e_cued
    global U_cued
    global compressed_im_cued

    global e_uncued
    global U_uncued
    global compressed_im_uncued

    #to speed things up, load previously generated ones
    if load_gabors_svd & os.path.isfile('Stimuli/gabors_svd_cued.npz'):
        print("Start loding SVD")
        gabors_svd_cued = np.load('Stimuli/gabors_svd_cued.npz') #load stims if previously generated
        e_cued = gabors_svd_cued['e_cued']
        U_cued = gabors_svd_cued['U_cued']
        compressed_im_cued = gabors_svd_cued['compressed_im_cued']
        print("SVD cued loaded")

    else: #or generate and save

        #cued module
        #for each neuron in the sensory layer, generate a Gabor of 1/3 of the image size
        gabors_cued = Gabor().generate(Ns, (int(col/3), int(row/3)))
        #put gabors on image and make them the same shape as the stimuli
        gabors_cued = Mask((col, row)).populate(gabors_cued, flatten=True).reshape(Ns, -1)
        #normalize
        gabors_cued=gabors_cued/abs(max(np.amax(gabors_cued),abs(np.amin(gabors_cued))))
        #gabors are added to imagearr for SVD
        x_cued=np.vstack((imagearr,gabors_cued))

        #SVD
        print("SVD cued started...")
        U_cued, S_cued, V_cued = np.linalg.svd(x_cued.T)
        print("SVD cued done")

        #Use result of SVD to create encoders
        e_cued = np.dot(gabors_cued, U_cued[:,:D]) #encoders
        compressed_im_cued = np.dot(imagearr[:1800,:]/100, U_cued[:,:D]) #D-dimensional vector reps of the images
        compressed_im_cued = np.vstack((compressed_im_cued, np.dot(imagearr[-1,:]/50, U_cued[:,:D])))

        np.savez('Stimuli/gabors_svd_cued.npz', e_cued=e_cued, U_cued=U_cued, compressed_im_cued=compressed_im_cued)

    #same for uncued module
    if uncued:

        if load_gabors_svd & os.path.isfile('/Stimuli/gabors_svd_uncued.npz'):
            gabors_svd_uncued = np.load('Stimuli/gabors_svd_uncued.npz') #load stims if previously generated
            e_uncued = gabors_svd_uncued['e_uncued']
            U_uncued = gabors_svd_uncued['U_uncued']
            compressed_im_uncued = gabors_svd_uncued['compressed_im_uncued']
            print("SVD uncued loaded")
        else:
            gabors_uncued = Gabor().generate(Ns, (int(col/3), int(row/3)))#.reshape(N, -1)
            gabors_uncued = Mask((col, row)).populate(gabors_uncued, flatten=True).reshape(Ns, -1)
            gabors_uncued=gabors_uncued/abs(max(np.amax(gabors_uncued),abs(np.amin(gabors_uncued))))
            x_uncued=np.vstack((imagearr,gabors_uncued))

            print("SVD uncued started...")
            U_uncued, S_uncued, V_uncued = np.linalg.svd(x_uncued.T)
            print("SVD uncued done")
            e_uncued = np.dot(gabors_uncued, U_uncued[:,:D])
            compressed_im_uncued=np.dot(imagearr[:1800,:]/100, U_uncued[:,:D])
            compressed_im_uncued = np.vstack((compressed_im_uncued, np.dot(imagearr[-1,:]/50, U_uncued[:,:D])))

            np.savez('Stimuli/gabors_svd_uncued.npz', e_uncued=e_uncued, U_uncued=U_uncued, compressed_im_uncued=compressed_im_uncued)


nengo_gui_on = __name__ == 'builtins' #python3

def create_model(seed=None):

    global model

    #create vocabulary to show representations in gui
    if nengo_gui_on:
        vocab_angles = spa.Vocabulary(D)
        for name in [0, 3, 7, 12, 18, 25, 33, 42]:
            #vocab_angles.add('D' + str(name), np.linalg.norm(compressed_im_cued[name+90])) #take mean across phases
            v = compressed_im_cued[name+90]
            nrm = np.linalg.norm(v)
            if nrm > 0:
                v /= nrm
            vocab_angles.add('D' + str(name), v) #take mean across phases

        v = np.dot(imagearr[-1,:]/50, U_cued[:,:D])
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v /= nrm
        vocab_angles.add('Impulse', v)

    #model = nengo.Network(seed=seed)
    model = spa.SPA(seed=seed)
    with model:

        #input nodes
        if not batch_processing:
            model.inputNode_cued=nengo.Node(input_func_cued,label='input_cued')
            model.reactivate=nengo.Node(reactivate_func,label='reactivate')
            model.gateNode=nengo.Node(gate_func, label='gate')
        elif batch_processing:
            model.inputNode_cued=nengo.Node(np.zeros(128*128))
            model.reactivate=nengo.Node(np.zeros(Nm))
            model.gateNode=nengo.Node(np.zeros(Ns))

        #eye ensemble
        eye_cued = nengo.Ensemble(Ns, D, encoders=e_cued, intercepts=Uniform(0.01, 0.1),radius=1, label='eye_cued', seed=seed)
        nengo.Connection(model.inputNode_cued, eye_cued, transform=U_cued[:,:D].T)

        #sensory ensemble
        sensory_cued = nengo.Ensemble(Ns, D, encoders=e_cued, intercepts=Uniform(0.01, .1),radius=1,label='sensory_cued')

        conn_cued = nengo.Connection(eye_cued, sensory_cued, function=lambda x: x) #default: identity function

        if nengo_gui_on:
            with nengo.simulator.Simulator(network=model, progress_bar=False) as sim_cued:
                weights_cued = sim_cued.data[conn_cued].weights

            model.connections.remove(conn_cued)

        elif not(nengo_gui_on):
            with nengo_dl.Simulator(network=model, progress_bar=False, device=device) as sim_cued: #, device=device
                weights_cued = sim_cued.data[conn_cued].weights

            model.connections.remove(conn_cued)

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

        if uncued:
            # Build second simulation to fetch weights before making the large amount of connections between eye_cued and sensory_cued, place all
            # of uncued model inside this if statement to keep code close together
            if not batch_processing:
                model.inputNode_uncued=nengo.Node(input_func_uncued,label='input_first')

            elif batch_processing:
                model.inputNode_uncued=nengo.Node(np.zeros(128*128))

            eye_uncued = nengo.Ensemble(Ns, D, encoders=e_uncued, intercepts=Uniform(0.01, 0.1),radius=1, label='eye_uncued', seed=seed)
            nengo.Connection(model.inputNode_uncued, eye_uncued, transform=U_uncued[:,:D].T)

            sensory_uncued = nengo.Ensemble(Ns, D, encoders=e_uncued, intercepts=Uniform(0.01, .1),radius=1,label='sensory_uncued')

            conn_uncued = nengo.Connection(eye_uncued, sensory_uncued, function=lambda x: x) #default: identity function

            if nengo_gui_on:
                with nengo.simulator.Simulator(network=model, progress_bar=False) as sim_uncued:
                    weights_uncued = sim_uncued.data[conn_uncued].weights

                model.connections.remove(conn_uncued)

            elif not(nengo_gui_on):
                with nengo_dl.Simulator(network=model, progress_bar=False, device=device) as sim_uncued: #device=device
                    weights_uncued = sim_uncued.data[conn_uncued].weights

                model.connections.remove(conn_uncued)

            synapse_uncued = np.random.gamma(k, theta, clusters.size) + shift

            for i in range(clusters.size):

                begin=clusters[i]
                end=(begin+Ncluster)

                nengo.Connection(eye_uncued.neurons[begin:end], sensory_uncued, transform=weights_uncued[:,begin:end]*weights_trans, synapse=synapse_uncued[i])

            nengo.Connection(sensory_uncued, eye_uncued, transform = rec_trans, solver=nengo.solvers.LstsqL2(weights=True), synapse=0.200)

            noise_uncued = nengo.processes.WhiteNoise(dist=Gaussian(0,noise_sd))

            memory_uncued = nengo.Ensemble(Nm, D,neuron_type=stpLIF(), intercepts=Uniform(0.01, .1), noise=noise_uncued, radius=1, label='memory_uncued')
            nengo.Connection(sensory_uncued, memory_uncued, transform=0.1)

            nengo.Connection(memory_uncued, memory_uncued,transform=1,learning_rule_type=STP(),solver=nengo.solvers.LstsqL2(weights=True))

            comparison_uncued = nengo.Ensemble(Nd, dimensions=4, radius=math.sqrt(2), intercepts=Uniform(.01, 1), label='comparison_uncued')

            nengo.Connection(memory_uncued, comparison_uncued[2:],eval_points=compressed_im_uncued[0:-1],function=sincos.T)
            nengo.Connection(sensory_uncued, comparison_uncued[:2],eval_points=compressed_im_uncued[0:-1],function=sincos.T)

            decision_uncued = nengo.Ensemble(n_neurons=Nd,  dimensions=1, radius=45, label='decision_uncued')
            nengo.Connection(comparison_uncued, decision_uncued, eval_points=ep, scale_eval_points=False, function=arctan_func)

        ### cued ###

        synapse_cued = np.random.gamma(k, theta, clusters.size) + shift

        for i in range(clusters.size):

            begin=clusters[i]
            end=(begin+Ncluster)

            nengo.Connection(eye_cued.neurons[begin:end], sensory_cued, transform=weights_cued[:,begin:end]*weights_trans, synapse=synapse_cued[i])

        nengo.Connection(sensory_cued, eye_cued, transform = rec_trans, solver=nengo.solvers.LstsqL2(weights=True), synapse=0.200)

        #gateNode = nengo.Node(gate_func)
        gate = nengo.Ensemble(Ns, D, intercepts=nengo.dists.Uniform(0.01, .1),radius=1)
        nengo.Connection(model.gateNode, gate.neurons)#,transform=np.ones((Ns,1)))
        nengo.Connection(sensory_cued, gate, synapse=.05, transform =1.0) #play with these transform values (eg 1.2/1.2 with gatefunc3)
        nengo.Connection(gate, sensory_cued, synapse=.05, transform = 0.8) #play with these transform values or 1.0/.8 with gatefunc4

        #memory ensemble
        noise_cued= nengo.processes.WhiteNoise(dist=Gaussian(0,noise_sd))

        memory_cued = nengo.Ensemble(Nm, D,neuron_type=stpLIF(), intercepts=Uniform(0.01, .1),noise=noise_cued,radius=1,label='memory_cued')
        nengo.Connection(sensory_cued, memory_cued, transform=0.1)
        nengo.Connection(model.reactivate, memory_cued.neurons) #potential reactivation

        #recurrent STSP connection
        nengo.Connection(memory_cued, memory_cued,transform=1, learning_rule_type=STP(), solver=nengo.solvers.LstsqL2(weights=True))

        #comparison represents sin, cosine of theta of both sensory and memory ensemble
        comparison_cued = nengo.Ensemble(Nc, dimensions=4, radius=math.sqrt(2), intercepts=Uniform(.01, 1), label='comparison_cued')
        nengo.Connection(sensory_cued, comparison_cued[:2], eval_points=compressed_im_cued[0:-1], function=sincos.T)
        nengo.Connection(memory_cued, comparison_cued[2:], eval_points=compressed_im_cued[0:-1], function=sincos.T)

        #decision represents the difference in theta decoded from the sensory and memory ensembles
        decision_cued = nengo.Ensemble(n_neurons=Nd,  dimensions=1,radius=45, label='decision_cued')
        nengo.Connection(comparison_cued, decision_cued, eval_points=ep, scale_eval_points=False, function=arctan_func)

        #decode for gui
        if nengo_gui_on:
            model.sensory_decode = spa.State(D, vocab=vocab_angles, subdimensions=12, label='sensory_decode')
            for ens in model.sensory_decode.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(sensory_cued, model.sensory_decode.input,synapse=None)

            model.memory_decode = spa.State(D, vocab=vocab_angles, subdimensions=12, label='memory_decode')
            for ens in model.memory_decode.all_ensembles:
                ens.neuron_type = nengo.Direct()
            nengo.Connection(memory_cued, model.memory_decode.input,synapse=None)

        #probes
        if not(nengo_gui_on):
            if store_representations: #sim 1 trials 1-100
                model.p_mem_cued=nengo.Probe(memory_cued, synapse=0.01)
                model.p_mem_uncued=nengo.Probe(memory_uncued, synapse=0.01)

            if store_spikes_and_resources: #sim 1 trial 1
                model.p_spikes_mem_cued=nengo.Probe(memory_cued.neurons, 'spikes')
                model.p_res_cued=nengo.Probe(memory_cued.neurons, 'resources')
                model.p_cal_cued=nengo.Probe(memory_cued.neurons, 'calcium')

                model.p_spikes_mem_uncued=nengo.Probe(memory_uncued.neurons, 'spikes')
                model.p_res_uncued=nengo.Probe(memory_uncued.neurons, 'resources')
                model.p_cal_uncued=nengo.Probe(memory_uncued.neurons, 'calcium')

            if store_decisions: #sim 2
                model.p_dec_cued=nengo.Probe(decision_cued, synapse=0.01)

            if store_memory:
                model.p_mem_cued_raw=nengo.Probe(memory_cued.neurons, 'output', synapse=0.01)
                model.p_mem_uncued_raw=nengo.Probe(memory_uncued.neurons, 'output', synapse=0.01)

    return model

#PLOTTING CODE
from nengo.utils.matplotlib import rasterplot
from matplotlib import style
from plotnine import *
theme = theme_classic()
plt.style.use('default')

def plot_sim_1(sp_c,sp_u,res_c,res_u,cal_c,cal_u, mem_cued, mem_uncued):

    figure_name = 'Sim_1/full_model/exp1_content_specific_reactivation_model/exp1_figure'

    #FIGURE 31
    with plt.rc_context():
        plt.rcParams.update(theme.rcParams)

        fig, axes, = plt.subplots(2,2,squeeze=True)
        theme.setup_figure(fig)
        t = sim.trange()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

        #spikes, calcium, resources Cued
        ax1=axes[0,0]
        ax1.set_title("Cued Module")
        ax1.set_ylabel('# cell', color='black')
        ax1.set_yticks(np.arange(0,Nm,500))
        ax1.tick_params('y')#, colors='black')
        rasterplot(sim.trange(), sp_c,ax1,colors=['black']*sp_c.shape[0])
        ax1.set_xticklabels([])
        ax1.set_xticks([])
        ax1.set_xlim(0,3)
        ax2 = ax1.twinx()
        ax2.plot(t, res_c, "#00bfc4",linewidth=2)
        ax2.plot(t, cal_c, "#e38900",linewidth=2)
        ax2.set_yticklabels([])
        ax2.set_yticks([])
        ax2.set_ylim(0,1.1)

        #spikes, calcium, resources Uncued
        ax3=axes[0,1]
        ax3.set_title("Uncued Module")
        rasterplot(sim.trange(), sp_u,ax3,colors=['black']*sp_u.shape[0])
        ax3.set_xticklabels([])
        ax3.set_xticks([])
        ax3.set_yticklabels([])
        ax3.set_yticks([])
        ax3.set_xlim(0,3)
        ax4 = ax3.twinx()
        ax4.plot(t, res_u, "#00bfc4",linewidth=2)
        ax4.plot(t, cal_u, "#e38900",linewidth=2)
        ax4.set_ylabel('synaptic variables', color="black",size=11)
        ax4.tick_params('y', labelcolor='#333333',labelsize=9,color='#333333')
        ax4.set_ylim(0,1.1)

        #representations cued
        plot_mc=axes[1,0]
        plot_mc.plot(sim.trange(),(mem_cued));

        plot_mc.set_ylabel("Cosine similarity")
        plot_mc.set_xticks(np.arange(0.0,3.45,0.5))
        plot_mc.set_xticklabels(np.arange(0,3500,500).tolist())
        plot_mc.set_xlabel('time (ms)')
        plot_mc.set_xlim(0,3)
        colors=["#00c094","#00bfc4","#00b6eb","#06a4ff","#a58aff","#df70f8","#fb61d7","#ff66a8", "#c49a00"]
        for i,j in enumerate(plot_mc.lines):
            j.set_color(colors[i])

        #representations uncued
        plot_mu=axes[1,1]

        plot_mu.plot(sim.trange(),(mem_uncued));
        plot_mu.set_xticks(np.arange(0.0,3.45,0.5))
        plot_mu.set_xticklabels(np.arange(0,3500,500).tolist())
        plot_mu.set_xlabel('time (ms)')
        plot_mu.set_yticks([])
        plot_mu.set_yticklabels([])
        plot_mu.set_xlim(0,3)
        for i,j in enumerate(plot_mu.lines):
            j.set_color(colors[i])
        plot_mu.legend(["0°","3°","7°","12°","18°","25°","33°","42°", "Impulse"], title="Stimulus", bbox_to_anchor=(1.02, -0.25, .30, 0.8), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)


        fig.set_size_inches(11, 5)
        theme.apply(plt.gcf().axes[0])
        theme.apply(plt.gcf().axes[1])
        theme.apply(plt.gcf().axes[2])
        theme.apply(plt.gcf().axes[3])
        plt.savefig(figure_name + '_3.eps', format='eps', dpi=1000)
        plt.show()


    #FIGURE 32
    with plt.rc_context():
        plt.rcParams.update(theme.rcParams)

        fig, axes, = plt.subplots(1,2,squeeze=True)
        theme.setup_figure(fig)
        t = sim.trange()
        plt.subplots_adjust(wspace=0.1, hspace=0.05)

        plot_mc=axes[0]
        plot_mc.set_title("Cued Module")
        plot_mc.plot(sim.trange(),(mem_cued));
        plot_mc.set_ylabel("Cosine similarity")
        plot_mc.set_xticks(np.arange(2.15,2.65,0.05))
        plot_mc.set_xticklabels(np.arange(0,500,50).tolist())
        plot_mc.set_xlabel('time after onset impulse (ms)')
        plot_mc.set_xlim(2.15,2.4)
        plot_mc.set_ylim(0,0.9)
        colors=["#00c094","#00bfc4","#00b6eb","#06a4ff","#a58aff","#df70f8","#fb61d7","#ff66a8", "#c49a00"]
        for i,j in enumerate(plot_mc.lines):
            j.set_color(colors[i])

        plot_mu=axes[1]
        plot_mu.set_title("Uncued Module")
        plot_mu.plot(sim.trange(),(mem_uncued));
        plot_mu.set_xticks(np.arange(2.15,2.65,0.05))
        plot_mu.set_xticklabels(np.arange(0,500,50).tolist())
        plot_mu.set_xlabel('time after onset impulse (ms)')
        plot_mu.set_yticks([])
        plot_mu.set_yticklabels([])
        plot_mu.set_xlim(2.15,2.40)
        plot_mu.set_ylim(0,0.9)
        for i,j in enumerate(plot_mu.lines):
            j.set_color(colors[i])
        plot_mu.legend(["0°","3°","7°","12°","18°","25°","33°","42°", "Impulse"], title="Stimulus", bbox_to_anchor=(0.85, 0.25, .55, 0.8), loc=3,
               ncol=1, mode="expand", borderaxespad=0.)

        fig.set_size_inches(6, 4)

        theme.apply(plt.gcf().axes[0])
        theme.apply(plt.gcf().axes[1])
        plt.savefig(figure_name + '_4.eps', format='eps', dpi=1000)
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
            out[i]=np.dot(a[i], b)/(np.linalg.norm(a[i])*np.linalg.norm(b))
    return out

if nengo_gui_on:
    generate_gabors() #generate gabors
    
    with nengo.Network() as model:
        create_model(seed=0) #build model

    memory_item_cued = 0 + 90
    probe_cued = 42 + 90
    memory_item_uncued = 0 + 90
    probe_uncued = 42 + 90

else: #no gui

    #path
    cur_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+'/data/' #store output in data subfolder

    #simulation 1: recreate fig 3 & 4 from Pals(2019), 100 trials for both cued and uncued with 0 and 42 degree memory items and probes
    if sim_to_run == 1:

        print('Running simulation 1')
        print('')

        load_gabors_svd = True #no need to randomize this

        ntrials = 10
        store_representations = True
        store_decisions = False
        uncued = True

        #store results
        templates= np.array([90,93,97,102,108,115,123,132])
        mem_cued = np.zeros((3000,len(templates)+1)) #keep cosine sim for 9 items (templates + impulse)
        mem_uncued = np.zeros((3000,len(templates)+1))

        #first, run 100 trials to get average cosine sim
        for run in range(ntrials):

            print('Run ' + str(run+1))

            #stimuli
            phase = 180*randint(0, 9)
            memory_item_cued = 0 + 90 + phase
            probe_cued = 42 + 90 + phase
            memory_item_uncued = memory_item_cued
            probe_uncued = probe_cued

            #create new gabor filters every 10 trials
            if run % 10 == 0:
                generate_gabors()

            model = create_model(seed=run)
            sim = nengo_dl.Simulator(network=model, seed=run, progress_bar=False, device=device)

            #run simulation
            sim.run(3)

            #reset simulator, clean probes thoroughly
            #print(sim.data[model.p_mem_cued].shape)
            #calc cosine sim with templates
            temp_phase = list(templates + phase) + [1800]
            for cnt, templ in enumerate(temp_phase):
                mem_cued[:,cnt] += cosine_sim(sim.data[model.p_mem_cued][:,:,],compressed_im_cued[templ,:])
                mem_uncued[:,cnt] += cosine_sim(sim.data[model.p_mem_uncued][:,:,],compressed_im_uncued[templ,:])

            sim.reset()
            sim.close()

        #average
        mem_cued /= ntrials
        mem_uncued /= ntrials

        #second, run 1 trial to get calcium and spikes
        store_spikes_and_resources = True
        store_representations = False
        create_model(seed=0) #recreate model to change probes
        sim = nengo_dl.Simulator(network=model, seed=0, progress_bar=False, device=device)

        print('Run ' + str(ntrials+1))
        sim.run(3)

        #store spikes and calcium
        sp_c = sim.data[model.p_spikes_mem_cued]
        res_c=np.mean(sim.data[model.p_res_cued][:,:,],1) #take mean over neurons
        cal_c=np.mean(sim.data[model.p_cal_cued][:,:,],1) #take mean over neurons

        sp_u=sim.data[model.p_spikes_mem_uncued]
        res_u=np.mean(sim.data[model.p_res_uncued][:,:,],1)
        cal_u=np.mean(sim.data[model.p_cal_uncued][:,:,],1)

        sim.close()

        #plot
        plot_sim_1(sp_c,sp_u,res_c,res_u,cal_c,cal_u, mem_cued, mem_uncued)


    #simulation 2: collect data for fig 5 & 6. 1344 trials for 30 subjects
    if sim_to_run == 2:

        load_gabors_svd = False #set to false for real simulation
        uncued = True

        data_path = "/Users/s3344282/"

        #set to 1 for (default) simulator dt of 0.001, set to 2 for simulator dt of 0.002, ....
        res = 2 #resolution

        n_subj = 1 #30
        trials_per_subj = 280 #1344
        store_representations = False
        store_decisions = True
        store_memory = True

        split = 4 #we split the neuron data into this many separate files to decrease the cpu memory use

        #division in the form of n_trials/minibatch_size/split should yield a whole integer, otherwise indexing errors will occur and/or the number of trials performed will not be as specified.

        minibatch_size=14 #multiples of 14
        n_steps=int(3000/res) #3000 ms for one trial

        #orientation differences between probe and memory item for each run
        probelist=[-42, -33, -25, -18, -12, -7, -3, 3, 7, 12, 18, 25, 33, 42]

        #arrays for storing input data
        input_data_cued = np.zeros((minibatch_size, n_steps, 128*128))
        input_data_uncued = np.zeros((minibatch_size, n_steps, 128*128))
        input_data_reactivate = np.zeros((minibatch_size, n_steps, Nm))
        input_data_gate = np.ones((minibatch_size, n_steps, Ns)) * (-1)
        data_initialangles = np.zeros(minibatch_size)

        if store_memory:
            initialangles = np.zeros(int(trials_per_subj/split))
            neuro_mem_cued = np.zeros((int(trials_per_subj/split), n_steps, Nm))
            neuro_mem_uncued = np.zeros((int(trials_per_subj/split), n_steps, Nm))

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

                get_data(input_data_cued, input_data_uncued, input_data_reactivate, input_data_gate, data_initialangles, minibatch_size, probelist,res)

                #run simulation
                sim.run_steps(
                    n_steps,
                    data = {
                        model.inputNode_cued: input_data_cued,
                        model.inputNode_uncued: input_data_uncued,
                        model.reactivate: input_data_reactivate,
                        model.gateNode: input_data_gate
                    },
                )

                #store performance data (sim 2)
                if store_decisions:
                    for in_batch_trial in range(minibatch_size):

                        anglediff = probelist[in_batch_trial%14] #modulo; 0%14 returns 0, 1%14 returns 1, ..., 14%14 returns 0

                        trial = batch * minibatch_size + in_batch_trial + 1

                        # np.savetxt(data_path+"Performance/full_model/exp1_content_specific_reactivation_model/"+sim_no+"_Diff_Theta_%i_subj_%i_trial_%i.csv" % (anglediff, subj+1, trial),
                        #     sim.data[model.p_dec_cued][in_batch_trial,int((2650)/res):int((2950)/res),:], delimiter=",") 
                        #     #this saves the output of the decision population during the timeframe at which the probe is presented (+50ms)

                        np.savetxt(data_path+"Performance/attempts/exp1_content_specific_reactivation_model/"+sim_no+"_Diff_Theta_%i_subj_%i_trial_%i.csv" % (anglediff, subj+1, trial),
                            sim.data[model.p_dec_cued][in_batch_trial,int((2650)/res):int((2950)/res),:], delimiter=",") 
                            #this saves the output of the decision population during the timeframe at which the probe is presented (+50ms)

                #store neuron data (sim 3)
                if store_memory:
                    trial_index_start=int((batch*minibatch_size)%(trials_per_subj/split))
                    trial_index_end=trial_index_start+minibatch_size

                    initialangles[trial_index_start:trial_index_end]=data_initialangles

                    neuro_mem_cued[trial_index_start:trial_index_end, :, :] = sim.data[model.p_mem_cued_raw]
                    neuro_mem_uncued[trial_index_start:trial_index_end, :, :] = sim.data[model.p_mem_uncued_raw]

                    if (batch+1)%(n_batches/split)==0:

                        # np.save(data_path+"Decoding/full_model/exp1_content_specific_reactivation_model/subj_%i_mem_neuron_data_cued_%i.npy" % (subj+1, split_index), neuro_mem_cued)
                        # np.save(data_path+"Decoding/full_model/exp1_content_specific_reactivation_model/subj_%i_mem_neuron_data_uncued_%i.npy" % (subj+1, split_index), neuro_mem_uncued)
                        # np.save(data_path+"Decoding/full_model/exp1_content_specific_reactivation_model/subj_%i_initial_angles_%i.npy" % (subj+1, split_index), initialangles)

                        np.save(data_path+"Decoding/attempts/exp1_content_specific_reactivation_model/subj_%i_mem_neuron_data_cued_%i.npy" % (subj+1, split_index), neuro_mem_cued)
                        np.save(data_path+"Decoding/attempts/exp1_content_specific_reactivation_model/subj_%i_mem_neuron_data_uncued_%i.npy" % (subj+1, split_index), neuro_mem_uncued)
                        np.save(data_path+"Decoding/attempts/exp1_content_specific_reactivation_model/subj_%i_initial_angles_%i.npy" % (subj+1, split_index), initialangles)

                        initialangles[:]=0
                        neuro_mem_cued[:,:,:]=0
                        neuro_mem_uncued[:,:,:]=0

                        split_index+=1

                        print("Neuron dataset %i/%i stored" % (split_index, split))

                #reset simulator
                sim.reset()

            sim.close()

            