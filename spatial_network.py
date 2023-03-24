#Imports -----
print('Importing packages...')

import os
from os import mkdir, makedirs
from datetime import datetime
from pathlib import Path

import torch, torch.nn as nn
from torch.utils.data import DataLoader

import snntorch as snn
from snntorch import surrogate
import snntorch.functional as SF

import numpy as np

import scipy
from scipy.stats import pearsonr

import tonic
import tonic.transforms as transforms
from tonic import DiskCachedDataset

import random

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import seaborn as sns
import pandas as pd


#Input variable values from slurm command -----
import sys
regu_strength = float(sys.argv[1])
index = str(sys.argv[1])

print(f"Regularization strength: {regu_strength}\n")


#Random seeds -----
print('Setting random seeds...')
np.random.seed(211)
random.seed(211)
torch.manual_seed(211)


#DVS Gesture -----
print('Downloading and extracting DVS Gesture dataset...')

#Define variables (batch size, sensor size)
batch_size = 64
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
sensor_size = tonic.datasets.DVSGesture.sensor_size

#Define transformations
frame_transform = transforms.Compose([transforms.ToFrame(sensor_size=sensor_size, n_time_bins = 20), transforms.DropEvent(p = 0.001)])

#Define training and test sets
DVS_train = tonic.datasets.DVSGesture(save_to='/imaging/shared/users/ja02/CBUActors/MPhil/Andrew/wrapper_seRSNN/DVSGesture', transform=frame_transform, train=True)
DVS_test = tonic.datasets.DVSGesture(save_to='/imaging/shared/users/ja02/CBUActors/MPhil/Andrew/wrapper_seRSNN/DVSGesture', transform=frame_transform, train=False)

#Create dataloaders
trainloader = DataLoader(DVS_train, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle = True, drop_last = True)
testloader = DataLoader(DVS_test, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle = True, drop_last = True)


#Initialize membrane time constant distribution -----
print('Initializing membrane time constant distribution...')

#Membrane parameters
tau_mem = 20e-3
dist_shape = 3
time_step = 0.5e-3

#Clipping function
def clip_tc(x):
    clipped_tc = x.clamp_(0.7165, 0.995)
    return clipped_tc

#Initialize membrane time constant distribution
def init_tc():
    dist_gamma = np.random.gamma(dist_shape, tau_mem / 3, 100)
    dist_beta = torch.from_numpy(np.exp(-time_step / dist_gamma))
    clipped_beta = clip_tc(dist_beta)
    return clipped_beta


#Model architecture -----
print('Constructing model architecture...')

#Size parameters
num_inputs = 128*128*2
num_hidden = 100
num_outputs = 11

#Network parameters
het_tau = init_tc()
hom_tau = 0.9753

#Optimization mechanism
spike_grad = surrogate.fast_sigmoid(slope = 100)

#Model definition
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.RLeaky(beta = het_tau, linear_features = num_hidden, learn_beta = True, spike_grad = spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta = hom_tau, spike_grad = spike_grad)

    def forward(self, x):

        #Initialize parameters
        spk1, mem1 = self.lif1.init_rleaky()
        mem2 = self.lif2.init_leaky()

        #Record output layer
        spk_out_rec = []
        mem_out_rec = []
        
        #Forward loop
        for step in range(data.size(0)):
            batched_data = data[step].view(batch_size, num_inputs)
            cur1 = self.fc1(batched_data)
            spk1, mem1 = self.lif1(cur1, spk1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_out_rec.append(spk2)
            mem_out_rec.append(mem2)

        #Convert output lists to tensors
        spk_out_rec = torch.stack(spk_out_rec)
        mem_out_rec = torch.stack(mem_out_rec)
        
        return spk_out_rec, mem_out_rec

net = Net()


#Extract membrane time constants (pre-training) -----
print('Extracting initialized membrane time constants before training...')
tc_hist = []
pretrain_tau = (-time_step / np.log(het_tau)) / 1e-3
tc_hist.append(pretrain_tau.numpy())


#Optimizer and loss function -----
print('Defining optimizer parameters and loss function...')
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3, betas = (0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate = 0.8, incorrect_rate = 0.2)


#Distance matrix -----
print('Creating Euclidean distance matrix...')

network_structure = [5, 5, 4]
distance_metric = 'euclidean'
distance_power = 1

nx = np.arange(network_structure[0])
ny = np.arange(network_structure[1])
nz = np.arange(network_structure[2])

[x,y,z] = np.meshgrid(nx,ny,nz)
coordinates = [x.ravel(),y.ravel(),z.ravel()]

euclidean_vector = scipy.spatial.distance.pdist(np.transpose(coordinates), metric=distance_metric)
euclidean = scipy.spatial.distance.squareform(euclidean_vector**distance_power)
distance_matrix = euclidean.astype('float64')

distance_matrix = torch.from_numpy(distance_matrix)


#Diagnostic tests -----
print('Running diagnostic tests...')

#Test for spatial regularization
def test_euclidean(x, y):
    x = torch.abs(x)
    x_array = x.detach().numpy()
    flat_x_array = x_array.flatten()
    y = torch.abs(y)
    y_array = y.detach().numpy()
    flat_y_array = y_array.flatten()
    correlation = pearsonr(flat_x_array, flat_y_array)[0]
    return correlation

print(f"Initial correlation between distance and weight matrices: {test_euclidean(distance_matrix, net.lif1.recurrent.weight)}")


#Training paradigm -----
print('Beginning training...\n')

#Training parameters
num_epochs = 50

#Regularization parameters
regu_strength = float(sys.argv[1])

#Initialize variables of interest
train_loss_hist = []
train_acc_hist = []
feed_tot_hist = []
rec_tot_hist = []
corr_hist = []
test_acc_hist = []
test_loss_hist = []

#Initialize hidden layer attributes
spk_hist = []
mem_hist = []

#Pre-training extractions
feed_tot_hist.append(torch.sum(torch.abs(net.fc1.weight.detach())))
rec_tot_hist.append(torch.sum(torch.abs(net.lif1.recurrent.weight.detach())))
corr_hist.append(test_euclidean(distance_matrix, net.lif1.recurrent.weight))

#Training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(iter(trainloader)):

        #Set model to training mode
        net.train()
        spk_outputs, mem_outputs = net(data)

        #Spatial + L1 regularization
        abs_weight_matrix = torch.abs(net.lif1.recurrent.weight)
        spatial_L1_loss = regu_strength * torch.sum(torch.mul(abs_weight_matrix, distance_matrix))

        #Calculate loss
        loss_val = loss_fn(spk_outputs, targets) + spatial_L1_loss

        #Gradient calculation and weight updates
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        clip_tc(net.lif1.beta.detach())

        #Store loss history
        train_loss_hist.append(loss_val.item())

    #Evaluations (every epoch)
    net.eval()

    #Report training loss
    print(f"Epoch {epoch} \nTraining Loss: {loss_val.item():.2f}")

    #Training accuracy
    acc = SF.accuracy_rate(spk_outputs, targets)
    train_acc_hist.append(acc)
    print(f"Training accuracy: {acc * 100:.2f}%")

    #Sum of feedforward weights
    feed_tot = torch.sum(torch.abs(net.fc1.weight.detach()))
    feed_tot_hist.append(feed_tot)

    #Sum of recurrent weights
    rec_tot = torch.sum(torch.abs(net.lif1.recurrent.weight.detach()))
    rec_tot_hist.append(rec_tot)

    #Correlation of distance and weight matrices
    corr_matrix = test_euclidean(distance_matrix, net.lif1.recurrent.weight.detach())
    corr_hist.append(corr_matrix)

    #Save membrane time constant matrix
    converted_tc = (-time_step / np.log(net.lif1.beta.detach())) / 1e-3
    tc_hist.append(converted_tc.numpy())

    #Validation accuracy
    with torch.no_grad():
        net.eval()
        total = 0
        correct = 0

        for data, targets in testloader:
            test_spk, test_mem = net(data)

            _, predicted = test_spk.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            test_loss = loss_fn(test_spk, targets) + spatial_L1_loss

        test_acc_hist.append(correct / total)
        test_loss_hist.append(test_loss.item())

    print(f'Test loss: {test_loss.item():.2f}')
    print(f'Test set accuracy: {100 * correct / total:.2f}%\n')

print('Finished training...')


#Set up directories -----
run_id = datetime.now()
run_path = "/imaging/shared/users/ja02/CBUActors/MPhil/Andrew/wrapper_seRSNN/seRSNN_run_" + run_id.strftime("%Y_%m_%d")
Path(run_path).mkdir(parents=True, exist_ok=True)


#Save model -----
model_dir = run_path + "/model_architecture"
Path(model_dir).mkdir(parents=True, exist_ok=True)
model_path = model_dir + "/model_architecture_" + index + "_.pt"

model_architecture = {'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'loss': loss_val}
torch.save(model_architecture, model_path)

print('Saved model architecture...')


#Extract data as .csv files -----
trainingloss = np.array(train_loss_hist)
tl_dir = run_path + "/training_loss"
Path(tl_dir).mkdir(parents=True, exist_ok=True)
tl_path = tl_dir + "/training_loss_" + index + "_.csv"
pd.DataFrame(trainingloss).to_csv(tl_path, header=False, index=False)

trainingaccuracy = np.array(train_acc_hist)
ta_dir = run_path + "/training_accuracy"
Path(ta_dir).mkdir(parents=True, exist_ok=True)
ta_path = ta_dir + "/training_accuracy_" + index + "_.csv"
pd.DataFrame(trainingaccuracy).to_csv(ta_path, header=False, index=False)

forwardweights = np.array(feed_tot_hist)
fw_dir = run_path + "/forward_weights"
Path(fw_dir).mkdir(parents=True, exist_ok=True)
fw_path = fw_dir + "/forward_weights_" + index + "_.csv"
pd.DataFrame(forwardweights).to_csv(fw_path, header=False, index=False)

recurrentweights = np.array(rec_tot_hist)
rw_dir = run_path + "/recurrent_weights"
Path(rw_dir).mkdir(parents=True, exist_ok=True)
rw_path = rw_dir + "/recurrent_weights_" + index + "_.csv"
pd.DataFrame(recurrentweights).to_csv(rw_path, header=False, index=False)

correlations = np.array(corr_hist)
c_dir = run_path + "/correlations"
Path(c_dir).mkdir(parents=True, exist_ok=True)
c_path = c_dir + "/correlations_" + index + "_.csv"
pd.DataFrame(correlations).to_csv(c_path, header=False, index=False)

testaccuracy = np.array(test_acc_hist)
test_dir = run_path + "/test_accuracy"
Path(test_dir).mkdir(parents=True, exist_ok=True)
test_path = test_dir + "/test_accuracy_" + index + "_.csv"
pd.DataFrame(testaccuracy).to_csv(test_path, header=False, index=False)

testloss = np.array(test_loss_hist)
tloss_dir = run_path + "/test_loss"
Path(tloss_dir).mkdir(parents=True, exist_ok=True)
tloss_path = tloss_dir + "/test_loss_" + index + "_.csv"
pd.DataFrame(testloss).to_csv(tloss_path, header=False, index=False)

timeconstants = np.array(tc_hist)
tc_dir = run_path + "/time_constants"
Path(tc_dir).mkdir(parents=True, exist_ok=True)
tc_path = tc_dir + "/time_constants_" + index + "_.csv"
pd.DataFrame(timeconstants).to_csv(tc_path, header=False, index=False)

print('Extracted model data...')
print('Finished.')

