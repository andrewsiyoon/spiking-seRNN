# seRSNNs (Spatially-embedded recurrent spiking neural networks)

Spatially-embedded recurrent spiking neural networks (seRSNNs) are newly-developed in order to subject recurrent spiking neural networks (RSNNs) to biologically-feasible constraints. Neurons are assigned a physical location in Euclidean space, and the physical distance and communicability of connections are optimized for efficient topological organization. Sample implementations are provided in Jupyter notebooks; for instructive notebooks training seRSNNs on various datasets, read seRSNN_DVS_Gesture_demo.ipynb and seRSNN_SHD_demo.ipynb.

The folder basemodels contains implementations of standard spiking neural networks and recurrent spiking neural networks that formed the basis of later seRSNNs.

The folder finalscripts contains Python scripts for L1-regularized and spatially-regularized networks.
