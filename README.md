# GaussianDistributionGenerator
A Gaussian Distribution Generator written in C++ with OpenMP. This implements either the Marsaglia or the Box-Muller version.

When running with only one thread, Marsaglia is by far faster. After this though, the parallel speedup remains close to 1 for Marsaglia while Box-Muller demonstrates much higher levels of possible paralleism. 
