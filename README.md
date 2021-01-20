# SimpleAutoGrad
A simple reverse-mode automatic differentiation library

This is my first attempt at understanding and implementing automatic differentiation. This is not optimized, and should not be used for any production code. 

TODO:
1) Look more into the unbroadcasting, I think I might be missing some stuff
2) Better vectorization?
3) Make the grad() method of each node return a function for the VJP, instead of computing it on the spot. This is probably more efficient?
4) Implement more examples (ResNet, a meaningful NN exampe, simple GAN, etc...)
5) Integrate with Neuro and demonstrate functionality
