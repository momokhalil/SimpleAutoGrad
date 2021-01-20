# SimpleAutoGrad
A simple reverse-mode automatic differentiation library

This is my first attempt at understanding and implementing automatic differentiation. This is not optimized, and should not be used for any production code. This is also not heavily tested yet, so there may be some bugs waiting around :)

Basic features:
1) High order differentiation of scalar functions (Example in high_order.py)
2) Matrix/vector function differentiation by simply defining the important matrix operations (matmul, transpose, etc...) as nodes that return the correct VJP (evaluated, not a function, more on this in the TODO section). Example in simple_NN.py

TODO:
1) Look more into the unbroadcasting, I think I might be missing some stuff
2) Better vectorization?
3) Make the grad() method of each node return a function for the VJP, instead of computing it on the spot. This is probably more efficient?
4) Implement more examples (ResNet, a meaningful NN exampe, simple GAN, etc...)
5) Integrate with Neuro and demonstrate functionality
6) This needs ALOT more testing :)

Inspirations:
1) https://github.com/bgavran/autodiff 
2) https://github.com/mattjj/autodidact
3) https://github.com/hips/autograd
4) https://sidsite.com/posts/autodiff/
5) https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec10.pdf
