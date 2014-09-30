Lattice Regression model.

The big idea is pretty straightforward.  We want to model some function.  We throw down a grid (i.e. a lattice)
over the space of possible inputs and learn something like the values of the function at those points from
some training data.  We store those values and use them to predict the value of the function for new inputs
via an interpolation scheme.  For more info, see, e.g.:

http://www.mayagupta.org/publications/GarciaAroraGupta_lattice_regression_IEEETransImageProcessing2012.pdf

Uses batch gradient descent and a regularizer that penalizes differences between adjacent lattice points.
Assumes features (i.e. inputs) are normalized to fall in [0,1].  The response variable (i.e. output) need not
be normalized.
