import numpy as np

#assum zero-mean one-std, input
def do_random_affine(xyz,
    scale  = (0.8,1.5),
    shift  = (-0.1,0.1),
    degree = (-15,15),
    p=0.5
):
	if np.random.rand() < p:
		if scale is not None:
			scale = np.random.uniform(*scale)
			xyz = scale*xyz

		if shift is not None:
			shift = np.random.uniform(*shift)
			xyz = xyz + shift

		if degree is not None:
			degree = np.random.uniform(*degree)
			radian = degree/180*np.pi
			c = np.cos(radian)
			s = np.sin(radian)
			rotate = np.array([
				[c,-s],
				[s, c],
			]).T
			xyz[...,:2] = xyz[...,:2] @rotate

	return xyz