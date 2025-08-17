-> Hough transform with polar co-ordinates has been implemented
-> Hough transform with Euclidean co-ordinates has been implemented

-> Preprocessing steps :
	1. Gaussian smoothing has been applied to remove minor edges.
	2. Laplacian of the result is calculated this gives us the second-order derivative.
	3. Implemented zero-crossing detector based on a neighborhood on the result of Laplacian.
	4. Two methods have been used for plotting accumulator array :
		1. Polar Co-ordinate Method.
		2. Cartesian Method.