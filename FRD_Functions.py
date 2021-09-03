from scipy import fft
from scipy.interpolate import RectBivariateSpline
import numpy as np

'''
This script contains functions for generating FRDs

TODO: 
    -speed up interpolation
'''



def circle_values(center, R, numPoints, image):
    '''
    calculate the interpolated circle values
    
    Inputs: 
        center - a list of numpy array of the center of the image [y-value, x-value]
        R  - the radius of the circle
        num_Points - the number of evenly spaced points on the circle to be interpolated
        image - a numpy array of the whole image
    Outputs:
        interpolated - a numpy array of the interpolated pixel values
    
    '''
	angles = np.linspace(0,2*np.pi,numPoints)
	x_values = R*np.cos(angles)+center[1]
	y_values = R*np.sin(angles)+center[0]
	M,N = image.shape
	spline = RectBivariateSpline(np.arange(M), np.arange(N), image)
	interpolated = spline(y_values, x_values, grid = False)
	return interpolated


def FRDs(image, numPoints, maxR, center):
    '''
    Generate an FRD of the image
    
    Inputs:
        image - a numpy array the input image for the FRD
        numPoints - the number of points on the smallest ring
        maxR - the radius of the biggest ring
        center - a list of numpy array of the center of the image [y-value, x-value]
    Outputs:
        FRD - a numpy array of the FRD
    '''
	FRD = np.array([])
	for i in range(maxR):
		ringValues = circle_values(center, i+1, numPoints*(i+1), image)
		fourierValues = abs(fft(ringValues))[:numPoints*(i+1)+1]
		FRD = np.append(FRD, fourierValues)
	return FRD
