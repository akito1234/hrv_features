# -*- coding: utf-8 -*-
"""
pyHRV - Nonlinear Parameters
----------------------------
心拍変動の非線形解析

"""

# Third party libraries
import nolds
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# BioSPPy imports
import biosppy

# Local imports/pyHRV toolbox imports
#import pyhrv
import src.features.featuresHRV.featuresDomain.utils


# TODO add devplot mode
def poincare(nni=None,
			 show=False,
			 figsize=None,
			 ellipse=True,
			 vectors=True,
			 legend=True,
			 marker='o'):
	"""Creates Poincaré plot from a series of NN intervals or R-peak locations and derives the Poincaré related
	parameters SD1, SD2, SD1/SD2 ratio, and area of the Poincaré ellipse.

	References: [Tayel2015][Brennan2001]

	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s]
	rpeaks : array
		R-peak times in [ms] or [s]
	show : bool, optional
		If true, shows Poincaré plot (default: True)
	show : bool, optional
		If true, shows generated plot
	figsize : array, optional
		Matplotlib figure size (width, height) (default: (6, 6))
	ellipse : bool, optional
		If true, shows fitted ellipse in plot (default: True)
	vectors : bool, optional
		If true, shows SD1 and SD2 vectors in plot (default: True)
	legend : bool, optional
		If True, adds legend to the Poincaré plot (default: True)
	marker : character, optional
		NNI marker in plot (default: 'o')

	Returns (biosppy.utils.ReturnTuple Object)
	------------------------------------------
	[key : format]
		Description.
	poincare_plot : matplotlib figure object
		Poincaré plot figure
	sd1 : float
		Standard deviation (SD1) of the major axis
	sd2 : float, key: 'sd2'
		Standard deviation (SD2) of the minor axis
	sd_ratio: float
		Ratio between SD2 and SD1 (SD2/SD1)
	ellipse_area : float
		Area of the fitted ellipse

	"""
	# Prepare Poincaré data
	x1 = np.asarray(nni[:-1])
	x2 = np.asarray(nni[1:])

	# SD1 & SD2 Computation
	sd1 = np.std(np.subtract(x1, x2) / np.sqrt(2))
	sd2 = np.std(np.add(x1, x2) / np.sqrt(2))

	# Area of ellipse
	area = np.pi * sd1 * sd2
	# Output
	args = (sd1, sd2, sd2/sd1, area)
	names = ('sd1', 'sd2', 'sd_ratio', 'ellipse_area')
	return biosppy.utils.ReturnTuple(args, names)


def sample_entropy(nni=None, dim=2, tolerance=None):
	"""Computes the sample entropy (sampen) of the NNI series.

	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].
	dim : int, optional
		Entropy embedding dimension (default: 2).
	tolerance : int, float, optional
		Tolerance distance for which the vectors to be considered equal (default: std(NNI) * 0.2).

	Returns (biosppy.utils.ReturnTuple Object)
	------------------------------------------
	[key : format]
		Description.
	sample_entropy : float
		Sample entropy of the NNI series.

	Raises
	------
	TypeError
		If 'tolerance' is no numeric value.

	"""
	if tolerance is None:
		tolerance = np.std(nni, ddof=-1) * 0.2
	else:
		try:
			tolerance = float(tolerance)
		except:
			raise TypeError('Tolerance level cannot be converted to float.'
							'Please verify that tolerance is a numeric (int or float).')

	# Compute Sample Entropy
	sampen = float(nolds.sampen(nni, dim, tolerance))

	# Output
	args = (sampen, )
	names = ('sampen', )
	return biosppy.utils.ReturnTuple(args, names)

def dfa(nni=None, short=(4, 16), long=(17, 64), show=False, figsize=None, legend=True):
	"""Conducts Detrended Fluctuation Analysis for short and long-term fluctuation of an NNI series.

	References: [Joshua2008][Kuusela2014][Fred2017]
	Docs:		https://pyhrv.readthedocs.io/en/latest/_pages/api/nonlinear.html#sample-entropy-sample-entropy

	Parameters
	----------
	nn : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].
	short : array, 2 elements
		Interval limits of the short term fluctuations (default: None: [4, 16]).
	long : array, 2 elements
		Interval limits of the long term fluctuations (default: None: [17, 64]).
	show : bool
		If True, shows DFA plot (default: True)
	legend : bool
		If True, adds legend with alpha1 and alpha2 values to the DFA plot (default: True)

	Returns (biosppy.utils.ReturnTuple Object)
	------------------------------------------
	[key : format]
		Description.
	dfa_short : float
		Alpha value of the short term fluctuations
	dfa_long : float
		Alpha value of the long term fluctuations
	dfa_plot : matplotlib plot figure
		Matplotlib plot figure of the DFA

	"""

	# Create arrays
	short = range(short[0], short[1] + 1)
	long = range(long[0], long[1] + 1)
	alpha1, dfa_short = nolds.dfa(nni, short, debug_data=True, overlap=False)
	alpha2, dfa_long = nolds.dfa(nni, long, debug_data=True, overlap=False)
	
    # Output
	args = (alpha1, alpha2, short, long)
	return biosppy.utils.ReturnTuple(args, ('dfa_alpha1', 'dfa_alpha2', 'dfa_alpha1_beats', 'dfa_alpha2_beats'))


def nonlinear(nni=None,
			  rpeaks=None,
			  signal=None,
			  sampling_rate=1000.,
			  show=False,
			  kwargs_poincare=None,
			  kwargs_sampen=None,
			  kwargs_dfa=None):
	"""Computes all time domain parameters of the HRV time domain module
		and returns them in a ReturnTuple object.

	References: [Peng1995][Willson2002]

	Parameters
	----------
	nni : array
		NN intervals in [ms] or [s].
	rpeaks : array
		R-peak times in [ms] or [s].
	signal : array
		ECG signal.
	sampling_rate : int, float
		Sampling rate used for the ECG acquisition in (Hz).
	show : bool, optional
		If True, shows DFA plot.

	kwargs_poincare : dict, optional
		Dictionary containing the kwargs for the nonlinear 'poincare()' function:
			..	ellipse : bool, optional
					If true, shows fitted ellipse in plot (default: True).
			..	vectors : bool, optional
					If true, shows SD1 and SD2 vectors in plot (default: True).
			..	legend : bool, optional
					If True, adds legend to the Poincaré plot (default: True).
			..	marker : character, optional
					NNI marker in plot (default: 'o').

	kwargs_dfa : dict, optional
		Dictionary containing the kwargs for the nonlinear 'dfa()' function:
			..	short : array, 2 elements
					Interval limits of the short term fluctuations (default: None: [4, 16]).
			..	long : array, 2 elements
					Interval limits of the long term fluctuations (default: None: [17, 64]).
			..	legend : bool
					If True, adds legend with alpha1 and alpha2 values to the DFA plot (default: True)

	kwargs_sampen : dict, optional
		Dictionary containing the kwargs for the nonlinear 'sample_entropy()' function:
			..	dim : int, optional
					Entropy embedding dimension (default: 2).
			..	tolerance : int, float, optional
					Tolerance distance for which the vectors to be considered equal (default: std(NNI) * 0.2).

	Returns
	-------
	results : biosppy.utils.ReturnTuple object
		All time domain results.

	Returned Parameters
	-------------------
	..	SD1	in [ms] (key: 'sd1')
	..	SD2 in [ms] (key: 'sd2')
	..	SD2/SD1 [-] (key: 'sd_ratio')
	..	Area of the fitted ellipse in [ms^2] (key: 'ellipse_area')
	..	Sample Entropy [-] (key: 'sampen')
	..	Detrended Fluctuations Analysis [-] (short and long term fluctuations (key: 'dfa_short', 'dfa_long')

	Returned Figures
	----------------
	..	Poincaré plot (key: 'poincare_plot')

	Notes
	-----
	..	Results are stored in a biosppy.utils.ReturnTuple object and need to be accessed with the respective keys as
		done with dictionaries (see list of parameters and keys above)
	..	Provide at least one type of input data (signal, nn, or rpeaks).
	..	Input data will be prioritized in the following order: 1. signal, 2. nn, 3. rpeaks.
	..	NN and R-peak series provided in [s] format will be converted to [ms] format.
	..	Currently only calls the poincare() function.

	Raises
	------
	TypeError
		If no input data for 'nn', 'rpeaks', and 'signal' provided.

	"""
	# Check input
	if signal is not None:
		rpeaks = ecg(signal=signal, sampling_rate=sampling_rate, show=False)[2]
	elif nni is None and rpeaks is None:
		raise TypeError('No input data provided. Please specify input data.')

	# Get NNI series
	nn = pyhrv.utils.check_input(nni, rpeaks)

	# Unwrap kwargs_poincare dictionary & compute Poincaré
	if kwargs_poincare is not None:
		if type(kwargs_poincare) is not dict:
			raise TypeError("Expected <type 'dict'>, got %s: 'kwargs_poincare' must be a dictionary containing "
							"parameters (keys) and values for the 'poincare()' function." % type(kwargs_poincare))

		# Supported kwargs
		available_kwargs = ['ellipse', 'vectors', 'legend', 'marker']

		# Unwrwap kwargs dictionaries
		ellipse = kwargs_poincare['ellipse'] if 'ellipse' in kwargs_poincare.keys() else True
		vectors = kwargs_poincare['vectors'] if 'vectors' in kwargs_poincare.keys() else True
		legend = kwargs_poincare['legend'] if 'legend' in kwargs_poincare.keys() else True
		marker = kwargs_poincare['marker'] if 'marker' in kwargs_poincare.keys() else 'o'

		unsupported_kwargs = []
		for args in kwargs_poincare.keys():
			if args not in available_kwargs:
				unsupported_kwargs.append(args)

		# Throw warning if additional unsupported kwargs have been provided
		if unsupported_kwargs:
			warnings.warn("Unknown kwargs for 'poincare()': %s. These kwargs have no effect."
						  % unsupported_kwargs, stacklevel=2)

		# Compute Poincaré plot with custom configuration
		p_results = poincare(nn, show=False, ellipse=ellipse, vectors=vectors, legend=legend, marker=marker)
	else:
		# Compute Poincaré plot with default values
		p_results = poincare(nn, show=False)

	# Unwrap kwargs_sampen dictionary & compute Sample Entropy
	if kwargs_sampen is not None:
		if type(kwargs_sampen) is not dict:
			raise TypeError("Expected <type 'dict'>, got %s: 'kwargs_sampen' must be a dictionary containing "
							"parameters (keys) and values for the 'sample_entropy()' function." % type(kwargs_sampen))

		# Supported kwargs
		available_kwargs = ['dim', 'tolerance']

		# Unwrwap kwargs dictionaries
		dim = kwargs_sampen['dim'] if 'dim' in kwargs_sampen.keys() else 2
		tolerance = kwargs_sampen['tolerance'] if 'tolerance' in kwargs_sampen.keys() else None

		unsupported_kwargs = []
		for args in kwargs_sampen.keys():
			if args not in available_kwargs:
				unsupported_kwargs.append(args)

		# Throw warning if additional unsupported kwargs have been provided
		if unsupported_kwargs:
			warnings.warn("Unknown kwargs for 'sample_entropy()': %s. These kwargs have no effect."
						  % unsupported_kwargs, stacklevel=2)

		# Compute Poincaré plot with custom configuration
		s_results = sample_entropy(nn, dim=dim, tolerance=tolerance)
	else:
		# Compute Poincaré plot with default values
		s_results = sample_entropy(nn)

	# Unwrap kwargs_dfa dictionary & compute Poincaré
	if kwargs_dfa is not None:
		if type(kwargs_dfa) is not dict:
			raise TypeError("Expected <type 'dict'>, got %s: 'kwargs_dfa' must be a dictionary containing "
							"parameters (keys) and values for the 'dfa()' function." % type(kwargs_dfa))

		# Supported kwargs
		available_kwargs = ['short', 'legend', 'long']

		# Unwrwap kwargs dictionaries
		short = kwargs_dfa['short'] if 'short' in kwargs_dfa.keys() else None
		long = kwargs_dfa['long'] if 'long' in kwargs_dfa.keys() else None
		legend = kwargs_dfa['legend'] if 'legend' in kwargs_dfa.keys() else True

		unsupported_kwargs = []
		for args in kwargs_dfa.keys():
			if args not in available_kwargs:
				unsupported_kwargs.append(args)

		# Throw warning if additional unsupported kwargs have been provided
		if unsupported_kwargs:
			warnings.warn("Unknown kwargs for 'dfa()': %s. These kwargs have no effect."
						  % unsupported_kwargs, stacklevel=2)

		# Compute Poincaré plot with custom configuration
		d_results = dfa(nn, show=False, short=short, long=long, legend=legend)
	else:
		# Compute Poincaré plot with default values
		d_results = dfa(nn, show=False)

	# Join Results
	results = pyhrv.utils.join_tuples(p_results, s_results, d_results)

	# Plot
	if show:
		plt.show()

	# Output
	return results


if __name__ == "__main__":
	"""
	Example Script - Nonlinear Parameters
	"""
	import numpy as np

	# Get Sample Data
	nni = np.loadtxt(r"C:\Users\akito\Desktop\stress\03.Analysis\Analysis_BioSignal\ECG\RRI_kaneko_2019-11-21_14-58-59.csv",
                     delimiter=",")

	# Compute Poincaré
	res1 = poincare(nni, show=False)

	# Compute Triangular Index
	res2 = sample_entropy(nni)

	# Compute Detrended Fluctuation Analysis
	res3 = dfa(nni, show=False)
	# Join results
	results = pyhrv.utils.join_tuples(res1, res2, res3)

	# Results
	print("=========================")
	print("NONLINEAR ANALYSIS")
	print("=========================")
	print("Poincaré Plot")
	print("SD1:				%f [ms]" % results['sd1'])
	print("SD2:				%f [ms]" % results['sd2'])
	print("SD2/SD1: 		%f [ms]" % results['sd_ratio'])
	print("Area S:			%f [ms]" % results['ellipse_area'])
	print("Sample Entropy:	%f" % results['sampen'])
	print("DFA (alpha1):	%f	[ms]" % results['dfa_alpha1'])
	print("DFA (alpha2):	%f	[ms]" % results['dfa_alpha2'])

	# Alternatively use the nonlinear() function to compute all the nonlinear parameters at once
	nonlinear(nni=nni)
