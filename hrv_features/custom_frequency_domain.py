# -*- coding: utf-8 -*-
# 注意　nfft = 2**14
# frequency_domainを変更したバージョン

import frequency_domain as fd
from util import detrending

# Imports
import warnings
import spectrum
import numpy as np
import scipy as sp
from scipy.signal import welch
import biosppy
from biosppy import utils
# Local imports/HRV toolbox imports
import pyhrv.tools as tools

# Surpress Lapack bug 0038 warning from scipy (may occur with older versions of the packages above)
warnings.filterwarnings(action="ignore", module="scipy")
def welch_psd(detrend_nni = None,
              fs = 4.,
			  fbands=None,
			  nfft=2**10,
			  detrend=True,
			  window='hamming',
			  show=True,
			  show_param=True,
			  legend=True,
			  mode='normal'):
	"""Computes a Power Spectral Density (PSD) estimation from the NNI series using the Welch’s method
	and computes all frequency domain parameters from this PSD according to the specified frequency bands.

	References: [Electrophysiology1996], [Umberto2017], [Welch2017]
	Docs:		https://pyhrv.readthedocs.io/en/latest kwa/_pages/api/frequency.html#welch-s-method-welch-psd

	Parameters
	----------
	nni : array
		NN-Intervals in [ms] or [s]
	rpeaks : array
		R-peak locations in [ms] or [s]
	fbands : dict, optional
		Dictionary with frequency bands (2-element tuples or list)
		Value format:	(lower_freq_band_boundary, upper_freq_band_boundary)
		Keys:	'ulf'	Ultra low frequency		(default: none) optional
				'vlf'	Very low frequency		(default: (0.000Hz, 0.04Hz))
				'lf'	Low frequency			(default: (0.04Hz - 0.15Hz))
				'hf'	High frequency			(default: (0.15Hz - 0.4Hz))
	nfft : int, optional
		Number of points computed for the FFT result (default: 2**12)
	detrend : bool optional
		If True, detrend NNI series by subtracting the mean NNI (default: True)
	window : scipy window function, optional
		Window function used for PSD estimation (default: 'hamming')
	show : bool, optional
		If true, show PSD plot (default: True)
	show_param : bool, optional
		If true, list all computed PSD parameters next to the plot (default: True)
	legend : bool, optional
		If true, add a legend with frequency bands to the plot (default: True)

	Returns
	-------
	results : biosppy.utils.ReturnTuple object
		All results of the Welch's method's PSD estimation (see list and keys below)

	Returned Parameters & Keys
	--------------------------
	..	Peak frequencies of all frequency bands in [Hz] (key: 'fft_peak')
	..	Absolute powers of all frequency bands in [ms^2][(key: 'fft_abs')
	..	Relative powers of all frequency bands [%] (key: 'fft_rel')
	..	Logarithmic powers of all frequency bands [-] (key: 'fft_log')
	..	Normalized powers of all frequency bands [-] (key: 'fft_norms')
	..	LF/HF ratio [-] (key: 'fft_ratio')
	..	Total power over all frequency bands in [ms^2] (key: 'fft_total')
	..	Interpolation method used for NNI interpolation (key: 'fft_interpolation')
	..	Resampling frequency used for NNI interpolation (key: 'fft_resampling_frequency')
	..	Spectral window used for PSD estimation of the Welch's method (key: 'fft_spectral_window)'

	"""

	# Verify or set default frequency bands
	fbands = fd._check_freq_bands(fbands)

	nn_interpol = detrend_nni
    # Subtract mean value from each sample for surpression of DC-offsets
	if detrend:
		nn_interpol = nn_interpol - np.mean(nn_interpol)

	# Adapt 'nperseg' according to the total duration of the NNI series (5min threshold = 300000ms)
	if (len(nn_interpol)/fs) < 300:
		nperseg = nfft
	else:
		nperseg = 300

	# Compute power spectral density estimation (where the magic happens)
	frequencies, powers = welch(
		x=nn_interpol,
		fs=fs,
		window=window,
		nperseg=nfft,
		nfft=nfft,
		scaling='density'
	)

	if mode not in ['normal', 'dev', 'devplot']:
		warnings.warn("Unknown mode '%s'. Will proceed with 'normal' mode." % mode, stacklevel=2)
		mode = 'normal'

	# Normal Mode:
	# Returns frequency parameters, PSD plot figure and no frequency & power series/arrays
	if mode == 'normal':
		# Compute frequency parameters
		params, freq_i = fd._compute_parameters('fft', frequencies, powers, fbands)

		# Plot PSD
		figure = fd._plot_psd('fft', frequencies, powers, freq_i, params, show, show_param, legend)
		figure = utils.ReturnTuple((figure, ), ('fft_plot', ))

		# Output
		return tools.join_tuples(params)

	# Dev Mode:
	# Returns frequency parameters and frequency & power series/array; does not create a plot figure nor plot the data
	elif mode == 'dev':
		# Compute frequency parameters
		params, _ = _compute_parameters('fft', frequencies, powers, fbands)

		# Output
		return tools.join_tuples(params, meta), frequencies, (powers / 10 ** 6)

	# Devplot Mode:
	# Returns frequency parameters, PSD plot figure, and frequency & power series/arrays
	elif mode == 'devplot':
		# Compute frequency parameters
		params, freq_i = _compute_parameters('fft', frequencies, powers, fbands)

		# Plot PSD
		figure = _plot_psd('fft', frequencies, powers, freq_i, params, show, show_param, legend)
		figure = utils.ReturnTuple((figure, ), ('fft_plot', ))

		# Output
		return tools.join_tuples(params, figure, meta), frequencies, (powers / 10 ** 6)
if __name__ == '__main__':
    rri = np.loadtxt(r"Z:\theme\mental_stress\03.Analysis\Analysis_BioSignal\ECG\RRI_kishida_2019-10-22.csv",delimiter=",")
    detrending_rri = detrending.detrend(rri, Lambda= 500)
    ts = np.arange(0,len(detrending_rri)*0.25,step=0.25)
    fs= 4
    start= 300
    duration = 300
    freq_parameter = welch_psd(detrending_rri[(ts > start) &(ts <= (start + duration) )],fs = fs, nfft=2 ** 12)