import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import hilbert
from types import MethodType, FunctionType


class PolarizationFP:
    """
    Calculation of polarization from absorption spectra using lineshape functions
    """

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        self.freq = np.zeros(self.interpolate_num)
        self.absorption = np.zeros(self.freq.size, dtype='complex')
        self.freq_shift = np.zeros_like(self.freq)
        self.freq_0 = 0.0
        self.t = np.zeros_like(self.freq)
        self.exp_lineshape = np.zeros(self.absorption.size, dtype='complex')
        self.t_plus = np.zeros(self.interpolate_num / 2)
        self.exp_lineshape_plus = np.zeros(self.t_plus.size, dtype='complex')
        self.g = np.zeros_like(self.exp_lineshape_plus)
        self.absorption_hilbert = np.zeros_like(self.absorption)

    @staticmethod
    def fourier_1(data_time):
        j = (-1)**np.arange(len(data_time))
        return j*np.fft.fft(j*data_time)

    @staticmethod
    def inverse_fourier_1(data_freq):
        j = (-1)**np.arange(len(data_freq))
        return j*np.fft.ifft(j*data_freq)

    def spectra2lineshape(self):

        # --------------------------------------------------------------------------- #
        # READING DATA AND TRANSFORMING |lambda| AXIS TO EQUALLY SPACED |omega| AXIS  #
        # --------------------------------------------------------------------------- #

        lamb, absorption_lamb = np.loadtxt(self.filename, usecols=(0, 1), unpack=True)
        freq_w2f = 2. * np.pi * 3e2 / lamb
        absorption_freq_w2f = absorption_lamb * 2. * np.pi * 3e2 / (freq_w2f*freq_w2f)

        interpolation_function = interp1d(freq_w2f, absorption_freq_w2f, kind='cubic')
        d_freq = (freq_w2f[1] - freq_w2f[0]) / self.interpolate_num
        self.freq = np.linspace(freq_w2f[0], freq_w2f[-1] - d_freq, self.interpolate_num)
        self. absorption = interpolation_function(self.freq)

        # --------------------------------------------------------------------------- #
        # A(|omega|) = RE[\int_-inf+inf e^(-g(t)) \theta(t)  e^(-i \omega t)]         #
        # => F^-1 [H[A(|omega|)]] = e^(-g(t)) \theta(t)                               #
        # --------------------------------------------------------------------------- #

        # --------------------------------------------------------------------------- #
        # --------------CHECKING THE QUALITY OF INTERPOLATION PERFORMED-------------- #
        # --------------------------------------------------------------------------- #

        assert all(self.absorption >= 0), "Interpolation made absorption negative"
        assert norm(interpolation_function(freq_w2f) - absorption_freq_w2f, 1) / norm(absorption_freq_w2f, 1) < 1e-6, \
            "Absorption interpolation error bounds crossed"

        self.absorption_hilbert = hilbert(self.absorption)
        self.freq_0 = (self.freq[0] + self.freq[-1]) / 2
        self.freq_shift = self.freq - self.freq_0
        self.t = np.fft.fftshift(np.fft.fftfreq(len(self.freq_shift), self.freq_shift[1] - self.freq_shift[0]))
        self.exp_lineshape = self.inverse_fourier_1(self.absorption_hilbert) * np.exp(1j * self.freq_0 * self.t)

        self.t_plus = self.t[:self.t.size / 2 + 1]
        self.exp_lineshape_plus = self.exp_lineshape[:self.exp_lineshape.size / 2 + 1]

        self.g = np.log(self.exp_lineshape_plus)

        plt.figure()
        plt.subplot(211)
        plt.plot(self.t_plus, self.g.real, 'r')
        plt.subplot(212)
        plt.plot(self.t_plus, self.g.imag, 'b')

        return 0

    def lineshape2spectra(self):

        assert norm(np.exp(self.g) - self.exp_lineshape_plus, 1) < 1e-6, "The exp-lineshape+plus does not match"

        exp_lineshape_l2s = np.zeros(self.t.size, dtype='complex')
        exp_lineshape_l2s[:self.t.size/2 + 1] = np.exp(self.g)

        assert norm(exp_lineshape_l2s - self.exp_lineshape, 1) < 1e-6, "The exp-lineshape does not match"

        absorption_getback = self.fourier_1(exp_lineshape_l2s * np.exp(-1j * self.freq_0 * self.t))

        assert norm(absorption_getback - self.absorption_hilbert) < 1e-6, "Original spectra not reproduced"

        return 0

    def __call__(self, parameters):

        # --------------------------------------------------------------------------- #
        # ----------CHECKING THAT FFT OF IFFT GIVES BACK THE ORIGINAL DATA----------- #
        # --------------------------------------------------------------------------- #

        ft_check_data = np.random.rand(256)
        assert norm(ft_check_data - self.inverse_fourier_1(self.fourier_1(ft_check_data)), 1) < 1e-6, \
            "ERROR: IFFT[FFT(x)] does reproduce original data x"

        self.spectra2lineshape()
        self.lineshape2spectra()
        plt.show()
        return 0

if __name__ == '__main__':
    print(PolarizationFP.__doc__)

    PolarizationFP(
        # kwargs
        filename="Data/tag_RFP_abs.txt",
        interpolate_num=256
    )(
        (
            # parameters
        )
    )