from types import MethodType, FunctionType
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hilbert
import matplotlib.pyplot as plt


class FluorescentProteinPolarization:
    """
    Calculating the polarization from FP data
    """

    def __init__(self, **kwargs):
        """
         The following parameters are to be specified as arguments:

        """
        self.__dict__.update(kwargs)

        self.lamb = np.loadtxt(self.filename, dtype='float')[:, 0]
        self.absorption = np.loadtxt(self.filename, dtype='float')[:, 1]

        self.freq = np.zeros_like(self.lamb)
        self.freq_new = np.zeros(self.interpolate_array_length)
        self.t_plus = np.zeros(self.interpolate_array_length / 2 - 1)
        self.g = np.zeros_like(self.t_plus)

    def fourier_1(self, time, time_data):
        freq = np.fft.fftshift(np.fft.fftfreq(len(time), time[1] - time[0]))
        J = (-1) ** np.arange(len(time_data))  # SIAM J. SCI. COMPUT. Vol.15, No.5, pp.1105-1110.
        return [freq, J * np.fft.fft(J * time_data)]

    def inverse_fourier_1(self, frequency, freq_data):
        t = np.fft.fftshift(np.fft.fftfreq(len(frequency), frequency[1] - frequency[0]))
        J = (-1) ** np.arange(len(freq_data))  # SIAM J. SCI. COMPUT. Vol.15, No.5, pp.1105-1110.
        return [t, J * np.fft.ifft(J * freq_data)]

    def spectra2lineshape(self):
        self.freq = 2. * np.pi * 300. / np.array(self.lamb)
        interpolation_function = interp1d(self.freq, self.absorption)
        d_freq = (self.freq[-1] - self.freq[0]) / self.interpolate_array_length
        freq_new = np.array(np.linspace(self.freq[0], self.freq[-1] - d_freq, self.interpolate_array_length))
        absorption_new = interpolation_function(freq_new)
        absorption_new *= 2 * np.pi * 3e2 / (freq_new * freq_new)

        [t, exp_lineshape] = self.inverse_fourier_1(freq_new, hilbert(absorption_new))
        t = t[::-1]
        self.t_plus = t[t.size / 2 - 1:]
        exp_lineshape = exp_lineshape[::-1]
        exp_lineshape_plus = exp_lineshape[exp_lineshape.size / 2 - 1:]

        self.g = np.log(exp_lineshape_plus)
        plt.figure()
        plt.subplot(211)
        plt.plot(self.t_plus, self.g.real, 'r', label='lineshape_real')
        plt.subplot(212)
        plt.plot(self.t_plus, self.g.imag, 'b', label='lineshape_imag')
        plt.show()
        return 0

    def lineshape2spectra(self):
        return 0

    def response_functions(self):
        return 0

    def polarization(self):
        return 0

    def __call__(self, parameters):
        self.spectra2lineshape()
        self.polarization()
        return 0

if __name__ == '__main__':

    print(FluorescentProteinPolarization.__doc__)

    FluorescentProteinPolarization(
        # kwargs
        filename="Data/tag_RFP_abs.txt",
        interpolate_array_length=256,

    )(
        (
            # parameters
        )
    )