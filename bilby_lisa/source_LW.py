import numpy as np
import matplotlib.pyplot as plt
import lal
import lalsimulation as lalsim

from scipy.interpolate import InterpolatedUnivariateSpline as spline
from bbhx.utils.transform import LISA_to_SSB

from bilby.core import utils
from bilby.core.utils import logger
from bilby.gw.conversion import bilby_to_lalsimulation_spins

from .source import LISAPolarizationDict


def Fplus_0pol(lam,beta):

    return 0.5*(1+np.sin(beta)**2)*np.cos(2*lam-np.pi/3)

def Fcross_0pol(lam,beta):

    return np.sin(beta)*np.sin(2*lam-np.pi/3)

def Fplus(lam,beta,psi):

    return np.cos(2*psi)*Fplus_0pol(lam,beta)+np.sin(2*psi)*Fcross_0pol(lam,beta)

def Fcross(lam,beta,psi):

    return -np.sin(2*psi)*Fplus_0pol(lam,beta)+np.cos(2*psi)*Fcross_0pol(lam,beta)


def lisa_LW_gwsignal_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance, 
                                       a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, 
                                       ra, dec, psi, geocent_time, **kwargs):

    from lalsimulation.gwsignal import GenerateTDWaveform
    from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator
    import astropy.units as u

    # LISA arm length
    Larm=2.5E9

    waveform_kwargs = dict(
        waveform_approximant="SEOBNRv5PHM",
        reference_frequency=1e-4,
        minimum_frequency=1e-4,
        maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False,
        mode_array=None,
        pn_amplitude_order=0,
    )
    waveform_kwargs.update(kwargs)

    waveform_approximant = waveform_kwargs['waveform_approximant']
    reference_frequency = waveform_kwargs['reference_frequency']
    minimum_frequency = waveform_kwargs['minimum_frequency']
    maximum_frequency = waveform_kwargs['maximum_frequency']
    catch_waveform_errors = waveform_kwargs['catch_waveform_errors']
    mode_array = waveform_kwargs['mode_array']
    pn_amplitude_order = waveform_kwargs['pn_amplitude_order']

    if pn_amplitude_order != 0:
        # This is to mimic the behaviour in
        # https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimInspiral.c#L5542
        if pn_amplitude_order == -1:
            if waveform_approximant in ["SpinTaylorT4", "SpinTaylorT5"]:
                pn_amplitude_order = 3  # Equivalent to MAX_PRECESSING_AMP_PN_ORDER in LALSimulation
            else:
                pn_amplitude_order = 6  # Equivalent to MAX_NONPRECESSING_AMP_PN_ORDER in LALSimulation
        start_frequency = minimum_frequency * 2. / (pn_amplitude_order + 2)
    else:
        start_frequency = minimum_frequency

    # Call GWsignal generator
    wf_gen = gwsignal_get_waveform_generator(waveform_approximant)

    delta_frequency = frequency_array[1] - frequency_array[0]

    frequency_bounds = ((frequency_array >= minimum_frequency) *
                        (frequency_array <= maximum_frequency))

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1 * utils.solar_mass, mass_2=mass_2 * utils.solar_mass,
        reference_frequency=reference_frequency, phase=phase)

    eccentricity = 0.0
    longitude_ascending_nodes = 0.0
    mean_per_ano = 0.0

    # Check if conditioning is needed
    condition = 0
    if wf_gen.metadata["implemented_domain"] == 'time':
        condition = 1

    # Adjust deltaT depending on sampling rate
    f_nyquist = maximum_frequency

    n = int(np.round(maximum_frequency / delta_frequency))
    if n & (n - 1):
        chirplen_exp = np.frexp(n)
        f_nyquist = np.ldexp(1, chirplen_exp[1]) * delta_frequency
    
    deltaT = 0.5 / f_nyquist
    deltaF = delta_frequency

    # Create dict for gwsignal generator
    gwsignal_dict = {'mass1' : mass_1 * u.solMass,
                     'mass2' : mass_2 * u.solMass,
                     'spin1x' : spin_1x * u.dimensionless_unscaled,
                     'spin1y' : spin_1y * u.dimensionless_unscaled,
                     'spin1z' : spin_1z * u.dimensionless_unscaled,
                     'spin2x' : spin_2x * u.dimensionless_unscaled,
                     'spin2y' : spin_2y * u.dimensionless_unscaled,
                     'spin2z' : spin_2z * u.dimensionless_unscaled,
                     'deltaF' : delta_frequency * u.Hz,
                     'f22_start' : start_frequency * u.Hz,
                     'f_max': maximum_frequency * u.Hz,
                     'f22_ref': reference_frequency * u.Hz,
                     'phi_ref' : phase * u.rad,
                     'distance' : luminosity_distance * u.Mpc,
                     'inclination' : iota * u.rad,
                     'eccentricity' : eccentricity * u.dimensionless_unscaled,
                     'longAscNodes' : longitude_ascending_nodes * u.rad,
                     'meanPerAno' : mean_per_ano * u.rad,
                     'deltaT': deltaT *u.s,
                     # 'ModeArray': mode_array,
                     'condition': condition
                     }

    if mode_array is not None:
        gwsignal_dict.update(ModeArray=mode_array)

    # Pass extra waveform arguments to gwsignal
    extra_args = waveform_kwargs.copy()

    for key in [
            "waveform_approximant",
            "reference_frequency",
            "minimum_frequency",
            "maximum_frequency",
            "catch_waveform_errors",
            "mode_array",
            "pn_spin_order",
            "pn_amplitude_order",
            "pn_tidal_order",
            "pn_phase_order",
            "numerical_relativity_file",
    ]:
        if key in extra_args.keys():
            del extra_args[key]

    gwsignal_dict.update(extra_args)

    try:
        hp, hc = GenerateTDWaveform(gwsignal_dict, wf_gen)
    except Exception as e:
        if not catch_waveform_errors:
            raise
        else:
            EDOM = (
                "Internal function call failed: Input domain error" in e.args[0]
            ) or "Input domain error" in e.args[
                0
            ]
            if EDOM:
                failed_parameters = dict(mass_1=mass_1, mass_2=mass_2,
                                         spin_1=(spin_1x, spin_1y, spin_1z),
                                         spin_2=(spin_2x, spin_2y, spin_2z),
                                         luminosity_distance=luminosity_distance,
                                         iota=iota, phase=phase,
                                         eccentricity=eccentricity,
                                         start_frequency=minimum_frequency)
                logger.warning("Evaluating the waveform failed with error: {}\n".format(e) +
                               "The parameters were {}\n".format(failed_parameters) +
                               "Likelihood will be set to -inf.")
                return None
            else:
                raise

    h = hp +1j * hc
    times = deltaT * np.arange(len(h))

    # Compute long-wavelength LISA response

    # We use 'ra' and 'dec' to remain consistent with existing models in `bilby`
    lam = ra
    beta = dec

    amp = np.abs(h)
    phase = np.unwrap(np.angle(h))

    indnzero = np.argwhere(amp > 0)
    indbeg = indnzero[0, 0]

    tape_fact = 1e-3
    indkeep = np.argwhere(amp > np.amax(amp) * tape_fact)
    indend = indkeep[-1, 0]

    amp = amp[indbeg:indend]
    phase = phase[indbeg:indend]
    times = times[indbeg:indend]

    amp = np.array(amp)
    phase = np.array(phase)

    spl_ampl = spline(times, amp)
    spl_phase = spline(times, phase)

    damp = spl_ampl.derivative(1)(times)
    d2amp = spl_ampl.derivative(2)(times)

    dphase = spl_phase.derivative(1)(times)
    d2phase = spl_phase.derivative(2)(times)

    d2h = (
        d2amp - (dphase) ** 2 * amp + 2 * 1j * dphase * damp + 1j * d2phase * amp
    ) * np.exp(1j * phase)

    d2hp = np.real(d2h)
    d2hc = np.imag(d2h)


    d2ha = (
        Fplus(lam, beta, psi) * d2hp
        + Fcross(lam, beta, psi) * d2hc
    )

    d2he = (
        Fplus(lam + np.pi / 4, beta, psi) * d2hp
        + Fcross(lam + np.pi / 4, beta, psi) * d2hc
    )

    A0 = -3 * np.sqrt(2) * (Larm / lal.C_SI) ** 2 * d2ha
    E0 = -3 * np.sqrt(2) * (Larm / lal.C_SI) ** 2 * d2he

    A = np.zeros(len(h))
    E = np.zeros(len(h))

    A[indbeg:indend] = np.copy(A0)
    E[indbeg:indend] = np.copy(E0)

    A = A[indbeg:indend]
    E = E[indbeg:indend]

    # FFT of TD LISA response following LAL routines
    epoch = lal.LIGOTimeGPS(geocent_time)

    A_lal = lal.CreateREAL8TimeSeries(
        "A", epoch, 0, deltaT, lal.DimensionlessUnit, len(A)
    )
    E_lal = lal.CreateREAL8TimeSeries(
        "E",epoch, 0, deltaT, lal.DimensionlessUnit, len(E)
    )

    A_lal.data.data = A
    E_lal.data.data = E

    lalsim.SimInspiralREAL8WaveTaper(A_lal.data, 1)
    lalsim.SimInspiralREAL8WaveTaper(E_lal.data, 1)

    # Adjust signal duration
    chirplen = int(1.0 / (deltaF * deltaT))

    # resize waveforms to the required length
    lal.ResizeREAL8TimeSeries(A_lal, A_lal.data.length - chirplen, chirplen)
    lal.ResizeREAL8TimeSeries(E_lal, E_lal.data.length - chirplen, chirplen)

    # FFT - Using LAL routines
    A_tilde = lal.CreateCOMPLEX16FrequencySeries(
        "FD H_PLUS",
        A_lal.epoch,
        0.0,
        deltaF,
        lal.DimensionlessUnit,
        int(chirplen / 2.0 + 1),
    )
    E_tilde = lal.CreateCOMPLEX16FrequencySeries(
        "FD H_CROSS",
        E_lal.epoch,
        0.0,
        deltaF,
        lal.DimensionlessUnit,
        int(chirplen / 2.0 + 1),
    )

    plan = lal.CreateForwardREAL8FFTPlan(chirplen, 0)
    lal.REAL8TimeFreqFFT(A_tilde, A_lal, plan)
    lal.REAL8TimeFreqFFT(E_tilde, E_lal, plan)

    frequency_array_2 = np.arange(len(A_tilde.data.data)) * A_tilde.deltaF
    frequency_bounds_2 = ((frequency_array_2 >= minimum_frequency) *
                        (frequency_array_2 <= maximum_frequency))

    A_tilde.data.data *= frequency_bounds_2
    E_tilde.data.data *= frequency_bounds_2
    
    # plt.loglog(frequency_array_2, np.abs(A_tilde.data.data))

    indnzero_res = np.argwhere(np.abs(A_tilde.data.data) > 0)
    indbeg_res = indnzero_res[0, 0]

    A_new = np.zeros_like(frequency_array, dtype=complex)
    E_new = np.zeros_like(frequency_array, dtype=complex)

    if len(A_tilde.data.data) > len(frequency_array):
        logger.debug("GWsignal waveform longer than bilby's `frequency_array`" +
                     "({} vs {}), ".format(len(A_tilde.data.data), len(frequency_array)) +
                     "probably because padded with zeros up to the next power of two length." +
                     " Truncating GWsignal array.")
        A_new = A_tilde.data.data[indbeg_res:len(A_new)+indbeg_res]
        E_new = E_tilde.data.data[indbeg_res:len(E_new)+indbeg_res]

    else:
        A_new[:len(A_tilde.data.data)] = A_tilde.data.data
        E_new[:len(E_tilde.data.data)] = E_tilde.data.data

    A_new *= frequency_bounds
    E_new *= frequency_bounds

    if condition:
        dt = 1 / A_tilde.deltaF + float(A_tilde.epoch)
        time_shift = np.exp(-1j * 2 * np.pi * dt * frequency_array[frequency_bounds])
        A_new[frequency_bounds] *= time_shift
        E_new[frequency_bounds] *= time_shift

    tape_fact = 1e-3
    indA = np.argwhere(np.abs(A_new) < np.amax(np.abs(A_new)) * tape_fact)[0,-1]
    indE = np.argwhere(np.abs(A_new) < np.amax(np.abs(A_new)) * tape_fact)[0,-1]

    A_new[indA+1:] = [0] * (len(A_new) - indA - 1)
    E_new[indE+1:] = [0] * (len(E_new) - indE - 1)

    _waveform_dict = {"LISA_A": A_new, "LISA_E": E_new}

    _implemented_channels = ["LISA_A", "LISA_E"]
    _waveform_dict = LISAPolarizationDict(
        {key: _waveform_dict.get(key, None) for key in _implemented_channels}
    )

    return _waveform_dict
