import numpy as np

from bilby.core import utils
from bilby.core.utils import logger
from bilby.gw.conversion import bilby_to_lalsimulation_spins
from bbhx.utils.transform import LISA_to_SSB

from .source import LISAPolarizationDict
from .source_utils import lisa_response_LW, fft_lisa_response, plot_fd_response


def lisa_binary_black_hole_LW(frequency_array, mass_1, mass_2, luminosity_distance, 
                                       a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, 
                                       ra, dec, psi, geocent_time, **kwargs):

    from lalsimulation.gwsignal import GenerateTDWaveform
    from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator
    import astropy.units as u

    _implemented_channels = ["LISA_A", "LISA_E", "LISA_T"]

    waveform_kwargs = dict(
        waveform_approximant="SEOBNRv5HM",
        reference_frequency=1e-4,
        minimum_frequency=1e-4,
        maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False,
        mode_array=None,
        pn_amplitude_order=0,
        ifos=_implemented_channels,
        debug=False,
    )
    waveform_kwargs.update(kwargs)

    waveform_approximant = waveform_kwargs['waveform_approximant']
    reference_frequency = waveform_kwargs['reference_frequency']
    minimum_frequency = waveform_kwargs['minimum_frequency']
    maximum_frequency = waveform_kwargs['maximum_frequency']
    catch_waveform_errors = waveform_kwargs['catch_waveform_errors']
    mode_array = waveform_kwargs['mode_array']
    pn_amplitude_order = waveform_kwargs['pn_amplitude_order']
    debug = waveform_kwargs['debug']

    waveform_kwargs.pop("ifos")

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

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1 * utils.solar_mass, mass_2=mass_2 * utils.solar_mass,
        reference_frequency=reference_frequency, phase=phase)

    eccentricity = 0.0
    longitude_ascending_nodes = 0.0
    mean_per_ano = 0.0

    # Adjust deltaT depending on sampling rate
    f_nyquist = maximum_frequency

    n = int(np.round(maximum_frequency / delta_frequency))
    if n & (n - 1):
        chirplen_exp = np.frexp(n)
        f_nyquist = np.ldexp(1, chirplen_exp[1]) * delta_frequency

    deltaT = 0.5 / f_nyquist

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
                     'condition': 1
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
    A, E = lisa_response_LW(h, times, ra, dec, psi)

    A_new = fft_lisa_response(
        A,
        geocent_time,
        deltaT,
        frequency_array,
        minimum_frequency,
        maximum_frequency,
    )

    E_new = fft_lisa_response(
        E,
        geocent_time,
        deltaT,
        frequency_array,
        minimum_frequency,
        maximum_frequency,
    )
    T_new = np.zeros_like(A_new, dtype=complex)

    _waveform_dict = {"LISA_A": A_new, "LISA_E": E_new, "LISA_T": T_new}

    _waveform_dict = LISAPolarizationDict(
        {key: _waveform_dict.get(key, None) for key in _implemented_channels}
    )

    if debug == True:
        plot_fd_response(frequency_array, A_new, E_new)

    return _waveform_dict


def lisa_binary_black_hole_pseob_LW(frequency_array, mass_1, mass_2, luminosity_distance, 
                                       a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, 
                                       ra, dec, psi, geocent_time, domega220, dtau220, domega330, 
                                       dtau330, domega210, dtau210, domega440, dtau440, domega550, 
                                       dtau550, domega320, dtau320, domega430, dtau430, dA22, dw22, 
                                       dA33, dw33, dA21, dw21, dA44, dw44, dA55, dw55, dA32, dw32, 
                                       dA43, dw43, dTpeak, da6, ddSO, **kwargs):

    from lalsimulation.gwsignal import GenerateTDWaveform
    from lalsimulation.gwsignal.models import gwsignal_get_waveform_generator
    import astropy.units as u

    _implemented_channels = ["LISA_A", "LISA_E", "LISA_T"]

    waveform_kwargs = dict(
        waveform_approximant="SEOBNRv5HM",
        reference_frequency=1e-4,
        minimum_frequency=1e-4,
        maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False,
        mode_array=None,
        pn_amplitude_order=0,
        ifos=_implemented_channels,
    )
    waveform_kwargs.update(kwargs)

    waveform_approximant = waveform_kwargs['waveform_approximant']
    reference_frequency = waveform_kwargs['reference_frequency']
    minimum_frequency = waveform_kwargs['minimum_frequency']
    maximum_frequency = waveform_kwargs['maximum_frequency']
    catch_waveform_errors = waveform_kwargs['catch_waveform_errors']
    mode_array = waveform_kwargs['mode_array']
    pn_amplitude_order = waveform_kwargs['pn_amplitude_order']

    waveform_kwargs.pop("ifos")

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

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(
        theta_jn=theta_jn, phi_jl=phi_jl, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, a_1=a_1, a_2=a_2, mass_1=mass_1 * utils.solar_mass, mass_2=mass_2 * utils.solar_mass,
        reference_frequency=reference_frequency, phase=phase)

    eccentricity = 0.0
    longitude_ascending_nodes = 0.0
    mean_per_ano = 0.0

    # Adjust deltaT depending on sampling rate
    f_nyquist = maximum_frequency

    n = int(np.round(maximum_frequency / delta_frequency))
    if n & (n - 1):
        chirplen_exp = np.frexp(n)
        f_nyquist = np.ldexp(1, chirplen_exp[1]) * delta_frequency
    
    deltaT = 0.5 / f_nyquist

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
                     'condition': 1
                     }

    if mode_array is not None:
        gwsignal_dict.update(ModeArray=mode_array)

    if (
        domega220 != 0.0
        or domega330 != 0.0
        or domega210 != 0.0
        or domega440 != 0.0
        or domega550 != 0.0
        or domega320 != 0.0
        or domega430 != 0.0
    ):
        domega_dict = {'2,2': domega220,
                       '2,1': domega210,
                       '3,3': domega330,
                       '3,2': domega320,
                       '4,4': domega440,
                       '4,3': domega430,
                       '5,5': domega550,
                       }
        gwsignal_dict.update(domega_dict=domega_dict)

    if (
        dtau220 != 0.0
        or dtau330 != 0.0
        or dtau210 != 0.0
        or dtau440 != 0.0
        or dtau550 != 0.0
        or dtau320 != 0.0
        or dtau430 != 0.0
    ):
        dtau_dict = {'2,2': dtau220,
                     '2,1': dtau210,
                     '3,3': dtau330,
                     '3,2': dtau320,
                     '4,4': dtau440,
                     '4,3': dtau430,
                     '5,5': dtau550,
                     }
        gwsignal_dict.update(dtau_dict=dtau_dict)
    if (
        dA22 != 0.0
        or dA33 != 0.0
        or dA21 != 0.0
        or dA44 != 0.0
        or dA55 != 0.0
        or dA32 != 0.0
        or dA43 != 0.0
    ):
        dA_dict = {'2,2': dA22,
                   '2,1': dA21,
                   '3,3': dA33,
                   '3,2': dA32,
                   '4,4': dA44,
                   '4,3': dA43,
                   '5,5': dA55,
                   }
        gwsignal_dict.update(dA_dict=dA_dict)

    if (
        dw22 != 0.0
        or dw33 != 0.0
        or dw21 != 0.0
        or dw44 != 0.0
        or dw55 != 0.0
        or dw32 != 0.0
        or dw43 != 0.0
    ):
        dw_dict = {'2,2': dw22,
                   '2,1': dw21,
                   '3,3': dw33,
                   '3,2': dw32,
                   '4,4': dw44,
                   '4,3': dw43,
                   '5,5': dw55,
                   }
        gwsignal_dict.update(dw_dict=dw_dict)

    if dTpeak != 0.0:
        gwsignal_dict.update(dTpeak=dTpeak)

    if da6 != 0.0:
        gwsignal_dict.update(da6=da6)

    if ddSO != 0.0:
        gwsignal_dict.update(ddSO=ddSO)

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
    A, E = lisa_response_LW(h, times, ra, dec, psi)

    A_new = fft_lisa_response(
        A,
        geocent_time,
        deltaT,
        frequency_array,
        minimum_frequency,
        maximum_frequency,
    )

    E_new = fft_lisa_response(
        E,
        geocent_time,
        deltaT,
        frequency_array,
        minimum_frequency,
        maximum_frequency,
    )
    T_new = np.zeros_like(A_new, dtype=complex)

    _waveform_dict = {"LISA_A": A_new, "LISA_E": E_new, "LISA_T": T_new}

    _waveform_dict = LISAPolarizationDict(
        {key: _waveform_dict.get(key, None) for key in _implemented_channels}
    )
    return _waveform_dict
