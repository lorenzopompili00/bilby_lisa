import numpy as np
import matplotlib.pyplot as plt
import lal
import lalsimulation as lalsim

from bbhx.utils.transform import LISA_to_SSB

from ldc.waveform.waveform import NumericHpHc
from ldc.lisa.orbits import Orbits
from ldc.lisa.projection import ProjectedStrain

from bilby.core import utils
from bilby.core.utils import logger
from bilby.gw.conversion import bilby_to_lalsimulation_spins

from .source import LISAPolarizationDict


def lisa_binary_black_hole_TD(frequency_array, mass_1, mass_2, luminosity_distance, 
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

    frequency_bounds = ((frequency_array >= minimum_frequency) *
                        (frequency_array <= maximum_frequency))

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
        hSp, hSc = GenerateTDWaveform(gwsignal_dict, wf_gen)
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

    cos2psi = np.cos(2.0 * psi)
    sin2psi = np.sin(2.0 * psi)

    # Transform to the SSB frame with the polarization angle
    hp = hSp * cos2psi - hSc * sin2psi
    hc = hSp * sin2psi + hSc * cos2psi

    h = hp +1j * hc
    times = deltaT * np.arange(len(h))

    # Compute TD LISA response

    # We use 'ra' and 'dec' to remain consistent with existing models in `bilby`
    EclipticLongitude = ra
    EclipticLatitude = dec

    t_min = times[0]
    t_max = times[-1]

    hphc_num = NumericHpHc(times, hp, hc, EclipticLongitude, EclipticLatitude)

    config = {
        "nominal_arm_length": 2.5e9,  # "m",
        "initial_rotation": 0,  #'rad',
        "initial_position": 0,  #'rad',
        "orbit_type": "analytic",
    }

    orbits = Orbits.type(config)
    P = ProjectedStrain(orbits)

    yArm = P.arm_response(t_min, t_max, deltaT, [hphc_num], tt_order=1)

    tdi_X_num = P.compute_tdi_x(times)
    tdi_Y_num = P.compute_tdi_y(times)
    tdi_Z_num = P.compute_tdi_z(times)

    A = (tdi_Z_num - tdi_X_num) / np.sqrt(2)
    E = (tdi_X_num - 2 * tdi_Y_num + tdi_Z_num) / np.sqrt(6)
    T = (tdi_X_num + tdi_Y_num + tdi_Z_num) / np.sqrt(3)

    # FFT of TD LISA response following LAL routines
    epoch = lal.LIGOTimeGPS(geocent_time)

    A_lal = lal.CreateREAL8TimeSeries(
        "A", epoch, 0, deltaT, lal.DimensionlessUnit, len(A)
    )
    E_lal = lal.CreateREAL8TimeSeries(
        "E",epoch, 0, deltaT, lal.DimensionlessUnit, len(E)
    )
    T_lal = lal.CreateREAL8TimeSeries(
        "T",epoch, 0, deltaT, lal.DimensionlessUnit, len(T)
    )

    A_lal.data.data = A
    E_lal.data.data = E
    T_lal.data.data = T

    lalsim.SimInspiralREAL8WaveTaper(A_lal.data, 1)
    lalsim.SimInspiralREAL8WaveTaper(E_lal.data, 1)
    lalsim.SimInspiralREAL8WaveTaper(T_lal.data, 1)

    # Adjust signal duration
    chirplen = int(1.0 / (deltaF * deltaT))

    # resize waveforms to the required length
    lal.ResizeREAL8TimeSeries(A_lal, A_lal.data.length - chirplen, chirplen)
    lal.ResizeREAL8TimeSeries(E_lal, E_lal.data.length - chirplen, chirplen)
    lal.ResizeREAL8TimeSeries(T_lal, T_lal.data.length - chirplen, chirplen)

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
    T_tilde = lal.CreateCOMPLEX16FrequencySeries(
        "FD H_CROSS",
        T_lal.epoch,
        0.0,
        deltaF,
        lal.DimensionlessUnit,
        int(chirplen / 2.0 + 1),
    )

    plan = lal.CreateForwardREAL8FFTPlan(chirplen, 0)
    lal.REAL8TimeFreqFFT(A_tilde, A_lal, plan)
    lal.REAL8TimeFreqFFT(E_tilde, E_lal, plan)
    lal.REAL8TimeFreqFFT(T_tilde, T_lal, plan)

    frequency_array_2 = np.arange(len(A_tilde.data.data)) * A_tilde.deltaF
    frequency_bounds_2 = ((frequency_array_2 >= minimum_frequency) *
                        (frequency_array_2 <= maximum_frequency))

    A_tilde.data.data *= frequency_bounds_2
    E_tilde.data.data *= frequency_bounds_2
    T_tilde.data.data *= frequency_bounds_2

    # plt.loglog(frequency_array_2, np.abs(A_tilde.data.data))
    # plt.savefig("test.png")

    indnzero_res = np.argwhere(np.abs(A_tilde.data.data) > 0)
    indbeg_res = indnzero_res[0, 0]

    A_new = np.zeros_like(frequency_array, dtype=complex)
    E_new = np.zeros_like(frequency_array, dtype=complex)
    T_new = np.zeros_like(frequency_array, dtype=complex)

    if len(A_tilde.data.data) > len(frequency_array):
        logger.debug("GWsignal waveform longer than bilby's `frequency_array`" +
                     "({} vs {}), ".format(len(A_tilde.data.data), len(frequency_array)) +
                     "probably because padded with zeros up to the next power of two length." +
                     " Truncating GWsignal array.")
        A_new = A_tilde.data.data[indbeg_res:len(A_new)+indbeg_res]
        E_new = E_tilde.data.data[indbeg_res:len(E_new)+indbeg_res]
        T_new = T_tilde.data.data[indbeg_res:len(T_new)+indbeg_res]

    else:
        A_new[:len(A_tilde.data.data)] = A_tilde.data.data
        E_new[:len(E_tilde.data.data)] = E_tilde.data.data
        T_new[:len(T_tilde.data.data)] = T_tilde.data.data

    A_new *= frequency_bounds
    E_new *= frequency_bounds
    T_new *= frequency_bounds

    dt = 1 / A_tilde.deltaF + float(A_tilde.epoch)
    time_shift = np.exp(-1j * 2 * np.pi * dt * frequency_array[frequency_bounds])
    A_new[frequency_bounds] *= time_shift
    E_new[frequency_bounds] *= time_shift
    T_new[frequency_bounds] *= time_shift

    tape_fact = 1e-4

    max_idx_A = np.argmax(np.abs(A_new))
    max_idx_E = np.argmax(np.abs(E_new))
    max_idx_T = np.argmax(np.abs(T_new))

    # Find the first index after max_idx_A where the condition is satisfied
    indices_A = np.argwhere(np.abs(A_new) < np.amax(np.abs(A_new)) * tape_fact)
    indA = indices_A[indices_A > max_idx_A].min() if np.any(indices_A > max_idx_A) else len(A_new)
    A_new[indA:] = 0

    indices_E = np.argwhere(np.abs(E_new) < np.amax(np.abs(E_new)) * tape_fact)
    indE = indices_E[indices_E > max_idx_E].min() if np.any(indices_E > max_idx_E) else len(E_new)
    E_new[indE:] = 0

    indices_T = np.argwhere(np.abs(T_new) < np.amax(np.abs(T_new)) * tape_fact)
    indT = indices_T[indices_T > max_idx_T].min() if np.any(indices_T > max_idx_T) else len(T_new)
    T_new[indT:] = 0

    _waveform_dict = {"LISA_A": A_new, "LISA_E": E_new, "LISA_T": T_new}

    _waveform_dict = LISAPolarizationDict(
        {key: _waveform_dict.get(key, None) for key in _implemented_channels}
    )
    return _waveform_dict


def lisa_binary_black_hole_pseob_TD(frequency_array, mass_1, mass_2, luminosity_distance, 
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

    frequency_bounds = ((frequency_array >= minimum_frequency) *
                        (frequency_array <= maximum_frequency))

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
        hSp, hSc = GenerateTDWaveform(gwsignal_dict, wf_gen)
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

    cos2psi = np.cos(2.0 * psi)
    sin2psi = np.sin(2.0 * psi)

    # Transform to the SSB frame with the polarization angle
    hp = hSp * cos2psi - hSc * sin2psi
    hc = hSp * sin2psi + hSc * cos2psi

    h = hp +1j * hc
    times = deltaT * np.arange(len(h))

    # Compute TD LISA response

    # We use 'ra' and 'dec' to remain consistent with existing models in `bilby`
    EclipticLongitude = ra
    EclipticLatitude = dec

    t_min = times[0]
    t_max = times[-1]

    hphc_num = NumericHpHc(times, hp, hc, EclipticLongitude, EclipticLatitude)

    config = {
        "nominal_arm_length": 2.5e9,  # "m",
        "initial_rotation": 0,  #'rad',
        "initial_position": 0,  #'rad',
        "orbit_type": "analytic",
    }

    orbits = Orbits.type(config)
    P = ProjectedStrain(orbits)

    yArm = P.arm_response(t_min, t_max, deltaT, [hphc_num], tt_order=1)

    tdi_X_num = P.compute_tdi_x(times)
    tdi_Y_num = P.compute_tdi_y(times)
    tdi_Z_num = P.compute_tdi_z(times)

    A = (tdi_Z_num - tdi_X_num) / np.sqrt(2)
    E = (tdi_X_num - 2 * tdi_Y_num + tdi_Z_num) / np.sqrt(6)
    T = (tdi_X_num + tdi_Y_num + tdi_Z_num) / np.sqrt(3)

    # FFT of TD LISA response following LAL routines
    epoch = lal.LIGOTimeGPS(geocent_time)

    A_lal = lal.CreateREAL8TimeSeries(
        "A", epoch, 0, deltaT, lal.DimensionlessUnit, len(A)
    )
    E_lal = lal.CreateREAL8TimeSeries(
        "E",epoch, 0, deltaT, lal.DimensionlessUnit, len(E)
    )
    T_lal = lal.CreateREAL8TimeSeries(
        "T",epoch, 0, deltaT, lal.DimensionlessUnit, len(T)
    )

    A_lal.data.data = A
    E_lal.data.data = E
    T_lal.data.data = T

    lalsim.SimInspiralREAL8WaveTaper(A_lal.data, 1)
    lalsim.SimInspiralREAL8WaveTaper(E_lal.data, 1)
    lalsim.SimInspiralREAL8WaveTaper(T_lal.data, 1)

    # Adjust signal duration
    chirplen = int(1.0 / (deltaF * deltaT))

    # resize waveforms to the required length
    lal.ResizeREAL8TimeSeries(A_lal, A_lal.data.length - chirplen, chirplen)
    lal.ResizeREAL8TimeSeries(E_lal, E_lal.data.length - chirplen, chirplen)
    lal.ResizeREAL8TimeSeries(T_lal, T_lal.data.length - chirplen, chirplen)

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
    T_tilde = lal.CreateCOMPLEX16FrequencySeries(
        "FD H_CROSS",
        T_lal.epoch,
        0.0,
        deltaF,
        lal.DimensionlessUnit,
        int(chirplen / 2.0 + 1),
    )

    plan = lal.CreateForwardREAL8FFTPlan(chirplen, 0)
    lal.REAL8TimeFreqFFT(A_tilde, A_lal, plan)
    lal.REAL8TimeFreqFFT(E_tilde, E_lal, plan)
    lal.REAL8TimeFreqFFT(T_tilde, T_lal, plan)

    frequency_array_2 = np.arange(len(A_tilde.data.data)) * A_tilde.deltaF
    frequency_bounds_2 = ((frequency_array_2 >= minimum_frequency) *
                        (frequency_array_2 <= maximum_frequency))

    A_tilde.data.data *= frequency_bounds_2
    E_tilde.data.data *= frequency_bounds_2
    T_tilde.data.data *= frequency_bounds_2

    # plt.loglog(frequency_array_2, np.abs(A_tilde.data.data))
    # plt.savefig("test.png")

    indnzero_res = np.argwhere(np.abs(A_tilde.data.data) > 0)
    indbeg_res = indnzero_res[0, 0]

    A_new = np.zeros_like(frequency_array, dtype=complex)
    E_new = np.zeros_like(frequency_array, dtype=complex)
    T_new = np.zeros_like(frequency_array, dtype=complex)

    if len(A_tilde.data.data) > len(frequency_array):
        logger.debug("GWsignal waveform longer than bilby's `frequency_array`" +
                     "({} vs {}), ".format(len(A_tilde.data.data), len(frequency_array)) +
                     "probably because padded with zeros up to the next power of two length." +
                     " Truncating GWsignal array.")
        A_new = A_tilde.data.data[indbeg_res:len(A_new)+indbeg_res]
        E_new = E_tilde.data.data[indbeg_res:len(E_new)+indbeg_res]
        T_new = T_tilde.data.data[indbeg_res:len(T_new)+indbeg_res]

    else:
        A_new[:len(A_tilde.data.data)] = A_tilde.data.data
        E_new[:len(E_tilde.data.data)] = E_tilde.data.data
        T_new[:len(T_tilde.data.data)] = T_tilde.data.data

    A_new *= frequency_bounds
    E_new *= frequency_bounds
    T_new *= frequency_bounds

    dt = 1 / A_tilde.deltaF + float(A_tilde.epoch)
    time_shift = np.exp(-1j * 2 * np.pi * dt * frequency_array[frequency_bounds])
    A_new[frequency_bounds] *= time_shift
    E_new[frequency_bounds] *= time_shift
    T_new[frequency_bounds] *= time_shift

    tape_fact = 1e-4

    max_idx_A = np.argmax(np.abs(A_new))
    max_idx_E = np.argmax(np.abs(E_new))
    max_idx_T = np.argmax(np.abs(T_new))

    # Find the first index after max_idx_A where the condition is satisfied
    indices_A = np.argwhere(np.abs(A_new) < np.amax(np.abs(A_new)) * tape_fact)
    indA = indices_A[indices_A > max_idx_A].min() if np.any(indices_A > max_idx_A) else len(A_new)
    A_new[indA:] = 0

    indices_E = np.argwhere(np.abs(E_new) < np.amax(np.abs(E_new)) * tape_fact)
    indE = indices_E[indices_E > max_idx_E].min() if np.any(indices_E > max_idx_E) else len(E_new)
    E_new[indE:] = 0

    indices_T = np.argwhere(np.abs(T_new) < np.amax(np.abs(T_new)) * tape_fact)
    indT = indices_T[indices_T > max_idx_T].min() if np.any(indices_T > max_idx_T) else len(T_new)
    T_new[indT:] = 0

    _waveform_dict = {"LISA_A": A_new, "LISA_E": E_new, "LISA_T": T_new}

    _waveform_dict = LISAPolarizationDict(
        {key: _waveform_dict.get(key, None) for key in _implemented_channels}
    )
    return _waveform_dict

