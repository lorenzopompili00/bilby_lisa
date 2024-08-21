import numpy as np
import lal
import lalsimulation as lalsim
import matplotlib.pyplot as plt

from scipy.interpolate import InterpolatedUnivariateSpline as spline
from bilby.core.utils import logger

from ldc.waveform.waveform import NumericHpHc
from ldc.lisa.orbits import Orbits
from ldc.lisa.projection import ProjectedStrain
from lisatools.sensitivity import get_sensitivity


def Fplus_0pol(lam, beta):

    return 0.5 * (1 + np.sin(beta) ** 2) * np.cos(2 * lam - np.pi / 3)


def Fcross_0pol(lam, beta):

    return np.sin(beta) * np.sin(2 * lam - np.pi / 3)


def Fplus(lam, beta, psi):

    return np.cos(2 * psi) * Fplus_0pol(lam, beta) + np.sin(2 * psi) * Fcross_0pol(
        lam, beta
    )


def Fcross(lam, beta, psi):

    return -np.sin(2 * psi) * Fplus_0pol(lam, beta) + np.cos(2 * psi) * Fcross_0pol(
        lam, beta
    )


def lisa_response_LW(h, times, ra, dec, psi, Larm=2.5e9):

    # We use 'ra' and 'dec' to remain consistent with existing models in bilby
    # Note: angles are in the LISA frame
    lam = ra
    beta = dec

    amp = np.abs(h)
    phase = np.unwrap(np.angle(h))

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

    d2ha = Fplus(lam, beta, psi) * d2hp + Fcross(lam, beta, psi) * d2hc

    d2he = (
        Fplus(lam + np.pi / 4, beta, psi) * d2hp
        + Fcross(lam + np.pi / 4, beta, psi) * d2hc
    )

    A = -3 * np.sqrt(2) * (Larm / lal.C_SI) ** 2 * d2ha
    E = -3 * np.sqrt(2) * (Larm / lal.C_SI) ** 2 * d2he

    return A, E


def lisa_response_TD(hSp, hSc, times, ra, dec, psi):

    # Transform to the SSB frame with the polarization angle
    cos2psi = np.cos(2.0 * psi)
    sin2psi = np.sin(2.0 * psi)

    hp = hSp * cos2psi - hSc * sin2psi
    hc = hSp * sin2psi + hSc * cos2psi

    # We use 'ra' and 'dec' to remain consistent with existing models in `bilby`
    # Note: angles are in the SSB frame
    lam = ra
    beta = dec

    t_min = times[0]
    t_max = times[-1]
    deltaT = times[1] - times[0]

    hphc_num = NumericHpHc(times, hp, hc, lam, beta)

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

    return A, E, T


def fft_lisa_response(
    A,
    geocent_time,
    deltaT,
    frequency_array,
    minimum_frequency,
    maximum_frequency,
):

    deltaF = frequency_array[1] - frequency_array[0]

    frequency_bounds = (frequency_array >= minimum_frequency) * (
        frequency_array <= maximum_frequency
    )

    # FFT of TD LISA response following LAL routines
    epoch = lal.LIGOTimeGPS(geocent_time)

    A_lal = lal.CreateREAL8TimeSeries(
        "A", epoch, 0, deltaT, lal.DimensionlessUnit, len(A)
    )
    A_lal.data.data = A

    lalsim.SimInspiralREAL8WaveTaper(A_lal.data, 1)

    # Adjust signal duration
    chirplen = int(1.0 / (deltaF * deltaT))

    # resize waveforms to the required length
    lal.ResizeREAL8TimeSeries(A_lal, A_lal.data.length - chirplen, chirplen)

    # FFT - Using LAL routines
    A_tilde = lal.CreateCOMPLEX16FrequencySeries(
        "FD H_PLUS",
        A_lal.epoch,
        0.0,
        deltaF,
        lal.DimensionlessUnit,
        int(chirplen / 2.0 + 1),
    )

    plan = lal.CreateForwardREAL8FFTPlan(chirplen, 0)
    lal.REAL8TimeFreqFFT(A_tilde, A_lal, plan)

    # import matplotlib.pyplot as plt
    # plt.loglog(frequency_array_2, np.abs(A_tilde.data.data))
    # plt.xlim(minimum_frequency, maximum_frequency)
    # plt.show()

    indnzero_res = np.argwhere(np.abs(A_tilde.data.data) > 0)
    indbeg_res = indnzero_res[0, 0]

    A_new = np.zeros_like(frequency_array, dtype=complex)

    if len(A_tilde.data.data) > len(frequency_array):
        logger.debug(
            "GWsignal waveform longer than bilby's `frequency_array`"
            + "({} vs {}), ".format(len(A_tilde.data.data), len(frequency_array))
            + "probably because padded with zeros up to the next power of two length."
            + " Truncating GWsignal array."
        )
        A_new = A_tilde.data.data[indbeg_res : len(A_new) + indbeg_res]

    else:
        A_new[: len(A_tilde.data.data)] = A_tilde.data.data

    A_new *= frequency_bounds

    dt = 1 / A_tilde.deltaF + float(A_tilde.epoch)
    time_shift = np.exp(-1j * 2 * np.pi * dt * frequency_array[frequency_bounds])
    A_new[frequency_bounds] *= time_shift

    tape_fact = 1e-4

    max_idx_A = np.argmax(np.abs(A_new))

    # Find the first index after max_idx_A where the condition is satisfied
    indices_A = np.argwhere(np.abs(A_new) < np.amax(np.abs(A_new)) * tape_fact)
    indA = (
        indices_A[indices_A > max_idx_A].min()
        if np.any(indices_A > max_idx_A)
        else len(A_new)
    )
    A_new[indA:] = 0

    return A_new


def plot_fd_response(frequency_array, f_min, f_max, A, E):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))
    ax1.loglog(frequency_array, np.sqrt(frequency_array) * np.abs(A))
    ax2.loglog(frequency_array, np.sqrt(frequency_array) * np.abs(E))

    fn = np.logspace(-5, -1, 10000)

    Sn_char_strain = get_sensitivity(fn, sens_fn="A1TDISens", return_type="char_strain")
    ax1.loglog(fn, Sn_char_strain, c="k")

    Sn_char_strain = get_sensitivity(fn, sens_fn="E1TDISens", return_type="char_strain")
    ax2.loglog(fn, Sn_char_strain, c="k")

    for ax in [ax1, ax2]:
        ax.set_xlabel(r"$f~[\rm{Hz}]$")
        ax.set_xlim(f_min, f_max)

    ax1.set_ylabel(r"$\tilde{A}\sqrt{f}~[\rm{Hz}^{1/2}]$")
    ax2.set_ylabel(r"$\tilde{E}\sqrt{f}~[\rm{Hz}^{1/2}]$")


    plt.tight_layout()
    plt.savefig("test_response.png")

    plt.close()
