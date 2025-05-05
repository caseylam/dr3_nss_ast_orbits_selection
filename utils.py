import numpy as np
from astropy.table import Table
from scipy.stats import norm
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
import healpy as hp
import pickle

def pred_ruwe_obs_using_sims(a0, G, i, e, omega, P, ra, dec):
    """
    Predict RUWE in the Gaia DR3 NSS catalog.
    
    a0 : photocenter size (mas)
    G : Gaia G magnitude
    i : inclination (radians)
    e : eccentricity
    omega : argument of periapse (radians)
    P : orbital period (days)
    ra, dec: RA and Dec (deg)

    Return
    ------
    ruwe_pred : predicted RUWE
    """
    # Get the number of visibility periods
    nvis = healpix_nviz_periods(ra, dec)

    # Figure out how much to rescale the delta theta prediction, based on Kareem's simulations.
    # These arrays are copy/pasted from the print statement in referee1.ruwe_vs_predict.
    # nvis >= 25
    mean_arr_hi_nvis = [0.5835, 0.5749, 0.5829, 0.5556, 0.5982,
                        0.6105, 0.6039, 0.5938, 0.6070, 0.6256,
                        0.6192, 0.6127, 0.5861, 0.5539, 0.5384,
                        0.5509, 0.5639, 0.5724, 0.5843, 0.5857,
                        0.5846, 0.5924, 0.6030, 0.6002, 0.6030,
                        0.6003, 0.6055, 0.6046, 0.5955, 0.6007,
                        0.6049, 0.5962, 0.5793, 0.5796, 0.5740,
                        0.5627, 0.5422, 0.5332, 0.5264, 0.5071,
                        0.5033, 0.4848, 0.4884, 0.4708, 0.4618,
                        0.4468, 0.4425, 0.4352, 0.4223, 0.4125,
                        0.3929, 0.4006, 0.3829, 0.3780, 0.3683,
                        0.3596, 0.3502, 0.3487, 0.3405, 0.3269,
                        0.3241]

    mean_arr_lo_nvis = [0.5557, 0.5224, 0.5262, 0.5071, 0.5639,
                        0.5703, 0.5385, 0.4859, 0.5183, 0.5686,
                        0.5762, 0.5632, 0.5274, 0.4963, 0.4679,
                        0.4823, 0.4956, 0.5181, 0.5315, 0.5257,
                        0.5297, 0.5433, 0.5402, 0.5472, 0.5580,
                        0.5569, 0.5565, 0.5643, 0.5521, 0.5527,
                        0.5466, 0.5401, 0.5393, 0.5276, 0.5149,
                        0.5168, 0.5013, 0.4946, 0.4914, 0.4741,
                        0.4727, 0.4627, 0.4429, 0.4369, 0.4303,
                        0.4211, 0.4029, 0.4004, 0.3969, 0.3803,
                        0.3686, 0.3695, 0.3545, 0.3568, 0.3371,
                        0.3448, 0.3288, 0.3248, 0.3156, 0.3095,
                        0.2965]
    
    # corresponding period array. Technically, we should use the midpoints, but this is fine.
    period_bins = np.arange(0, 1526, 25)
    period_arr = 0.5 * (period_bins[:-1] + period_bins[1:])

    # Interpolate the value of the rescale, as a function of orbital period.
    dtheta_scale = np.ones(len(nvis))
    dtheta_scale[nvis >= 25] = np.interp(P, period_arr, mean_arr_hi_nvis)[nvis >= 25] + 0.15 * np.random.randn((nvis >= 25).sum())
    dtheta_scale[nvis < 25] = np.interp(P, period_arr, mean_arr_lo_nvis)[nvis < 25] + 0.15 * np.random.randn((nvis < 25).sum())
    
    # Calculate analytic delta theta.
    dtheta_analytic = calc_delta_theta_analytic(a0, i, e, omega)

    # Calculate sigmaAL
    sigma_al = al_uncertainty_per_ccd_interp(G)

    # Calculate ruwe
    ruwe_pred = np.sqrt(1 + (dtheta_analytic * dtheta_scale/sigma_al)**2)
    
    return ruwe_pred

def remove_acc9(period, a0, g):
    """
    Probabilistically remove 9-parameter acceleration solutions.

    period : orbital period (days)
    a0 : photocenter size (mas)
    g : Gaia G magnitude

    Return
    ------
    keep_idx : indices of the solutions that make it past the
    removal of 9-parameter solutions in the astrometric cascade.
    """
    p_bins = np.arange(0, 1601, 100)
    a_bins = np.arange(0, 2.1, 0.2)

    with open('acc9_prob.pickle', 'rb') as handle:
        prob = pickle.load(handle)

    x_mid = 0.5 * (p_bins[1:] + p_bins[:-1])
    y_mid = 0.5 * (a_bins[1:] + a_bins[:-1])

    p_arr, a_arr = np.meshgrid(x_mid, y_mid)
    
    probs = griddata((p_arr.flatten(), a_arr.flatten()), prob, (period, np.log10(a0/al_uncertainty_per_ccd_interp(g))))

    rng = np.random.default_rng()
    random = rng.random(len(period))
                   
    keep_idx = probs < random

    return keep_idx


def remove_acc7(period, a0, g):
    """
    Probabilistically remove 7-parameter acceleration solutions.

    period : orbital period (days)
    a0 : photocenter size (mas)
    g : Gaia G magnitude

    Return
    ------
    keep_idx : indices of the solutions that make it past the
    removal of 7-parameter solutions in the astrometric cascade.
    """
    p_bins = np.arange(0, 1601, 100)
    a_bins = np.arange(0, 2.1, 0.2)

    with open('acc7_prob.pickle', 'rb') as handle:
        prob = pickle.load(handle)

    x_mid = 0.5 * (p_bins[1:] + p_bins[:-1])
    y_mid = 0.5 * (a_bins[1:] + a_bins[:-1])

    p_arr, a_arr = np.meshgrid(x_mid, y_mid)

    probs = griddata((p_arr.flatten(), a_arr.flatten()), prob, (period, np.log10(a0/al_uncertainty_per_ccd_interp(g))))

    rng = np.random.default_rng()
    random = rng.random(len(period))
                   
    keep_idx = probs < random

    return keep_idx


def predict_single_star_parallax_error(g, nvis):
    """
    Get the predicted parallax error of a single star.
    This also assumed to be the a0 error.

    g : Gaia G magnitude
    nvis : number of visibility periods
    """
    sigmaAL = al_uncertainty_per_ccd_interp(g)

    # Scale factor.
    s = np.random.normal(-0.313, 0.047, len(sigmaAL))
    
    parallax_error = 10**s * sigmaAL/np.sqrt(nvis)

    return parallax_error

def calc_orb_error_rescale_candidate(param, period, eccentricity, g):
    """
    Return error rescale factor (fit error/predicted error) as a function of
    orbital period, eccentricity, source magnitude, and number
    of visibility periods. Interpolate over the results of
    Kareem's simulations.

    param : either 'a0' or 'parallax'
    period : orbital period (days)
    eccentricity : eccentricity
    g : Gaia G magnitude

    Return
    ------
    sig : rescale factor to correct the predicted a0 or parallax error
    """

    with open('orb_pred_' + param + '_over_error_avg.pickle', 'rb') as handle:
        avg_arr = pickle.load(handle)
    with open('orb_pred_' + param + '_over_error_std.pickle', 'rb') as handle:
        std_arr =  pickle.load(handle)
    
    log10_sig = np.zeros(len(period))
    sigmaAL = al_uncertainty_per_ccd_interp(g)

    p_bins = np.array([0, 50, 100, 150, 200, 250, 300, 320, 340, 360, 370, 380, 400, 420, 450,
                       550, 650, 950, 1050, 1150, 1250, 1350, 1450, 1550])
    e_bins = np.array([-0.01, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.01])
    # a_bins = np.arange(0, 2.1, 0.2)
    
    p_mid = 0.5 * (p_bins[1:] + p_bins[:-1])
    e_mid = 0.5 * (e_bins[1:] + e_bins[:-1])
    # a_mid = 0.5 * (a_bins[1:] + a_bins[:-1])

    # Check this indexing is right....
    p_arr, e_arr = np.meshgrid(p_mid, e_mid, indexing='ij')

    # Interpolate over the low Nvis array (note that you have to exclude solutions with non-positive standard deviations)...
    avgs = griddata((p_arr.flatten(), e_arr.flatten()), avg_arr.flatten(), (period, eccentricity))
    stds = griddata((p_arr.flatten(), e_arr.flatten()), std_arr.flatten(), (period, eccentricity))
    log10_sig[stds > 0] = norm.rvs(loc=avgs[stds > 0], scale=stds[stds > 0])

    sig = 10**log10_sig

    return sig

def calc_delta_theta_analytic(a0, i, e, omega):
    """
    Analytic prediction of the astrometric excess noise.
    
    a0 : photocenter size (mas)
    i : inclination (radians)
    e : eccentricity
    omega : argument of periapse (radians)

    Return
    ------
    delta_theta_analytic : delta theta (mas)
    """
    cosw = np.cos(omega)
    sini = np.cos(i)
    term1 = 1 - 0.5*sini**2
    term2 = -0.25*(3 + sini**2*(cosw**2 - 2))*e**2
    delta_theta_analytic = a0 * np.sqrt(term1 + term2)
    
    return delta_theta_analytic

def get_abs_mag(app_mag, parallax):
    """
    Calculte the absolution magnitude, from the apparent magnitude.
    Assumes parallaxes are in mas.
    """
    abs_mag = app_mag + 5*np.log10(parallax) - 10

    return abs_mag


def get_a_Kepler(P, M1, M2):
    """
    Calculate the semimajor axis of an orbit (in AU).
    M1, M2 : masses (Msun)
    P : orbital period (days)
    """
    a_AU =  (M1 + M2)**(1/3) * (P/365.25)**(2/3)

    return a_AU

def add_mags(mag1, mag2):
    """
    Add 2 magnitudes. NOTE this does not work with nans.
    """
    if (~np.isfinite(mag1)).sum() > 0:
        raise ValueError('mag1 must be finite.')

    if (~np.isfinite(mag2)).sum() > 0:
        raise ValueError('mag2 must be finite.')
    
    flux1 = 10**(-0.4 * mag1)
    flux2 = 10**(-0.4 * mag2)
    mag_tot = -2.5 * np.log10(flux1 + flux2)

    return mag_tot

def get_dql(mass1, mass2, mag1, mag2):
    """
    Calculate the size of the photocenter scale.
    Inputs are the masses and magnitudes of the two components.
    
    mass1, mag1 = object 1
    mass2, mag2 = object 2
    """
    l = np.zeros(len(mass1))
    q = np.zeros(len(mass1))

    lum1 = 10**(-0.4 * mag1)
    lum2 = 10**(-0.4 * mag2)
    
    ratio_21 = np.where(lum2 <= lum1)[0]
    ratio_12 = np.where(lum1 < lum2)[0]

    l[ratio_21] = lum2[ratio_21]/lum1[ratio_21]
    q[ratio_21] = mass2[ratio_21]/mass1[ratio_21]

    l[ratio_12] = lum1[ratio_12]/lum2[ratio_12]
    q[ratio_12] = mass1[ratio_12]/mass2[ratio_12]

    dql = np.abs(q-l)/((q+1)*(l+1))

    return dql

def get_app_mag(abs_mag, parallax, ext):
    """
    Distance modulus formula to convert absolute to apparent magnitude.
    abs_mag : absolute magnitude
    parallax (mas)
    ext : extinction (mag)
    """
    
    app_mag = abs_mag - 5 * np.log10(parallax) + 10 + ext
    
    return app_mag
    
def al_uncertainty_per_ccd_interp(G):
    '''
    This gives the uncertainty *per CCD* (not per FOV transit), taken from Fig 3 of https://arxiv.org/abs/2206.05439
    This is the "EDR3 adjusted" line from that Figure, which is already inflated compared to the formal uncertainties.
    '''
    G_vals =    [ 4,    5,   6,     7,   8.2,  8.4, 10,    11,    12,  13,    14,   15,   16,   17,   18,   19,  20]
    sigma_eta = [0.4, 0.35, 0.15, 0.17, 0.23, 0.13,0.13, 0.135, 0.125, 0.13, 0.15, 0.23, 0.36, 0.63, 1.05, 2.05, 4.1]
    return np.interp(G, G_vals, sigma_eta)

def get_a0_error_values_nss(G, nvis):
    """
    Estimate the photocenter orbit size uncertainty, based on Gaia G magnitude and number of visibility periods used nvis.
    """
    t = Table.read('nss_info.fits')
    a0_error = np.nan * np.ones(len(G))
    
    med, x_edge, y_edge, binnumber = binned_statistic_2d(t['phot_g_mean_mag'],
                                                         t['visibility_periods_used'],
                                                         np.log10(t['a0_error']),
                                                         bins=[np.arange(2.9, 19.11, 0.2), np.arange(12.5, 34.5)],
                                                         statistic='median')

    std, x_edge, y_edge, binnumber = binned_statistic_2d(t['phot_g_mean_mag'],
                                                         t['visibility_periods_used'],
                                                         np.log10(t['a0_error']),
                                                         bins=[np.arange(2.9, 19.11, 0.2), np.arange(12.5, 34.5)],
                                                         statistic='std')

    mnm, x_edge, y_edge, binnumber = binned_statistic_2d(t['phot_g_mean_mag'],
                                                         t['visibility_periods_used'],
                                                         np.log10(t['a0_error']),
                                                         bins=[np.arange(2.9, 19.11, 0.2), np.arange(12.5, 34.5)],
                                                         statistic='min')

    XXc, YYc = np.meshgrid(0.5 * (x_edge[1:] + x_edge[:-1]), 0.5 * (y_edge[1:] + y_edge[:-1]))

    med_points = np.array(list((zip(XXc.T[~np.isnan(med)], YYc.T[~np.isnan(med)]))))
    mnm_points = np.array(list((zip(XXc.T[~np.isnan(mnm)], YYc.T[~np.isnan(mnm)]))))
    
    med_interp = LinearNDInterpolator(med_points, med[~np.isnan(med)])
    mnm_interp = LinearNDInterpolator(mnm_points, mnm[~np.isnan(mnm)])

    med_out = med_interp(G, nvis)[~np.isnan(med_interp(G, nvis))]
    mnm_out = mnm_interp(G, nvis)[~np.isnan(mnm_interp(G, nvis))]
    std_out = np.nanmedian(std)

    log10_a0_error = truncnorm.rvs((mnm_out - med_out)/std_out, 999,
                                         loc=med_out, scale=std_out)
    a0_error[~np.isnan(med_interp(G, nvis))] = 10**log10_a0_error

    nnan = np.isnan(med_interp(G, nvis)).sum()
    
    return a0_error

def healpix_nviz_periods(ra, dec):
    """
    Get the number of Gaia DR3 visibility periods.
    ra, dec : decimal degrees
    """
    t = Table.read('nside64_npix_dr3_nvis.fits')
    
    num = hp.ang2pix(64, np.radians(90.0 - np.array(dec)), np.radians(np.array(ra)))

    nvis = t[num]['nvis']
    
    return nvis
