import numpy as np
from astropy.table import Table
from scipy.stats import truncnorm
from scipy.interpolate import LinearNDInterpolator
from scipy.stats import binned_statistic_2d
import healpy as hp

def calc_delta_theta_analytic(a0, i, e):
    """
    Analytic prediction of the astrometric excess noise (mas)
    
    a0 : photocenter size (mas)
    i : inclination (radians)
    e : eccentricity
    """
    cosi = np.cos(i)
    cterm = 1 + cosi**2
    eterm = 1 - 0.75 * e**2
    delta_theta_analytic = a0 * np.sqrt(0.5 * cterm * eterm)
    
    return delta_theta_analytic


def pred_ruwe_obs(a0, G, i, e, ra, dec):
    """
    Predict the size of RUWE in the Gaia DR3 NSS catalog.
    
    a0 : photocenter size (mas)
    G : Gaia G magnitude
    i : inclination (radians)
    e : eccentricity
    ra, dec (decimal deg)
    """
    # Get the number of visibility periods
    nvis = healpix_nviz_periods(ra, dec)

    # Figure how much to re-scale the analytic prediction of the astrometric excess noise.
    # This rescaling is a function of the number of visibility periods used.
    dtheta_scale = np.ones(len(nvis))
    
    dtheta_scale[nvis < 23] = np.random.normal(loc=0.5735900796838884,
                                         scale=0.13284153743893617,
                                         size=(nvis < 23).sum())
    dtheta_scale[(nvis >= 23) & (nvis < 25)] = np.random.normal(loc=0.6059692325966662,
                                                          scale=0.11608460929891715,
                                                          size=((nvis >= 23) & (nvis < 25)).sum())
    dtheta_scale[nvis >= 25] = np.random.normal(loc=0.6428020142753021,
                                          scale=0.0985709259351141,
                                          size=(nvis >= 25).sum())

    # Calculate analytic delta theta.
    dtheta_analytic = calc_delta_theta_analytic(a0, i, e)

    # Calculate sigmaAL
    sigma_al = al_uncertainty_per_ccd_interp(G)

    # Calculate ruwe
    ruwe_pred = np.sqrt(1 + (dtheta_analytic * dtheta_scale/sigma_al)**2)
    
    return ruwe_pred


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

def get_parallax_error_values_nss(G, nvis):
    """
    Estimate the photocenter orbit size uncertainty, based on Gaia G magnitude and number of visibility periods used nvis.
    """
    t = Table.read('nss_info.fits')
    parallax_error = np.nan * np.ones(len(G))
    
    med, x_edge, y_edge, binnumber = binned_statistic_2d(t['phot_g_mean_mag'],
                                                         t['visibility_periods_used'],
                                                         np.log10(t['parallax_error']),
                                                         bins=[np.arange(2.9, 19.11, 0.2), np.arange(12.5, 34.5)],
                                                         statistic='median')

    std, x_edge, y_edge, binnumber = binned_statistic_2d(t['phot_g_mean_mag'],

                                                         t['visibility_periods_used'],
                                                         np.log10(t['parallax_error']),
                                                         bins=[np.arange(2.9, 19.11, 0.2), np.arange(12.5, 34.5)],
                                                         statistic='std')

    mnm, x_edge, y_edge, binnumber = binned_statistic_2d(t['phot_g_mean_mag'],
                                                         t['visibility_periods_used'],
                                                         np.log10(t['parallax_error']),
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
    log10_parallax_error = truncnorm.rvs((mnm_out - med_out)/std_out, 999,
                                         loc=med_out, scale=std_out)
    parallax_error[~np.isnan(med_interp(G, nvis))] = 10**log10_parallax_error

    nnan = np.isnan(med_interp(G, nvis)).sum()
    
    return parallax_error

def healpix_nviz_periods(ra, dec):
    """
    Get the number of Gaia DR3 visibility periods.
    ra, dec : decimal degrees
    """
    t = Table.read('nside64_npix_dr3_nvis.fits')
    
    num = hp.ang2pix(64, np.radians(90.0 - np.array(dec)), np.radians(np.array(ra)))

    nvis = t[num]['nvis']
    
    return nvis

def empirical_period_probability(period):
    """
    Given the orbital period, get the probability that the solution will get thrown out.
    Then apply that and return only the indicies and kept periods.
    """
    t = Table.read('empirical_period_filter.dat', format='ascii')
    pdf_x = t['period']
    pdf_y = t['prob']
    
    prob = np.ones(len(period))

    idx = np.where((period > 250) & (period < 3000))[0]
    prob[idx] = np.interp(period[idx], pdf_x, pdf_y)
    
    rng = np.random.default_rng()
    random = rng.random(len(period))

    idx = np.where(period >= 3000)[0]
    prob[idx] = 0
                   
    keep_idx = prob > random

    return keep_idx, period[keep_idx]
