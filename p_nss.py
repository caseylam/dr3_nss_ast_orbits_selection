import numpy as np
import utils
from astropy.coordinates import SkyCoord
import astropy.units as u

def p_nss(ra, dec, porb, e, parallax, mass1, mass2, gmag1, gmag2, n_sample = 100, mag='app', return_index=False):
    """
    Calculate the probability that a source is included in the Gaia DR3
    NSS astrometric orbit catalog. The probabilistic parts of this
    calculation are in calculating ruwe, parallax error, a0_error.
    Everything else is deterministic.
    
    ra, dec [deg]
    porb : orbital period [days]
    e : eccentricity
    parallax : mas
    m1, m2: primary, secondary mass [Msun]
    g1, g2: primary, secondary apparent magnitudes [Gaia mag].
    If you want a dark companion, give it magnitude of 9999.
    n_sample : number of random samples
    mag = 'app' or 'abs'. If you do 'app', then it won't call the dust map, etc.
    If you do 'abs', it WILL call the dust map and use the parallax to place it
    at the right distance.
    return_index (bool): 
    """
    # Make sure n_sample is an integer. This is so we can get the probabilistic parts of the calculation.
    # Make all the inputs arrays the length of n_sample.
    if n_sample > 1:
        ra = ra * np.ones(n_sample)
        dec = dec * np.ones(n_sample)
        porb = porb * np.ones(n_sample)
        e = e * np.ones(n_sample)
        parallax = parallax * np.ones(n_sample)
        mass1 = mass1 * np.ones(n_sample)
        mass2 = mass2 * np.ones(n_sample)
        gmag1 = gmag1 * np.ones(n_sample)
        gmag2 = gmag2 * np.ones(n_sample)

    # For inclination, we assume isotropic in cosi.
    cosi = np.random.random(n_sample)
    i = np.arccos(cosi)
    omega = np.random.uniform(0, 2*np.pi, n_sample)
    
    # Binary magnitude
    if mag == 'app':
        gmag1_app = gmag1
        gmag2_app = gmag2
        gmag = utils.add_mags(gmag1, gmag2)

    elif mag == 'abs':
        import mwdust
        # Extinction
        # Get in lat and lon
        coords = SkyCoord(ra=ra*u.deg,
                          dec=dec*u.deg,
                          distance=(1/parallax)*u.kpc)
        glat = coords.galactic.b.value
        glon = coords.galactic.l.value
        rad = 1/parallax
        combined19 = mwdust.Combined19(filter='Gunn r')
        
        ag = combined19(glon, glat, rad)
        gmag1_app = utils.get_app_mag(gmag1, parallax, ag)
        gmag2_app = utils.get_app_mag(gmag2, parallax, ag)
        gmag = utils.add_mags(gmag1_app, gmag2_app)
        
    # Calculate photocenter size (mas)
    a_AU = utils.get_a_Kepler(porb, mass1, mass2)
    dql = utils.get_dql(mass1, mass2, gmag1_app, gmag2_app)
    a0 = dql * a_AU * parallax

    # Calculate ruwe. Broadcast to the size of n_sample.
    ruwe = utils.pred_ruwe_obs_using_sims(a0, gmag, i, e, omega, porb, ra, dec)

    nviz = utils.healpix_nviz_periods(ra, dec)

    # Figure out what passes the ruwe cut.
    ruwe_idx = np.where(ruwe > 1.4)[0]

    # Figure out what passes the acceleration cuts.
    acc9_idx = utils.remove_acc9(porb[ruwe_idx], a0[ruwe_idx], gmag[ruwe_idx])
    acc7_idx = utils.remove_acc7(porb[ruwe_idx][acc9_idx],
                                 a0[ruwe_idx][acc9_idx],
                                 gmag[ruwe_idx][acc9_idx])

    # Figure out what passes the orbital solution cut.
    parallax_error = utils.predict_single_star_parallax_error(gmag[ruwe_idx][acc9_idx][acc7_idx],
                                                              nviz[ruwe_idx][acc9_idx][acc7_idx])
    
    # Despite the name of the function, this applies to the a0 error also.
    a0_error = utils.predict_single_star_parallax_error(gmag[ruwe_idx][acc9_idx][acc7_idx],
                                                              nviz[ruwe_idx][acc9_idx][acc7_idx])
    
    parallax_error_correction = utils.calc_orb_error_rescale_candidate('parallax',
                                                                       porb[ruwe_idx][acc9_idx][acc7_idx],
                                                                       e[ruwe_idx][acc9_idx][acc7_idx],
                                                                       gmag[ruwe_idx][acc9_idx][acc7_idx])
    
    a0_error_correction = utils.calc_orb_error_rescale_candidate('a0',
                                                                 porb[ruwe_idx][acc9_idx][acc7_idx],
                                                                 e[ruwe_idx][acc9_idx][acc7_idx],
                                                                 gmag[ruwe_idx][acc9_idx][acc7_idx])
    
    parallax_sig = parallax[ruwe_idx][acc9_idx][acc7_idx]/(parallax_error * parallax_error_correction)
    a0_sig = a0[ruwe_idx][acc9_idx][acc7_idx]/(a0_error * a0_error_correction)
    
    idx = np.where((parallax_sig > 20000/porb[ruwe_idx][acc9_idx][acc7_idx]) &
                   (a0_sig > np.maximum(np.ones(len(a0_sig)) * 5, 158/np.sqrt(porb[ruwe_idx][acc9_idx][acc7_idx]))))[0]

    print('Probability of detecting [0 - 1]: {0:.4f}'.format(len(idx)/n_sample))

    if return_index:
        print(len(idx))
        return ruwe_idx, acc9_idx, acc7_idx, idx
    else:
        return len(idx)/n_sample
