import numpy as np
import utils
from astropy.coordinates import SkyCoord
import astropy.units as u

def p_nss(ra, dec, porb, e, parallax, mass1, mass2, gmag1, gmag2, n_sample = 100, mag='app'):
    """
    ra, dec [decimal deg]
    porb : orbital period [days]
    e : eccentricity
    parallax : mas
    m1, m2: primary, secondary mass [Msun]
    g1, g2: primary, secondary apparent magnitudes [Gaia mag]. If you want a dark companion, give it magnitude of 9999.
    n_sample : number of random samples
    mag = 'app' or 'abs'. If you do 'app', then it won't call the dust map, etc.
    return_index : bool
        If True, return the indices of the good solutions

    If you do 'abs', it WILL call the dust map and use the parallax to place it at the right distance.
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
        
    # For inclination, we just assume isotropic in cosi.
    # is it random 0 to 1, or -1 to 1? Also arccos only is 0 to pi?
    # Since it's |cosi| that comes into the calculations, maybe this doesn't matter.
    # CHECK THOUGH!
    # i is in RADIANS.
    cosi = np.random.random(n_sample)
    i = np.arccos(cosi)

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
        # Add some noise here so you get a little variation.
        # Arbitrary... say within 1 deg^2 (how fine is dust map?) and 10% of distance?
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
    ruwe = utils.pred_ruwe_obs(a0, gmag, i, e, ra, dec)

    nviz = utils.healpix_nviz_periods(ra, dec)
    
    # Calculate parallax_error
    parallax_error = utils.get_parallax_error_values_nss(gmag, nviz)
    
    # Calculate photocenter error
    a0_error = utils.get_a0_error_values_nss(gmag, nviz)

    # Apply cuts.
    idx = np.where((ruwe > 1.4) &
                   (a0/a0_error > np.max([np.ones(n_sample) * 5, 158/np.sqrt(porb)], axis=0)) & 
                   (parallax/parallax_error > 20000/porb) &
                   (utils.empirical_period_probability(porb)[0]))[0]

    print('Number detected : ', len(idx))
    print('Probability of detecting [0 - 1]: {0:.4f}'.format(len(idx)/n_sample))

    return len(idx)/n_sample
