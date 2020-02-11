"""
2014.08
S.Rodney

Define parameters for generating sncosmo simulations of HST SN surveys.
"""

# Collecting area of HST's 2.4-m diameter primary mirror, in cm^2
HSTMIRRORAREA = 45240.0 # cm^2

# (incomplete) dictionary of AB and Vega zero points for all filters
ZPTDICT = {
    'AB':{
        'ACSWFC':{'F435W': 25.65777,'F475W': 26.05923,'F555W': 25.7184, 'F606W': 26.49113,'F625W': 25.90667,'F775W': 25.66504,'F814W': 25.94333,'F850LP':24.84245,},
        'WFC3IR':{'F098M':25.6674, 'F127M':24.6412,'F139M':24.4793,'F153M':24.4635,
                  'F105W':26.27,'F110W':26.83,'F125W':26.25,'F140W':26.46,'F160W':25.96,'G141':25.96,},
        'WFC3UVIS':{'F218W':22.94,'F225W':24.06,'F275W':24.14,'F336W':24.64,'F350LP':26.94,'F390W':25.37,'F438W':24.83,'F475W':25.69,'F555W':25.78,'F600LP':25.85,'F606W':26.08,'F625W':25.52,'F775W':24.85,'F814W':25.09,},
        },
    'VEGA':{
        'ACSWFC':{'F435W': 25.76695,'F475W': 26.16252,'F555W': 25.72747,'F606W': 26.40598,'F625W': 25.74339,'F775W': 25.27728,'F814W': 25.51994,'F850LP':24.32305,},
        'WFC3IR':{'F105W':25.63,'F110W':26.07,'F125W':25.35,'F140W':25.39,'F160W':24.70,'G141':24.70 },
        'WFC3UVIS':{'F218W':21.25,'F225W':22.40,'F275W':22.65,'F336W':23.46,'F350LP':26.78,'F390W':25.15,'F438W':24.98,'F475W':25.79,'F555W':25.81,'F600LP':25.53,'F606W':25.99,'F625W':25.37,'F775W':24.47,'F814W':24.67,},
        }
}


# HST Background in counts/sec for a circular aperture of radius 0.2 arcsec
# TODO : allow other apertures
# TODO : fill in the BG CPS VALUES FOR UVIS AND ACS FROM THE ETC
# TODO : add options for low, med, high background levels
HSTBGCPS = {
    'WFC3IR':{'F160W':7.170,'F153M':2.325,'F139M':2.309,'F127M':2.542,'F140W':10.252,'F125W':8.392,'F110W':13.766,'F105W':8.495,'F098M':5.178},
    'WFC3UVIS':{'F350LP':8.574 },
    'ACSWFC':{  'F475W':2.164, 'F606W':4.883, 'F775W':3.302, 'F814W':4.935,'F850LP':5.069 },
}

def gethstgain( exptime ):
    """
    sncosmo simulates fluxes with units of flux density: photons/sec/cm^2
    and computes the poisson noise as just the sqrt of the flux density times
    the gain (see sncosmo.simulation.realize_lcs())

    Therefore, to get the flux and fluxerr values computed properly
    for simulated SN, we have to provide sncosmo with an effective gain
    that incorporate the exposure time and the collecting area of HST.
    """
    # TODO : include the instrument gain as well ? i.e. 2.5 for WFC3-IR
    return(  HSTMIRRORAREA * exptime )

def gethstzp( band, magsys='ab', camera=None ):
    """ 
    band     : HST bandpass name (e.g.  'f160w')
    exptime  : exposure time in seconds
    """
    import exceptions

    band = band.upper()
    magsys = magsys.upper()
    if magsys not in ['AB','VEGA']:
        raise exceptions.ValueError( "magsys must be 'AB' or 'VEGA'")
    camera = getcamera( band, camera)
    zpt = ZPTDICT[magsys][camera][band]
    return( zpt )


def getcamera( band, camera=None ):
    """ If camera is None, define the camera for the given unique band.
    If camera is a string, reformat it for use as a key to the dicts.
    :param band:
    :param camera:
    :return:
    """
    import exceptions
    magsys = 'AB'
    band = band.upper()
    if camera is None :
        if band in ZPTDICT[magsys]['ACSWFC'] and band in ZPTDICT[magsys]['WFC3UVIS'] :
            raise exceptions.ValueError( "you must specify a camera for filter %s"%band)
        elif band in ZPTDICT[magsys]['ACSWFC'] : camera = 'ACSWFC'
        elif band in ZPTDICT[magsys]['WFC3IR'] : camera = 'WFC3IR'
        elif band in ZPTDICT[magsys]['WFC3UVIS'] : camera = 'WFC3UVIS'
    else :
        camera = camera.upper().translate(None, '_-+/ ')
        if camera.startswith('ACS'): camera = 'ACSWFC'
        elif camera.endswith('IR'): camera='WFC3IR'
        elif camera.endswith('UVIS'): camera='WFC3UVIS'
    return( camera )

def gethstbgnoise( band, exptime, camera=None ):
    """
    Returns an average HST background noise in cts/sec for an observation
    in the given band with the given exposure time in sec.
    Includes all prominent sources of noise :
       earthshine, zodiacal light, read noise, dark current.
    """
    import numpy as np
    camera = getcamera( band, camera )
    band = band.upper()
    bgcps = HSTBGCPS[camera][band]
    return( np.sqrt( bgcps / exptime ) )


