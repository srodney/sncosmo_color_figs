__author__ = 'rodney'

from astropy import cosmology
from astropy.io import ascii
import numpy as np
from scipy.stats.stats import chisqprob
import sncosmo

cosmo=cosmology.FlatLambdaCDM( name="WMAP9", H0=70.0, Om0=0.3 )
dm = lambda z : cosmo.distmod( z ).value
MBmodel = -19.223 # AB mags ( = -19.12 Vega )



def dofit( datfile='nebra_bestphot.dat', z=2.00, dz=0.02,
           t0=57575., dt0=20.0, x1=None, c=None,
           model='Ia', noUV=True, debug=False) :
    # TODO : read in the redshift, etc from the header.

    from .colorcolorfig import  SubClassDict_SNANA
    # read in the obs data
    sn = ascii.read( datfile, format='commented_header', header_start=-1, data_start=0 )

    if model == 'Ia' :
        # define SALT2 models and set initial guesses for z and t0
        if noUV :
            salt2ex = sncosmo.Model( source='salt2')
        else :
            salt2ex = sncosmo.Model( source='salt2-extended')
        salt2ex.source.set_peakmag( 0., 'bessellb', 'ab' )
        x0_AB0 = salt2ex.get('x0')
        salt2ex.set( z=z, t0=t0, x1=0.1, c=-0.2 )
        # salt2ex.set( z=1.33, t0=56814.6, hostebv=0.05, hostr_v=3.1 )

        # Do a bounded fit :
        # salt2res, salt2fit = sncosmo.fit_lc( sn, salt2, ['z','t0','x0','x1','c'], bounds={'z':(1.28,1.37),'t0':(56804,56824)} )
        varlist = varlist = ['z','t0','x0']
        bounds={  'z':(z-dz,z+dz), 't0':(t0-dt0,t0+dt0) }
        if x1 is not None:
            salt2ex.set( x1=x1 )
            bounds['x1'] = (x1-1e-6,x1+1e-6)
            varlist.append( 'x1' )
        else :
            bounds['x1'] = (-5,5)
            varlist.append( 'x1' )
        if c is not None:
            salt2ex.set( c=c )
        else :
            bounds['c'] = (-0.5,3.0)
            varlist.append( 'c' )

        res, fit = sncosmo.fit_lc( sn, salt2ex, varlist, bounds )

        x0 = fit.get( 'x0' )
        z = fit.get( 'z' )
        mB = -2.5*np.log10(  x0 / x0_AB0 )
        distmod = mB - MBmodel
        deltamuLCDM = distmod - dm(z)
        print( "mB = %.2f"%mB )
        print( "dist.mod. = %.2f"%distmod)
        print( "Delta.mu_LCDM = %.2f"%deltamuLCDM)

        chi2 = res.chisq
        ndof = res.ndof
        pval = chisqprob( chi2, ndof )

        if ndof>0:
            print( "chi2/dof= %.3f"% (chi2/float(ndof) ) )
            print( "p-value = %.3f"% pval )
        else :
            print( "chi2/dof= %.3f/%i"%( chi2, ndof) )
            print( "p-value = %.3f"% pval )

        print( "z = %.3f"% fit.get('z') )
        print( "t0 = %.3f"% fit.get('t0') )
        print( "x0 = %.3e"% fit.get('x0') )
        print( "x1 = %.3f"% fit.get('x1') )
        print( "c = %.3f"% fit.get('c') )

    elif model.lower() in ['cc','ib','ic','ii','ibc','iip','iin']:
        # remove the blue filters from the sn data
        bandlist = sn['filter'].data
        igood = np.array( [ band.lower().startswith('f1') for band in bandlist ] )
        sn = sn.copy()[igood]

        # define a host-galaxy dust model
        dust = sncosmo.CCM89Dust( )
        version = '1.0'

        if model.lower()=='cc' : classlist = ['Ib','Ic','IIP','IIn']
        elif model.lower()=='ii' : classlist = ['IIP','IIn']
        elif model.lower()=='ibc' : classlist = ['Ibc']
        else : classlist = [model]

        # find the best-fit from each CC sub-class
        chi2list, reslist, fitlist  = [],[],[]
        for snclass in classlist :
            for modname in SubClassDict_SNANA[snclass.lower()] :
                Av = 0.2
                modkey = ( sncosmo.Source, modname, version )
                if modkey not in sncosmo.registry._loaders : continue
                ccmodel = sncosmo.Model( source=modname, effects=[dust],
                                         effect_names=['host'], effect_frames=['rest'])
                ccmodel.set( z=z, t0=t0, hostr_v=3.1, hostebv=Av/3.1 )
                # Do a bounded fit :
                res, fit  = sncosmo.fit_lc(
                    sn, ccmodel, ['z','t0','amplitude','hostebv' ], debug=debug,
                    bounds={'z':(z-dz,z+dz),'t0':(t0-dt0,t0+dt0),
                            'hostebv':(0.0,1.0) } )

                chi2 = res.chisq
                ndof = res.ndof
                pval = chisqprob( chi2, ndof )

                print( "%s  chi2/dof= %.3f  p=%.3f"%(modname, chi2/float(ndof), pval  ) )
                chi2list.append( chi2/float(ndof) )
                reslist.append( res )
                fitlist.append( fit )
        ichi2min = np.argmin( chi2list )
        res, fit = reslist[ichi2min], fitlist[ichi2min]
    else : # 'nugent-sn91bg'
        # remove the blue filters from the sn data
        bandlist = sn['filter'].data
        igood = np.array( [ band.startswith('f1') for band in bandlist ] )
        sn = sn.copy()[igood]

        # define a host-galaxy dust model
        dust = sncosmo.CCM89Dust( )
        version = '1.0'

        Av = 0.2
        altmodel = sncosmo.Model( source=model, effects=[dust],
                                 effect_names=['host'], effect_frames=['rest'])
        altmodel.set( z=z, t0=t0, hostr_v=3.1, hostebv=Av/3.1 )
        # Do a bounded fit :
        res, fit  = sncosmo.fit_lc(
            sn, altmodel, ['z','t0','amplitude','hostebv' ], debug=debug,
            bounds={'z':(z-dz,z+dz),'t0':(t0-dt0,t0+dt0),
                    'hostebv':(0.0,1.0) } )

        chi2 = res.chisq
        ndof = res.ndof
        pval = chisqprob( chi2, ndof )

        print( "%s  chi2/dof= %.3f  p=%.3f"%(model, chi2/float(ndof), pval  ) )

    return( sn, fit, res )

def sncosmoplot( sn, fit, res ):
    from pytools import plotsetup
    from matplotlib import rcParams
    plotsetup.fullpaperfig()
    rcParams['text.usetex'] = False
    sncosmo.plot_lc( sn, model=fit, errors=res.errors )


def keck_proposal_plot(sn, fit, res):
    from pytools import plotsetup
    from matplotlib import rcParams, pyplot as pl
    plotsetup.fullpaperfig()
    pl.clf()
    tplot = np.arange(57525, 57720, 1)
    for bandname, color, offset in zip(
            ['f105w', 'f125w', 'f140w', 'f160w'],
            ['b','g','darkorange','k'],
            [+1.5, +1, +0.5, 0]):
        mag = fit.bandmag(bandname, 'ab', tplot)
        pl.plot(tplot, mag+offset, color=color)
        snmjd = sn['MJD']
        snmag = sn['MAG']
        snmagerr = sn['MAGERR']
        snband = sn['FILTER']
        isn = np.where(snband==bandname.upper())

        pl.errorbar(snmjd[isn], snmag[isn]+offset, snmagerr[isn], marker='o',
                    ls=' ', ms=8, capsize=0.1, color=color)

    ax = pl.gca()
    ax.invert_yaxis()
    ax.set_ylim(27.4, 23.8)
    ax.set_xlim(57540, 57690)
    ax.text(57585, 27.0, 'Y+1.5 (F105W)', ha='right', color='b', fontsize='large', va='top')
    ax.text(57597, 25.8, 'J+1 (F125W)', ha='right', color='g', fontsize='large', va='bottom')
    ax.text(57650, 26.5, 'JH+0.5 (F140W)', ha='left', color='darkorange', fontsize='large', va='top')
    ax.text(57642, 25.4, 'H (F160W)', ha='left', color='k', fontsize='large', va='bottom')
    ax.set_xlabel('Observed Time (MJD)')
    ax.set_ylabel('AB magnitude')

    fig = pl.gcf()
    fig.subplots_adjust(left=0.12, bottom=0.14, right=0.97, top=0.97)
    pl.draw()





