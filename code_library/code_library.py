## ADD YOUR FUNCTIONS / CLASSES BELOW ##
#
#
## ETIQUETTE:
# 1. Write (extensive) documentation into the doc-string so that people understand
#    what the code is doing. This includes
#    a) explanation what the code is doing, 
#    b) Detailed description of input ("Parameters")
#    c) Detailed description of output ("Returns")
#    d) If needed, some examples.
# 2. Keep a version history ("Versions"), including date, and what was changed, and
#    who changed.
# 3. Add your name and e-mail (or multiple if many worked on the code)
#    under "Contributors". That way, we know who to contact in the case of questions
#    or bugs.
#
## EXAMPLE:
#
# def myfunction(x,y):
# '''
# This function is multiplying x and y.
# 
# PARAMETERS
# ==========
# x : float
#   x is the first number to multiply
# y : float
#   y is the second number to multiply
#
# RETURNS
# =======
# Returns the multiplication of x and y: x*y
#
# VERSIONS
# =========
# 08/17/24 (Faisst): First version of this code
# 08/18/24 (Faisst): Bug fix (code performed division instead of multiplication)
#
# CONTRIBUTORS
# ===========
# Andreas Faisst (afaisst@caltech.edu)
# '''
# ** CODE COMES HERE...**
#
# ######################

########### IMPORTS AND NECESSARY PACKAGES ###################
import os, glob
import numpy as np
import warnings

from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from astropy import cosmology


from photutils.aperture import CircularAperture

from lmfit import Model

import matplotlib.pyplot as plt

cosmo = cosmology.FlatLambdaCDM(H0=70,Om0=0.3)

################

def create_database(path_cubes, target_catalog):
    '''
    Creates a database based on the cube path. The script enters a path
    and looks up every *.fits image. Then creates a database with several
    columns (detailed below).

    PARAMETERS
    ==========
    path_cubes : str
        Path where cubes are located. These need to come from Seiji Fujimoto's
        reductions, else the code doesn't know what to do.
    target_catalog : astropy.table
        Target catalog of the ALPINE-CRISTAL JWST program (can be downloaded here)

    RETURNS
    =======
    Returns a database with the following columns:
    - name: name of the sources as in the cube filename (e.g., DC_*)
    - realname: real name of the source (e.g., DEIMOS_COSMOS_*)
    - grating: grating of observations
    - filename: full file name to cube.
    - redshift: redshift of the source

    VERSIONS
    ========
    08/17/24 (Faisst): First version

    CONTRIBUTORS
    ============
    Andreas Faisst (afaisst@caltech.edu)
    '''
    
    files = glob.glob(os.path.join(path_cubes , "*.fits"))
    data = dict()
    for file in files:
    
        # get data
        name = file.split("/")[-1].split("_")[0]
        grating = file.split("/")[-1].split("_")[1]
        filename = file.split("/")[-1]
    
        # get real name
        if "VC-" in name:
            real_name = "vuds_cosmos_{}".format(name.split("-")[-1])
        elif "DC-" in name:
            real_name = "DEIMOS_COSMOS_{}".format(name.split("-")[-1])
        else:
            print("Something is wrong")
    
        # get some info from parent catalog
        print(real_name)
        idx = np.where(target_catalog["name"] == name)[0]
        if len(idx) == 0:
            print("Source not found in parent catalog")
        z = target_catalog["zCII"][idx[0]]
        
        
        # add to dictionary
        if name not in data.keys():
            data[name] = Table([[name] , [real_name] , [grating] , [filename] , [z]],
                               names = ["name","realname","grating","filename","redshift"],
                               dtype =[str,str,str,str,float])
        else:
            data[name].add_row([[name] , [real_name] , [grating] , [filename] , [z]])
        
    ## Create table
    data_table = Table()
    for ii,key in enumerate(list(data.keys())):
        if ii == 0:
            data_table = data[key]
        else:
            for rr in data[key]:
                data_table.add_row(rr)
    
    return(data_table)

def get_emlinmap_ALF(cube, cubeerr , wave , lam_min , lam_max):
    '''
    Creates an emission line map (in units of 1e-17 erg/s/cm2) by integrating
    wavelengh axis of cube between lam_min and lam_max.
    
    PARAMETERS
    ===========
    cube : numpy.ndarray [flux in uJy]
        3 dimensional cube (flux dimension: uJy)
    cubeerr : numpy.ndarray [flux in uJy]
        3 dimensional error cube (flux dimension: uJy) **ERROR ON EMISSION LINE MAP IS NOT
        FULLY IMPLEMENTED YET**
    wave : numpy.ndarray [Angstrom, observed frame]
        Observed wavelength array corresponding to each wavelength slice in cube
    lam_min : float [Angstrom, observed frame]
        Minimum wavelength in observed frame
    lam_max : float [Angstrom, observed frame]
        Maximum wavelength in observed frame


    RETURNS
    =======
    cube_collapsed : numpy.ndarray [flux in erg/s/cm2]
        Collapsed cube (= Emission line map)
    cubeerr_collapsed : numpy.ndarray [flux in erg/s/cm2]
        Collapsed error cube (= Uncertainty of emission line map)
    

    VERSIONS
    ========
    08/18/24 (Faisst): First version


    CONTRIBUTORS
    =============
    - Andreas Faisst (afaisst@caltech.edu)
    '''

    ## get lambda range
    sel_lam = np.where( (wave > lam_min) & (wave < lam_max) )[0]

    ## Get nu and dnu
    dlam = np.median(np.diff(wave)[sel_lam]) # angstroms (observed)
    dnu = np.abs( -3e18 / (np.median([lam_min,lam_max])**2) * dlam ) # nu = c / lam -> dnu = -c / lam^2 * dlam (note: convert to observed frame)

    ## Create collapsed cube (could be better?)
    cb = cube[sel_lam , : , :] # uJy
    cb = cb * 1e-6 * 1e-23 # uJy -> Jy -> erg/s/cm2/Hz
    cb = dnu * cb # erg/s/cm2/Hz -> erg/s/cm2
    cube_collapsed = np.nansum(cb,axis=0) # in erg/s/cm2
    cube_collapsed = cube_collapsed / 1e-17 # in 1e-17 erg/s/cm2
    cube_collapsed[cube_collapsed == 0] = np.nan

    cubeerr_collapsed = np.nansum(cubeerr[sel_lam , : , :],axis=0)
    cubeerr_collapsed[cubeerr_collapsed == 0] = np.nan

    
    return(cube_collapsed , cubeerr_collapsed)


def load_cube_from_file(cube_file_name):
    '''
    Loads a cube and calculates some essential things.

    PARAMETERS
    ===========
    cube_file_name : str
        File name of cube (full path)

    RETURNS
    =======
    - cube : cube flux in uJy (3D numpy.ndarray) [uJy]
    - cubeerr : Error of cube in uJy (rescaled) [uJy]
    - cubeflags : any flags in cube ("DQ" extension from pipeline)
    - wave : wavelength in angstroms for each slice in 3D cube [Angstrom, observed]
    - hdr : Science header of cube
    - pixscale : spatial pixel scale [arcsec/px]

    VERSIONS
    ========
    08/18/24 (Faisst): First version

    CONTRIBUTORS
    ============
    - Andreas Faisst (afaisst@caltech.edu)
    
    '''
    
    ## Load cube
    with fits.open(cube_file_name) as hdul:
        cube = hdul['BKG_AND_STR_SUBTRACTED'].data
        cubeflags = hdul['DQ'].data
        hdr = hdul['SCI'].header
        cubeerr = hdul['RESCALED_ERR'].data
        
    wave = (hdr["CRVAL3"] + (np.arange(1,cube.shape[0]+1)-1) * hdr["CDELT3"])*1e4 # Angstrom

    pixscale = hdr["CDELT1"]*3600 # arcsec/px

    cube = cube * hdr["PIXAR_SR"] * 1e6 * 1e6 # MJy/sr -> MJy -> Jy -> uJy
    cubeerr = cubeerr * hdr["PIXAR_SR"] * 1e6 * 1e6 # MJy/sr -> MJy -> Jy -> uJy

    return(cube, cubeerr, cubeflags, wave, hdr, pixscale)




def extract_spectrum(source_name, mask, grating, database):
    '''
    CHANGE THIS SO I CAN GIVE IT A CUBE AND A MASK AND IT CALCULATES THE SPECTRUM.
    
    '''


    ## Load cube
    cube, cubeerr, cubeflags, wave, hdr, pixscale = load_cube(name = source_name,
                        grating = grating,
                         database = database,
                         path_cubes = "../data_reduction/v2.0_20240601_pix0.1_Seiji/fits/"
                        )

    ## Calibrate cube
    # Cube is in MJy/sr. Update to uJy
    cube = cube * hdr["PIXAR_SR"] * 1e6 * 1e6 # MJy/sr -> MJy -> Jy -> uJy
    cubeerr = cubeerr * hdr["PIXAR_SR"] * 1e6 * 1e6 # MJy/sr -> MJy -> Jy -> uJy
    
    ## Extract spectrum
    spec = []
    err = []

    for ii in range(cube.shape[0]):  

        fluxtot = np.nansum(cube[ii,:,:]*mask)
        fluxerr = np.nansum(cubeerr[ii,:,:]*mask) / np.sqrt( len(np.where(mask == 1)[0]) )
        #tmp = cubeerr[ii,:,:]*mask
        #tmp[tmp == 0] = np.nan
        #fluxerr = np.nanmedian(tmp)

        spec.append(fluxtot)
        err.append(fluxerr)
    
    spec_table = Table([wave,spec,err] , names=["wave","flux","err"] , dtype=[float,float,float])
    #spec_table["wave_rest"] = spec_table["wave"] / (1+z)

    return(spec_table)


def create_aperture_mask(cube , radius, offset, centering, method):
    '''
    Creates a circular mask for a given cube size. The mask is positioned
    in the center by default (with a given center_offset) if centering is
    set to True.

    PARAMETERS
    ==========
    cube : numpy.ndarray (3D)
        3D cube.
    radius : float
        Radius in pixels.
    offset : list[2]
        If centering = True, this is used as offset from center position. If
        centering = False, this is the position of the mask center. For example [1,3]
    centering : bool
        If set to True, mask is centered and can be adjusted with `offset`.
    method : str
        The `method` keyword for creating the mask ("exact", "center", etc, 
        see photutils.CircularAperture for more information).
    
    RETURNS
    ========
    Returns a mask (numpy.ndarray 2D) with the same shape as the
    cube. The masked area is 1, the non-masked area is 0.

    VERSIONS
    ========
    08/19/24 (Faisst): First version
    
    CONTRIBUTORS
    ============
    - Andreas Faisst (afaisst@caltech.edu)
    
    '''

    if centering:
        position = (cube.shape[2]//2+offset[0], cube.shape[1]//2+offset[1])
    else:
        position = tuple(offset)
    aperture = CircularAperture(position, r=radius)
    mask = aperture.to_mask(method=method).to_image((cube.shape[1],cube.shape[2]))

    return(mask)


def extract_spectrum_from_cube(cube, cubeerr, wave , mask):
    '''
    Extracts a spectrum from a cube given a mask.

    PARAMETERS
    ==========
    cube : numpy.ndarray [flux in uJy]
        3 dimensional cube (flux dimension: uJy)
    cubeerr : numpy.ndarray [flux in uJy]
        3 dimensional error cube (flux dimension: uJy) **ERROR ON EMISSION LINE MAP IS NOT
        FULLY IMPLEMENTED YET**
    wave : numpy.ndarray [Angstrom]
        Observed wavelength array corresponding to each wavelength slice in cube
    mask : numpy.ndarray
        Mask with the same spatial shape as cube and cubeerr. The cube will be multiplied
        with the mask. Masked values have to be 0, non-masked values > 0.

    RETURNS
    =======
    A astropy table with columns ["wave", "flux", "err"] containing the wavelength (angstroms),
    the measured flux (uJy), and associated error (uJy).


    VERSIONS
    =========
    08/19/24 (Faisst): First version

    CONTRIBUTORS
    ============
    - Andreas Faisst (afaisst@caltech.edu)
    
    
    '''

    
    ## Extract spectrum
    spec = []
    err = []

    for ii in range(cube.shape[0]):  

        fluxtot = np.nansum(cube[ii,:,:]*mask)
        fluxerr = np.nansum(cubeerr[ii,:,:]*mask) / np.sqrt( len(np.where(mask == 1)[0]) )

        spec.append(fluxtot)
        err.append(fluxerr)
    
    spec_table = Table([wave,spec,err] , names=["wave","flux","err"] , dtype=[float,float,float])

    return(spec_table)


def gauss1(x, amp, lam, sigma):
    '''
    Creates a simple gaussian function.

    PARAMETERS
    ==========
    x : numpy.ndarray
        x-values
    amp : float
        Total integrated flux of gaussian
    lam : float
        Center wavelength
    sigma : float
        Sigma of gaussian

    RETURNS
    ========
    y = amp * exp( -(x-lam)**2/(2*sigma**2) ) / sqrt(2*pi*sigma**2)

    VERSIONS
    ========
    08/19/24 (Faisst): First version

    CONTRIBUTORS
    ============
    - Andreas Faisst (afaisst@caltech.edu)
        
    '''

    xx = np.linspace(np.nanmin(x) , np.nanmax(x) , int(len(x)*5))
    ff = amp * np.exp( (-(xx-lam)**2) / (2*sigma**2) ) / np.sqrt(2*np.pi*sigma**2)
    f = np.interp(x , xx , ff)
    return(f)


def fit_line_single_gauss(wave , flux, fluxerr, redshift, center_lam, checkplot):
    '''
    Fits a single gaussian to a line using the LMFIT package.
    
    PARAMETERS
    ==========
    wave : numpy.ndarray [Angstrom, observed]
        Wavelength of spectrum (observed frame in Angstrom)
    flux : numpy.ndarray [uJy]
        Flux of spectrum in uJy (observed frame)
    fluxerr : numpy.ndarray [uJy]
        Flux error of spectrum in uJy (observed frame)
    redshift : float
        Redshift of source (needed to get correct rest-frame flux)
    center_lam : float [Angstrom , rest-frame]
        Center redshift guess of line (in rest-frame in Angstrom)
    checkplot : bool
        Set to true if a check plot (data + model) should be shown.

    RETURNS
    =======
    A dictionary with:
    - fit: full output of fit results from LM fit
    - lam: best-fit center wavelength (rest-frame , Angstrom)
    - amp: best-fit total line flux (erg/s/cm2)
    - sigma: best-fit sigma width (rest-frame , Angstrom)
    - X: wavelength (X) that is used for fit in rest-frame Angstroms
    - Y: Flux (Y) that is used for fit in erg/s/cm2/A
    - E: Error on Flux (Y) that is use for fit in erg/s/cm2/A
    - Ybest: best-fit Y.

    If checkplot = True, then a figure with observations + model is shown.

    VERSIONS
    ========
    08/19/24 (Faisst): First version

    CONTRIBUTORS
    ============
    - Andreas Faisst (afaisst@caltech.edu)
    
    '''

    X = wave / (1+redshift) # wavelength in A and rest-frame
    Y = flux # flux in uJy
    E = fluxerr # flux error in uJy
    
    Y = Y/1e6*1e-23 / ((X*(1 + redshift))**2 / 3e18) / 1e-17 # uJy -> erg/s/cm2/A (flam) # NOTE: we have to use lam and not lam_rest!
    Y = Y - np.nanmedian(Y)
    E = E/1e6*1e-23 / ((X*(1 + redshift))**2 / 3e18) / 1e-17 # uJy -> erg/s/cm2/A (flam) # NOTE: we have to use lam and not lam_rest!
    SN = Y / E # S/N ratio used for fit (do this before continuum subtraction)
    

    ## Create model
    mod = Model(gauss1)


    ## Update parameters
    pars = mod.make_params()
    pars.add(name = "amp", value = 1, vary = True, min = 0)
    pars.add(name = "lam", value = center_lam, vary = True, min = center_lam-20 , max = center_lam+20)
    pars.add(name = "sigma", value = 5, vary = True, min = 0, max = 20)

    ## Fit
    fitresult = mod.fit(Y, pars, x=X, nan_policy="omit")

    ## Get best fit
    Ybest = gauss1(x = X ,
                            amp = fitresult.params["amp"].value,
                            lam = fitresult.params["lam"].value,
                            sigma = fitresult.params["sigma"].value
                          )
    ## Create output
    output = {
                "fit":fitresult,
                "lam":fitresult.params["lam"].value,
                "amp":fitresult.params["amp"].value,
                "sigma":fitresult.params["sigma"].value,
                "X":X,
                "Y":Y,
                "E":E,
                "Ybest":Ybest
             }

    ## Make check plot if requested
    if checkplot:
        fig = plt.figure(figsize=(6,5))
        ax1 = fig.add_subplot(1,1,1)

        ax1.plot(X , Y , "o" , label="Observations")
        ax1.step(X , Ybest , "-", where="mid", label="fit")

        ax1.legend(fontsize=11)
        ax1.set_xlabel(r"$\lambda_{\rm rest}$ [$\AA$]")
        ax1.set_ylabel(r"Flux [erg/s/cm$^2$/$\AA$]")
        
        plt.show()
    
    return(output)


def create_velocity_map(cube, cubeerr, wave, lam_min , lam_max, redshift, radius, line_lam, sn_cut):
    '''
    Creates a velocity map of a given emission line. The velocity field
    is simply created by fitting for each pixels (within apertures) the 
    central wavelength of a line using a Gaussian. It does not measure
    multiple lines. Therefore mostly effective for isolated lines and 
    unresolved lines.

    PARAMETERS
    ===========
    cube : numpy.ndarray [flux in uJy]
        3 dimensional cube (flux dimension: uJy)
    cubeerr : numpy.ndarray [flux in uJy]
        3 dimensional error cube (flux dimension: uJy) **ERROR ON EMISSION LINE MAP IS NOT
        FULLY IMPLEMENTED YET**
    wave : numpy.ndarray [Angstrom]
        Observed wavelength array corresponding to each wavelength slice in cube
    lam_min : float [Angstrom]
        Minimum wavelength in observed frame
    lam_max : float [Angstrom]
        Maximum wavelength in observed frame
    redshift : float
        Redshift of the source
    radius : float
        Radius of spectrum extraction in pixels
    line_lam : float [Angstrom, rest-frame]
        Rest-frame wavelength in A of line (used for initial guess for line fit)
    sn_cut : float
        S/N cut for creating the mask on which to compute the velocity field


    RETURNS
    =======
    - Velocity map (same shape as cube) in units of delta KM/S from the line center
    - Emission line map (used to create mask)
    - Mask on which velocity field is computed.

    VERSIONS
    ========
    08/19/24 (Faisst): First version

    CONTRIBUTORS
    ============
    - Andreas Faisst (afaisst@caltech.edu)
    
    
    '''

    ## Make quick emission line map so we can get the boundary for
    ## the velocity map creation.
    cb, _ = get_emlinmap_ALF(cube , cubeerr, wave , lam_min , lam_max)


    ## Create mask from emission line map on which we are evaluating
    ## the velocity map
    mean, median, stddev = sigma_clipped_stats(cb)
    cb_norm = cb / stddev
    cb_norm_mask = cb_norm * 0
    cb_norm_mask[cb_norm > sn_cut] = 1

    ## Create Grid for measuring velocity differences.
    xx = np.arange(0,cube.shape[2])
    yy = np.arange(0,cube.shape[1])
    xy_orig = np.asarray(np.meshgrid(xx,yy)).reshape(2,len(xx)*len(yy)).T
    
    ## Remove grid points that are masked
    tmp = np.asarray( [cb_norm_mask[int(gp[1]),int(gp[0])] for gp in xy_orig] )
    selgb = np.where(tmp == 1)[0]
    xy = xy_orig[selgb]

    ## Fit line for each grid point
    ## Note: turn off warnings for a bit.
    
    warnings.filterwarnings('ignore')
    tab = Table(names=["x","y","lam"], dtype=[float,float,float])
    
    print("Points to calculate: {}".format(len(xy)))
    for gp in xy:
        
        ## Create mask
        mask = create_aperture_mask(cube = cube, radius = radius , offset = tuple(gp), centering = False, method = "exact")
    
        ## Extract spectrum
        spec = extract_spectrum_from_cube(cube , cubeerr, wave , mask)
        spec["wave_rest"] = spec["wave"] / (1+redshift)
    
        ## fit spectrum in a given wavelength range
        #line_range = np.abs(lam_max - lam_min)/2 # angstroms
        #sel_lam = np.where( (spec["wave_rest"] > (line-line_range/2)) & (spec["wave_rest"] < (line+line_range/2)))[0]
        sel_lam = np.where( (spec["wave"] > lam_min) & (spec["wave"] < lam_max) )[0] # selection in observed frame
        fitoutput = fit_line_single_gauss(wave = spec["wave"][sel_lam] , 
                                          flux = spec["flux"][sel_lam] , 
                                          fluxerr = spec["err"][sel_lam] , 
                                          redshift = redshift , 
                                          center_lam = line_lam,
                                          checkplot = False
                                         )
        tab.add_row( [gp[0],gp[1],fitoutput["lam"]] )
        
    warnings.filterwarnings('default')
    
    
    ## Add additions:
    tab["dlam"] = tab["lam"] - line_lam
    # add velocity offset

    ## Put all together
    velmap_array = np.zeros(len(xy_orig))
    velmap_array[selgb] = tab["dlam"]
    velmap = velmap_array.reshape((len(yy),len(xx)))
    velmap[velmap == 0] = np.nan

    ## Convert to Delta KM/S
    # dv/c = dlam/lam
    # -> dv = dlam/lam * c
    velmap_kms = velmap / line_lam * 3e5 # km/s

    ## Test
    #plt.imshow(velmap_kms, origin="lower")
    #plt.contour(cb_norm_mask , levels=[0,1],origin="lower" , colors="black", alpha=0.2)
    #plt.show()
    
    return(velmap_kms, cb, mask)
