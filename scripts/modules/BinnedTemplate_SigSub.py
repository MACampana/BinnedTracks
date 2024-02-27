#Imports
import os, sys, gc
os.environ['OMP_NUM_THREADS'] = '3' #limits number of threads used (like by hp.smoothing)
import numpy as np
import scipy as sp 
import astropy as ap #Could only import SkyCoords or whatever else I use
import iminuit
import healpy as hp
import histlite as hl
from csky.utils import ensure_dir 
import weakref

from glob import glob

class BinnedTemplateAnalysis:
    """For conducting binned calculations using maximum likelihood statistical methods. 
    For binned sky map of IceCube event data, specifically for template analyses.
    
    (Now with energy!)
    
    
    """
    def __init__(self, data, sig, grl, template, is_binned=False, savedir=None, name='BinnedTemplateAnalysis', 
                 gamma=2.7, ebins=None,
                 nside=128, min_dec_deg=-80, max_dec_deg=80, qtot=True,
                 verbose=False, force=False, sigsub=True):
        """BinnedTemplateAllSky constructor
        
        Args:
            data: Path to numpy array(s) (directory or single file) containing  dtype=[('run', '<i8'), 
                                                 ('event', '<i8'),
                                                 ('subevent', '<i8'),
                                                 ('ra', '<f8'),
                                                 ('dec', '<f8'),
                                                 ('azi', '<f8'),
                                                 ('zen', '<f8'),
                                                 ('time', '<f8'),
                                                 ('logE', '<f8'),
                                                 ('angErr', '<f8')])
                                                 
                OR: Path to numpy array of binned data (in this case, set is_binned=True)
                                     
            sig: Path to numpy array containing  dtype = [('run', int), 
                                                 ('event', int), 
                                                 ('subevent', int),
                                                 ('ra', float), 
                                                 ('dec', float),
                                                 ('true_ra', float), 
                                                 ('true_dec', float),
                                                 ('azi', float), 
                                                 ('zen', float), 
                                                 ('time', float),
                                                 ('logE', float), 
                                                 ('angErr', 'float'),
                                                 ('true_angErr', float), 
                                                 ('oneweight', float),     
                                                 ('true_energy', float)]
            
            grl: Path to numpy array with GRL runs

            template: path to template array
                                     
            is_binned: boolean, True if argument data is an array of binned_data, otherwise False (data will be binned)
            
            savedir: path to directory to save binned data. Default: None (don't save)
            
            name: unique name to identify analysis (used in file names when saving)
                                                            
            gamma: spectral index for detector acceptance and injections. 
                    
            ebins: Choice of log10(energy) bins. If None, use default (hardcoded) bins. If a single integer,
                use `ebins` evenly spaced bins in log10(energy). If a list of numbers, assumed to be 
                log10(energy) bin edges. If `is_binned` is True, `ebins` will be determined from the loaded data.

            nside: integer for healpy nside (Default: 128)
                        
            min_/max_dec_deg: min and max declination in degrees for likelihood calculations.
            
            qtot: If True, use total charge instead of reconstructed energy for "energy" binning of events
            
            verbose: True for lots of output (Default: False)
            
            force: boolean, if `is_binned` is True and the loaded data does not match the provided `nside`, set this
                to True to resize the loaded, binned data to the give nside. If False, will raise ValueError (Default: False)

            sigsub: boolean, whether to use the signal subtraction likelihood/TS
            
        """
        
        print('Setting up:')
        
        self.name = name
        self.savedir = savedir
        ensure_dir(savedir)
        
        self.verbose = verbose
        
        self.gamma = gamma
        self.nside = nside
        self.min_dec_deg = min_dec_deg
        self.max_dec_deg = max_dec_deg
        self.sigsub = sigsub
        
        #sindec band edges: default here taken from PSTracks in csky
        #self.sindec_bins = np.unique(np.concatenate([
        #                     np.linspace(-1, -0.93, 4 + 1),
        #                     np.linspace(-0.93, -0.3, 10 + 1),
        #                     np.linspace(-0.3, 0.05, 9 + 1),
        #                     np.linspace(0.05, 1, 18 + 1) ]) )
        
        #Load GRL which should be in format of an array of (good) run numbers
        self.grl = np.load(grl)
        
        #Load signal events and calculate relative weights
        self.sig_evs = np.load(sig)
        print(f'Loaded signal array <-- {sig}')        
        self.sig_relweights = self.sig_evs['oneweight'] * self.sig_evs['true_energy']**(-self.gamma)
        
        #If using total charge as proxy for energy...
        if qtot:
            if ('qtot' not in self.sig_evs.dtype.names):
                raise ValueError('QTot option is True, but signal array does not contain `qtot` field!')
            #If sig array contains qtot with DeepCore, use with DeepCore
            if 'qtot_wdc' in self.sig_evs.dtype.names:
                q_name = 'qtot_wdc'
            else:
                q_name = 'qtot'
            #Set the name for "log Energy" bins to logQtot
            self.logE_name = 'logQtot'
            
            #Add log10 of QTot to the sig array for consistency with logE by constructing new array
            new_dtype = np.dtype(self.sig_evs.dtype.descr + [(self.logE_name, '<f8')])
            new_arr = np.empty(self.sig_evs.shape, dtype=new_dtype)
            for n in self.sig_evs.dtype.names:
                new_arr[n] = self.sig_evs[n]
            new_arr[self.logE_name] = np.log10(self.sig_evs[q_name])

            #Reassign the sig array to be the new array (includes logQtot field) and clear up the temporary array
            self.sig_evs = new_arr.copy()            
            del new_arr
            
        #Otherwise, if just using energy and not qtot...
        else:
            self.logE_name = 'logE'
            
        #ebins is really logEbins. If None, assign some default value which depends on if using energy or qtot.
        #Should likely avoid default bins, or determine a good set of default bins.
        if (ebins is None) or ((len(ebins)==1) and (ebins[0]==0)):
            if not is_binned:
                print('Using default EBins.')
            #Default qtot bins were determined to have roughly equal number of events in 100 bins
            if qtot:
                self.logE_bins = np.array([1.5051, 1.5493, 1.5761, 1.5963, 1.6125, 1.6261, 1.6382, 1.6493,
                                           1.6592, 1.6686, 1.6774, 1.6855, 1.6935, 1.7011, 1.7084, 1.7154,
                                           1.7224, 1.729 , 1.7356, 1.7419, 1.7482, 1.7543, 1.7604, 1.7664,
                                           1.7723, 1.7782, 1.7839, 1.7896, 1.7954, 1.8009, 1.8065, 1.8119,
                                           1.8176, 1.823 , 1.8285, 1.8341, 1.8395, 1.8451, 1.8506, 1.8561,
                                           1.8615, 1.8672, 1.8727, 1.8782, 1.8839, 1.8896, 1.8953, 1.9012,
                                           1.907 , 1.9128, 1.9187, 1.9248, 1.9308, 1.937 , 1.9434, 1.9498,
                                           1.9563, 1.963 , 1.9696, 1.9768, 1.9839, 1.9912, 1.9988, 2.0065,
                                           2.0144, 2.0226, 2.0311, 2.04  , 2.0492, 2.0589, 2.0689, 2.0795,
                                           2.0906, 2.1023, 2.1149, 2.1283, 2.1427, 2.1581, 2.1749, 2.1931,
                                           2.2131, 2.2353, 2.2596, 2.287 , 2.3175, 2.3517, 2.3896, 2.4321,
                                           2.4794, 2.5323, 2.592 , 2.6584, 2.7228, 2.7804, 2.8347, 2.8891,
                                           2.9452, 3.0136, 3.1295])
            else:
                self.logE_bins = np.array([1.,2.,3.,4.,5.,6.,7.])

            self.n_logE_bins = len(self.logE_bins)+1
        #If ebins is 1
        elif (len(ebins)==1) and (ebins[0]==1):
            self.logE_bins = np.array([0.0, np.inf])
            self.n_logE_bins = 1
        #If ebins argument is one number, make evenly space bins
        elif (len(ebins) == 1):
            ebins = ebins[0]
            if not is_binned:
                print(f'Using {ebins} EBins.')
            
            if qtot:
                self.logE_bins = np.linspace(1.5, 3.5, ebins-1)
            else:
                self.logE_bins = np.linspace(1, 7, ebins-1)

            self.n_logE_bins = ebins
        #If ebins is more than one number, use those as the bin edges
        else: #len(ebins) > 1
            if not is_binned:
                print('Using user-defined EBins.')
            self.logE_bins = np.array(ebins) 
            self.n_logE_bins = len(self.logE_bins)+1

        #Load experimental data (and bin it if it is not already binned)
        if is_binned:
            #Binned data will be dictionary with {'logE_bins': array, 'binned_data': ndarray}
            binned_dict = np.load(data, allow_pickle=True)
            
            #Replace ebins with those from loaded binned data
            print('Using loaded EBins.')
            self.logE_bins = binned_dict.item()['logE_bins']
            #Get binned data from dictionary
            self.binned_data = binned_dict.item()['binned_data'] 
            self.n_logE_bins = len(self.binned_data)

            #Check to see if binned_data nside matches that provided in argument, if not...
            if self.binned_data.shape[1] != hp.nside2npix(self.nside):
                #...and if force is True, resize the binned data.
                if force:
                    print(f"*** Re-sizing binned data with Nside={hp.npix2nside(self.binned_data.shape[1])} to {self.nside}! ***")
                    self.binned_data = hp.ud_grade(self.binned_data, self.nside)
                #Otherwise, if force is False, raise an error
                else:
                    raise ValueError(f"Nside of loaded binned data is {hp.npix2nside(self.binned_data.shape[1])}, but given nside is {self.nside}! \
                                     \n    You need to rebin your data or use force=True.")
            print(f'Load binned data <-- {data}')
            print(f'    Binned data loaded: Contains {self.binned_data.sum().sum()} total counts')
              
        else:
            #If the data is not already binned, load and bin it
            self.load(data)
            
        #Get logE bin indices for MC events
        if self.n_logE_bins > 1:
            sig_ebin_inds = np.digitize(self.sig_evs[self.logE_name], self.logE_bins)
        elif self.n_logE_bins == 1:
            sig_ebin_inds = np.zeros_like(self.sig_evs[self.logE_name], dtype=int)
        else:
            raise ValueError("Something is wrong with the number of energy bins!")
        
        #Get median angular error from truth in MC for each ebin (used for smoothing of template)
        #Also get signal acceptance fraction for use in TS and injections
        #Also bin the signal events which is used in PDF making
        med_sigs = np.zeros(self.n_logE_bins)
        self.sig_acc_frac = np.zeros(self.n_logE_bins)
        self.rel_binned_sig = np.zeros((self.n_logE_bins, hp.nside2npix(self.nside)))
        for e in range(self.n_logE_bins):
            #Mask events in this ebin
            e_mask = sig_ebin_inds == e
            #Get weighted median
            med_sigs[e] = self.weighted_quantile(self.sig_evs['true_angErr'][e_mask], self.sig_relweights[e_mask], 0.5)
            #Get fraction of expected signal events
            self.sig_acc_frac[e] = np.sum(self.sig_relweights[e_mask]) / np.sum(self.sig_relweights)
            #Get binned signal events (relative weights, not actual counts)
            self.rel_binned_sig[e] = self.bin_data(self.sig_evs[e_mask], verbose=False, sig=True, weights=self.sig_relweights[e_mask])
        
        #Get coordinates of pixels        
        self.bin_thetas, self.bin_phis = hp.pix2ang(self.nside, np.arange(hp.nside2npix(self.nside)))
        self.bin_ras = self.bin_phis
        self.bin_decs = np.pi/2.0 - self.bin_thetas

        #Load template, create versions with acceptance and/or smoothing
        print(f'Load template <-- {template}')
        template = np.load(template, allow_pickle=True)         
        self.template = template.copy()
        self.create_template_pdf(med_sigs)

        #Get S and B PDFs for likelihood (properly normalized, I hope)
        self.get_pdfs_sigsub()

        print('***Setup complete!*** \n')
        
    def weighted_quantile(self, data, weights, quantile):
        #Used mostly for getting weighted medians (quantile = .5)
        ix = np.argsort(data)
        data = data[ix] # sort data
        weights = weights[ix] # sort weights
        csw = np.cumsum(weights)
        cut = np.sum(weights) * quantile
        if len(data) == 0:
            q = 0.0
        else:
            q = data[csw >= cut][0]
        return q
        
    def bin_data(self, data, verbose=None, sig=False, weights=None):
        """
        Convert event data into bin counts using healpy. 
        
        Args:
            data: data event array(s)
            
            verbose: True to show more output (Defaults to class's initited value)
            
            sig: Boolean, True if binning signal (using true locations and weights). False (default) for data which uses reco locations.
            
            weights: Array of weights to use when sig=True (Default: None).
            
        """
        
        if verbose is None:
            verbose = self.verbose
        
        #Get dec and ra of events
        if sig:
            event_decs = data['true_dec']
            event_ras = data['true_ra']
        else:
            event_decs = data['dec']
            event_ras = data['ra']
        
        if verbose:
            print(f'Binning {len(event_ras)} events with nside={self.nside} ', end=' ')
        
        #Get pixel number for each event
        event_pix_nums = hp.ang2pix(self.nside, np.pi/2.0 - event_decs, event_ras)
        
        if verbose:
            print('--> Binning Done.')
        
        #Return the count for each pixel
        #minlength ensures the output array is the proper length even if there are pixels with zero counts on the end
        if sig:
            return np.bincount(event_pix_nums, minlength=hp.nside2npix(self.nside), weights=weights)
        else:
            return np.bincount(event_pix_nums, minlength=hp.nside2npix(self.nside))
    
    def load(self, path, verbose=None):
        """
        Loads data and bins it.
        
        Args:
            path: path to directory containing data files or path to a data file
            
            verbose: True to show more output (Defaults to class's initited value)
            
        """
        if verbose is None:
            verbose = self.verbose
        #Make sure path is a directory or a file
        assert (os.path.isdir(path) or os.path.isfile(path)), f"Expected path to directory or file, got: {path}"
        
        print(f'Loading and binning data from {path}')
        if os.path.isdir(path):
            #If path is a directory, make list of numpy files from that directory
            files_like = path+'/*.npy'
            file_list = sorted(glob(files_like))
            
        else: #path is one file
            #If just one file, make it into a list
            file_list = [path]
            
        self.binned_data = np.zeros((self.n_logE_bins, hp.nside2npix(self.nside)))
        #Loop through files    
        for file in file_list:
            #Load the data
            data = np.load(file)
            #If using qtot instead of energy, add the logQtot field to the data array (just like for MC above)
            if self.logE_name == 'logQtot':
                if 'qtot_wdc' in data.dtype.names:
                    q_name = 'qtot_wdc'
                else:
                    q_name = 'qtot'
                    
                #If binning data, make the same addition of logQtot field to the data array
                #Add log10 of QTot to the sig array for consistency with logE
                new_dtype = np.dtype(data.dtype.descr + [(self.logE_name, '<f8')])
                new_arr = np.empty(data.shape, dtype=new_dtype)
                for n in data.dtype.names:
                    new_arr[n] = data[n]
                new_arr[self.logE_name] = np.log10(data[q_name])

                data = new_arr.copy()
                del new_arr #free mem
            
            #Mask events from GRL
            mask = np.isin(data['run'], self.grl['run'])
            data = data[mask]
            
            #Get logE bin indices for each event (if only one bin, all zeros)
            if self.n_logE_bins > 1:
                e_inds = np.digitize(data[self.logE_name], self.logE_bins)
            elif self.n_logE_bins == 1:
                e_inds = np.zeros_like(data[self.logE_name], dtype=int)
            else:
                raise ValueError("Something is wrong with the number of energy bins!")

            if verbose:
                print(f'    {file} : ')
            
            #Loop through unique energy bin indices
            for e in np.unique(e_inds):
                if verbose:
                    print(f'        Energy Bin {e+1}/{self.n_logE_bins}...', end=' ')
                #Mask events in this energy bin
                e_mask = (e_inds == e)
                #bin the events from this file and add to the binned_data (initialized above as all zeros)
                self.binned_data[e] = self.binned_data[e] + self.bin_data(data[e_mask])
                
                if verbose:
                    print(' --> Done.')
        
        #Free up memory
        del data
        gc.collect()
        
        print('--> Data Loading Done. \n')
        
        #If loading and binning data, and if savedir is given, save
        if self.savedir is not None:
            binned_dict = {'logE_bins': self.logE_bins,
                           'binned_data': self.binned_data}
            savefile = f'{self.savedir}/{self.name}.binned_data.nside{self.nside}.npy'
            i = 0
            #if save file already exists, append a number to the file name
            if os.path.exists(savefile):
                print('Saved file of chosen name already exists!')
                while os.path.exists(savefile):
                    savefile = f'{self.savedir}/{self.name}.binned_data_{i}.nside{self.nside}.npy'
                    i += 1
            np.save(savefile, binned_dict)
            print(f'Binned data saved to --> {savefile}')
            
        return
        
    def create_bg_acc_spline(self, skw={}):
        """
        Create detector acceptance spline (*background*, sinDec-dependent).
        
        Pieces taken from csky.
        
        TO DO: Is this right?

        Args:          
            skw: histlite.Hist.spline_fit kwargs
        
        Returns: histlite spline object
        
        """
        #Spline Params
        skw.setdefault('s', 0.0) #smooth
        skw.setdefault('k', 2) #spline degree
        skw.setdefault('log', True) #fit to log values
        
        #Determine dec bands for pdf:
        bin_edges = np.unique(self.bin_decs)[:-1] + np.diff(np.unique(self.bin_decs))/2
        bin_edges = np.r_[-np.pi/2, bin_edges, np.pi/2]
    
        bg_acc_spline = np.empty(self.n_logE_bins, dtype=object)
        bg_hist = np.empty(self.n_logE_bins, dtype=object) # (test)
        for e in range(self.n_logE_bins):
            #histogram data in sinDec
            h_counts = hl.hist(np.sin(self.bin_decs), weights=self.binned_data[e], bins=np.sin(bin_edges))
            h_vals = np.where(h_counts.values==0, 1, h_counts.values)
            h_counts = hl.Hist(values=h_vals, bins=h_counts.bins)
            #Normalize such that integral over solid angle = 1
            h = h_counts.normalize(density=True) / (2*np.pi)
            h = h.gaussian_filter1d(sigma=2) # (test)
            #Fit spline          
            s_hl = h.spline_fit(**skw)

            bg_acc_spline[e] = s_hl.spline
            bg_hist[e] = h # (test)
        
        self.bg_acc_spline = bg_acc_spline
        self.binned_bg_hists = bg_hist # (test)
        #print('bg done')
        return
    
    def create_signal_acc_spline(self, skw={}):
        """
        Create detector acceptance spline (*signal*, sinDec-dependent).
        
        Pieces taken from csky.
        
        Args:          
            skw: histlite.Hist.spline_fit kwargs
        
        Returns: histlite spline object
        
        """
        #Spline Params
        skw.setdefault('s', 25)
        skw.setdefault('k', 2)
        skw.setdefault('log', True)
        
        sig_acc_spline = np.empty(self.n_logE_bins, dtype=object)
        sig_hist = np.empty(self.n_logE_bins, dtype=object)
        #Get ebin indices for MC events
        if self.n_logE_bins > 1:
            sig_ebin_inds = np.digitize(self.sig_evs[self.logE_name], self.logE_bins)
        elif self.n_logE_bins == 1:
            sig_ebin_inds = np.zeros_like(self.sig_evs[self.logE_name], dtype=int)
        else:
            raise ValueError("Something is wrong with the number of energy bins!")

        bin_edges = np.unique(self.bin_decs)[:-1] + np.diff(np.unique(self.bin_decs))/2
        bin_edges = np.r_[-np.pi/2, bin_edges, np.pi/2]
            
        #loop through ebins
        for e in range(self.n_logE_bins):
            #Mask events in this ebin
            sig_ebin_mask = sig_ebin_inds == e
            #Make hist, weighted with relative weights
            h_counts = hl.hist(np.sin(self.sig_evs['true_dec'][sig_ebin_mask]), weights=self.sig_relweights[sig_ebin_mask], bins=np.sin(bin_edges)) # (test)
            #Normalize such that integral over solid angle = 1
            h = h_counts.normalize(density=True) / (2*np.pi)
            h = h.gaussian_filter1d(sigma=2) # (test)
            #Fit spline
            s_hl = h.spline_fit(**skw)

            sig_acc_spline[e] = s_hl.spline
            sig_hist[e] = h
        
        self.signal_acc_spline = sig_acc_spline
        self.sig_hists = sig_hist
        #print('sig done')
        return  
    
    def get_acc_from_spline(self, sindec, e, acc):
        """
        Used spline to get acceptance at a give sin(Dec) for a given energy bin index.
        
        Args:
            sindec: Sine of declination(s)
            
            e: index of bin in range of 0 to len(logE_bins)-1
            
            acc: One of "signal" or "bg" for which acceptance spline to use.
            
        Returns: acceptance(s) for provided sin(Dec).
        
        """
            
        if acc == 'signal':
            try:
                out = self.sig_hists[e].get_values(sindec) # (test)
                
            #If the signal_acc_spline has not been created...
            except AttributeError:
                print('Signal acceptance spline not yet created. Creating now... \n')
                self.create_signal_acc_spline()
                out = self.sig_hists[e].get_values(sindec) # (test)
                
            return out
        
        #Same for background...
        elif acc == 'bg':
            try:
                out = self.binned_bg_hists[e].get_values(sindec) # (test)
                
            except AttributeError:
                print('Background acceptance spline not yet created. Creating now... \n')
                self.create_bg_acc_spline()
                out = self.binned_bg_hists[e].get_values(sindec) # (test)            
                
            return out
        
        else: #if acc is not signal or bg
            raise ValueError('Argument spline must be one of ["signal", "bg"].')
            
    def create_template_pdf(self, smooth_sigs):
        """
        Applies detector acceptance to template and smooths, normalizes.
        
        Args:
            smooth_sig: Sigma (in radians) for gaussian smoothing using healpy.smoothing(x, sigma=smooth_sig)

        """
   
        print("Applying detector acceptance to template...")
        if not hasattr(self, 'signal_acc_spline'):
            #Make acceptance spline, if not already made
            self.create_signal_acc_spline()
            
        template = self.template.copy()
        
        #If template nside does not match given nside, resize it
        if self.nside != hp.npix2nside(len(template)):
            print('*** Rescaling template to match provided Nside ***')
            template = hp.ud_grade(template, self.nside)

        self.template_acc = np.zeros((self.n_logE_bins, hp.nside2npix(self.nside)))
        self.template_acc_smoothed = np.zeros((self.n_logE_bins, hp.nside2npix(self.nside)))
        def normalize_pdf(temp, sig=None):
            #This normalization procedure is based on that of csky:
            #Normalize
            temp_pdf = temp * hp.nside2pixarea(self.nside)
            temp_pdf /= (np.sum(temp_pdf))
            temp_pdf /= hp.nside2pixarea(self.nside)
            #This mask usually will not have any true pixels, but should avoid strange behavior for very small division
            mask = (template > 0) & (temp_pdf <= 0)
            temp_pdf[mask] = hp.UNSEEN

            if sig is not None:
                #Smooth
                temp_pdf = hp.smoothing(temp_pdf, sigma=sig)

            #Reset nonsensical values, and apply a floor
            #The floor could help if the template is ever in a division
            #The floor shouldnt affect the TS so long as it is below the cutoff (it becomes 0 in the signal PDF)
            temp_pdf[mask] = 0
            temp_pdf[temp_pdf < 1e-12] = 1e-12
            
            #Re-normalize after smoothing and within dec bounds
            dec_mask = (self.bin_decs<=np.radians(self.max_dec_deg)) & (self.bin_decs>=np.radians(self.min_dec_deg))
            temp_pdf = temp_pdf / ( np.sum(temp_pdf[dec_mask]) * hp.nside2pixarea(self.nside) ) # [dec_mask]

            return temp_pdf
        
        print('Smoothing and normalizing template...')
        #Now for each ebin...
        for e in range(self.n_logE_bins):
            #Apply signal acceptance 
            template_acc = template * self.get_acc_from_spline(np.sin(self.bin_decs), e, acc='signal')                   
            #Do the normalization (and smoothing)
            self.template_acc[e] = normalize_pdf(template_acc, sig=None)
            self.template_acc_smoothed[e] = normalize_pdf(template_acc, sig=smooth_sigs[e])
            #self.template_acc_smoothed[e] = self.template_acc[e].copy() # (test)
        
        print('--> Template PDF-ization: Done. \n')
        
        return

    def create_sigsub_pdf(self):
        """
        For using sigsub, creates signal sindec pdf from mean per dec of template_acc_smoothed.
        """
        bin_edges = np.unique(self.bin_decs)[:-1] + np.diff(np.unique(self.bin_decs))/2
        bin_edges = np.r_[-np.pi/2, bin_edges, np.pi/2]

        #loop through ebins
        pdf_sigsub = np.zeros((self.n_logE_bins, hp.nside2npix(self.nside)))
        dec_bound_mask = (self.bin_decs<=np.radians(self.max_dec_deg)) & (self.bin_decs>=np.radians(self.min_dec_deg))
        for e in range(self.n_logE_bins):
            for d in np.unique(self.bin_decs):
                dec_mask = (self.bin_decs == d)
                pdf_sigsub[e, dec_mask] = np.mean(self.template_acc_smoothed[e, dec_mask])
            #Normalize to integrate to 1 over solid angle
            pdf_sigsub[e] = pdf_sigsub[e] / ( np.sum(pdf_sigsub[e, dec_bound_mask]) * hp.nside2pixarea(self.nside) )

        self.pdf_sigsub = pdf_sigsub

        return  

    def multinomial_TS(self, n_sig, n, p_s, p_b, p_ssub, frac=1.0):
        """
        This function is used to calculate the multinomial TS with sigsub:
        see: https://wiki.icecube.wisc.edu/index.php/Cascade_Neutrino_Source_Dataset/GP_Analysis#Likelihood
        
        It is minimized for n_sig in the fitting functions.
        
        Args:
            n_sig: number of (signal) events
            
            n: array of event counts in pixels (via healpy)
            
            p_s: array of pixel-wise signal probabilities using signal/template PDf
            
            p_b: array of pixel-wise scrambled data probabilities using background (scrambled data) PDF

            p_ssub: array of pixel-wise sigsub (sindec-dependent signal) probabilities
            
            frac: fraction of signal events (for different energy bins)
            
        Returns: Test Statistic
        
        """
        #Here, N has length of ebins, and is total events in each ebin
        N = np.sum(n, axis=1)[:, np.newaxis]

        if self.sigsub:
            #TS = 2.0 * np.sum( n * np.log( ( (frac*n_sig) / (N * p_b) ) * (p_s - p_ssub) + 1.0 ) )
            TS = 2.0 * np.sum( n * np.log( np.maximum( ( (frac*n_sig) / (N * p_b) ) * (p_s - p_ssub) + 1.0 , 1e-12 ) ) ) # (test)
        else:
            TS = 2.0 * np.sum( n * np.log( (frac*n_sig / N ) * (p_s / p_b - 1.0) + 1.0 ) )
        
        return TS
    
    def get_pdfs_sigsub(self, verbose=None):
        """
        Creates signal and background pdfs used in the test statistic calculation/minimization.
        
        `p_s` is the signal PDF; i.e., the template_acc_smoothed 
        
        `p_b` is the scrambled data PDF

        `p_ssub` is the sindec-dependent signal PDF for sigsub
        
        """
        if verbose is None:
            verbose = self.verbose
        
        if verbose:
            print('Creating signal and background PDFs for TS calculations...')
            
        self.p_s = np.zeros((self.n_logE_bins, hp.nside2npix(self.nside)))
        self.p_b = np.zeros((self.n_logE_bins, hp.nside2npix(self.nside)))
        #Mask for min/max decs used in normalizing PDFs
        dec_mask = (self.bin_decs<=np.radians(self.max_dec_deg)) & (self.bin_decs>=np.radians(self.min_dec_deg))
        
        #Signal PDF is template with acceptance and smoothing
        p_s = self.template_acc_smoothed.copy()
        p_s /= np.sum(p_s[:,dec_mask], axis=1)[:, np.newaxis] * hp.nside2pixarea(self.nside)    #Normalize within each ebin
        
        #Scrambled data PDF comes straight from the spline/hist
        p_b = np.array([self.get_acc_from_spline(np.sin(self.bin_decs), b, acc='bg') for b in range(self.n_logE_bins)])   
        p_b /= np.sum(p_b[:,dec_mask], axis=1)[:, np.newaxis] * hp.nside2pixarea(self.nside)    #Normalize within each ebin

        #Sigsub PDF
        self.create_sigsub_pdf()
        p_ssub = self.pdf_sigsub.copy()
        p_ssub /= np.sum(p_ssub[:,dec_mask], axis=1)[:, np.newaxis] * hp.nside2pixarea(self.nside)    #Normalize within each ebin
        
        self.p_s = p_s
        self.p_b = p_b
        self.p_ssub = p_ssub
        
        if verbose:
            print('--> PDFs Done.')
            
        return
    
    def get_one_fit(self, n_sig=0, truth=False, seed=None, verbose=None, poisson=True):
        """
        Obtains a single all-sky likelihood ratio.
        
        Args:
            n_sig: number of (signal) events to inject (Default: 0). Only used if truth=False
                    
            truth: Whether to use true event locations (True), or scramble in right ascension (False)
        
            seed: integer, Seed for numpy.random.default_rng() used to scramble events and pick injection locations
                  (Default: None, meaning unpredictable)
                  
            verbose: True to show more output (Defaults to class's initited value)
            
            poisson: boolean, if number of injected events should be poisson chosen around n_sig. (Default: True)
            
        Returns: dictionary containing all-sky llr, sinDec llrs, and sinDec acceptances
        
        """
        if verbose is None:
            verbose = self.verbose
        
        if truth:
            #If truth, then the counts used in the TS are the binned data
            self.counts = self.binned_data.copy()

        else:
            #If not truth, create a scramble
            self.counts = self.scrambler(seed=seed, verbose=verbose)                
            #if n_sig != 0:
                #If n_sig > 0, perform injections (which get added to self.counts)
            self.template_injector(n_sig=n_sig, seed=seed, verbose=verbose, poisson=poisson)
        
        #Mask the dec bounds
        dec_mask = (self.bin_decs<=np.radians(self.max_dec_deg)) & (self.bin_decs>=np.radians(self.min_dec_deg))

        # Attempt to fix memory leak: see https://github.com/icecube/csky/pull/61/commits/dcbd06fc72b096b6a3263ddf10744c4f50dff092
        # (test)
        proxy_self = weakref.proxy(self)
    
        #TS calculation uses pixels within dec bounds only (e.g., to exclude poles)
        n = proxy_self.counts[:,dec_mask].copy()
        p_s = proxy_self.p_s[:,dec_mask].copy()
        p_b = proxy_self.p_b[:,dec_mask].copy()
        p_ssub = proxy_self.p_ssub[:,dec_mask].copy()
        frac = proxy_self.sig_acc_frac[:, np.newaxis]
        
        #Convenience function is of one variable ns and is the negative of the TS
        #Then, minimize...
        def min_neg_TS(ns):
            return -1.0 * proxy_self.multinomial_TS(ns, n, p_s, p_b, p_ssub, frac=frac)
        
        if verbose:
            print('Minimizing -TS...')
        
        #Using iminuit brute force scan to find minimum, then Migrad for better accuracy
        res = iminuit.Minuit(min_neg_TS, 1e2)
        res.print_level = 0
        #Limit ns to between 0 and number of events
        res.limits['ns'] = (0.0, np.sum(self.counts))
        res.scan()
        #res.scan() # (test) For some reason, scanning twice fixes issue with occasional invalid minima
        res.migrad()
        #res.hesse()
        fit_ns = round(res.values['ns'], 4)
        fit_TS = -1.0 * min_neg_TS(fit_ns)
        self.minuit_result = res
        
        #Save result to a structured array
        result = np.array([(seed, fit_ns, fit_TS)], dtype=[('seed', int),('ns', float),('ts', float)])
        
        if verbose:
            print(f' --> One All Sky Fit Done: ns={fit_ns}, TS={fit_TS}')
            
        print(res)
        del proxy_self
        gc.collect()
        return result
    
    def get_many_fits(self, num, n_sig=0, seed=None, verbose=None, poisson=True):
        """
        Obtains multiple best-fit ns and TS.
        
        Args:
            num: integer, number of llrs to compute
            
            n_sig: number of (signal) events to inject (Default: 0)
                                    
            seed: integer, seed used to create multiple new seeds for scrambles (Default: None, unpredictable)
            
            verbose: True to show more output (Defaults to class's initited value)
            
            poisson: boolean, if number of injected events should be poisson chosen around n_sig. (Default: True)
            
        Returns: dictionary with {'n_sig': n_sig, 'results': structured array of (seed, ns, ts) }
        
        """
        print(f'Calculating {num} TS with {n_sig} injected event(s)...')
        if verbose is None:
            verbose = self.verbose
            
        results = np.array([],dtype=[('seed', int),('ns', float),('ts', float)])
        
        num = int(num)
        if num < 1:
            #number of trials cannot be < 1
            raise ValueError(f'num must be a positive integer, got: {num}')
        elif num == 1:
            #One trial, one fit
            results = np.append(results, self.get_one_fit(n_sig=n_sig, seed=seed, acc=acc, verbose=verbose, poisson=poisson))
        else:
            #Multiple trials, get random seeds for each trial
            rng_seed = np.random.default_rng(seed)
            new_seeds = rng_seed.integers(int(2**32), size=num)
            #Do the trials with these seeds
            for s in new_seeds:
                results = np.append(results, self.get_one_fit(n_sig=n_sig, seed=s, verbose=verbose, poisson=poisson))
        
        #Store result in dictionary
        res_dict = {'n_inj': n_sig, 'results': results}
        
        print(f'--> {num} Fits Done!')
        
        return res_dict        
         
    def fit_TS_chi2(self, tss): #Rarely, if ever, used
        """
        Fit distribution of TSs > 0 with a chi-squared funtion.
        
        Args:
            tss: array of TSs
        
        Returns: Chi2 degrees of freedom, location, scale parameters
        
        """
        self.eta = np.mean(tss>0)
        self.df, self.loc, self.scale = sp.stats.chi2.fit(tss[tss>0], 1)     # 1 indicates starting guess for chi2 degrees of freedom
        self.chi2_fit = sp.stats.chi2(self.df, self.loc, self.scale)
        
        print('Chi2 distribution fit to TS > 0 with:')
        print(f'    DoF: {self.df}')
        print(f'    loc: {self.loc}')
        print(f'    scale: {self.scale}')
        print(f'    eta: {self.eta}')
        
        return 
    
    def TS_to_p(self, ts, bg=None, use='fit', dof=1): #Only used after trials (does not contribute to ns issue)
        """
        Converts given TS to p-value using chosen method.
        
        Args:
            ts: float or array of floats, TS to convert to p-value
            
            bg: None or array, background TSs (Default: None)
            
            use: string, which method/distribution to use.
                 One of 'fit', 'chi2_dof', or 'hist'. If 'fit' and argument bg is not None, will fit a Chi2 distribution to\
                 the provided bg distribution. If 'chi2_dof', will use a Chi2 with provided dof and loc=0, scale=1. If \
                 'hist', will use the bg TSs themselves. (Default: 'fit')
                 
            dof: Number of degrees of freedom to use if use='chi2_ndof'
            
        Returns: p-value(s)
                 
        """
        if use not in ['fit', 'chi2_dof', 'hist']:
            raise ValueError(f'Argument use must be one of "fit", "chi2_dof", or "hist", got "{use}"')
            
        if (use == 'chi2_ndof') or (bg is None):
            #TS --> p using generic Chi2 function
            #Necessary if no BG distribution is provided
            
            print(f'Calculating p-value(s) using a Chi2 Distribution with {dof} degrees of freedom...')
            p = sp.stats.chi2.sf(ts, df=dof)
            print(f'    p = {p}')
            
        elif use == 'fit':
            #TS --> p using Chi2 fit
            #Requires BG dist to be provided
            
            print(f'Calculating p-value(s) using a Chi2 Distribution fit to provided background...')
            self.fit_TS_chi2(bg)
            p = self.chi2_fit.sf(ts)
            print(f'    p = {p}')
            
        elif use == 'hist':
            #TS --> p using distribution directly
            #Requires BG dist to be provided
            
            print(f'Calculating p-value(s) using background TSs distribution directly...')
            p = np.array([np.mean(bg >= t) for t in ts])
            print(f'    p = {p}')
            
        return p

    def scrambler(self, seed=None, verbose=None):
        """
        Gets a map of "scrambled" counts by sampling from self.bin_count_spline with given sindec
        
        Args:       
            seed: integer, Seed for numpy.random.default_rng() used to scramble events and pick injection locations
                  (Default: None, meaning unpredictable)
        
        """
        if verbose is None:
            verbose = self.verbose
        
        if verbose:
            print(f'Creating random scramble with seed {seed}...')
        
        #Get unique pixel decs
        unique_decs = np.unique(self.bin_decs)
        
        rng_scramble = np.random.default_rng(seed=seed)
        counts = np.zeros_like(self.binned_data)
        #Loop through ebins
        for e in range(self.n_logE_bins):
            #Get number of bg events in whole sky (for ebin e)
            num_bg_events = np.sum(self.binned_data[e])
            #Loop through unique pixel decs
            for i,dec in enumerate(unique_decs):
                #Mask pixels in a row of dec
                dec_mask = (self.bin_decs == dec)
                #Get per pixel bg events for dec
                val = self.get_acc_from_spline(np.sin(dec), e, acc='bg') * num_bg_events * hp.nside2pixarea(self.nside) 
                #Poisson sample around val in this dec
                counts[e,dec_mask] = rng_scramble.poisson(lam=val, size=np.sum(dec_mask))
                
            #Adjust so each ebin has sum(counts) ~= sum(binned_data)
            #counts[e] = np.around(counts[e] * np.sum(self.binned_data[e]) / np.sum(counts[e]))
            
        if verbose:
            print(f'--> Scrambling Done. Scramble contains {np.sum(counts)} total counts.')
            
        return counts            

    def template_injector(self, n_sig, seed=None, verbose=None, poisson=True):
        """
        Injects events based on template probabilities.
        
        Args:
            n_sig: number of events to inject (with poisson fluctuations)
            
            seed: integer, Seed for numpy.random.default_rng() used to pick injection locations (Default: None, meaning unpredictable)
            
            poisson: boolean, if number of injected events should be poisson chosen around n_sig. (Default: True)
            
        """
        if verbose is None:
            verbose = self.verbose
                           
        rng_inj = np.random.default_rng(seed=seed)
        
        #If injecting number of events based on Poisson around nsig...
        if poisson:
            n_sig = rng_inj.poisson(lam=n_sig)
        
        if verbose:
            print(f'Injecting {n_sig} events in "On" bins according to per-energy-bin template+acceptance probabilities...')
        
        self.inj_bins = np.empty(self.n_logE_bins, dtype=object)
        #Get ebin indicies for MC events
        if self.n_logE_bins > 1:
            sig_ebin_inds = np.digitize(self.sig_evs[self.logE_name], self.logE_bins)
        elif self.n_logE_bins == 1:
            sig_ebin_inds = np.zeros_like(self.sig_evs[self.logE_name], dtype=int)
        else:
            raise ValueError("Something is wrong with the number of energy bins!")
            
        #Loop through ebins
        for e in range(self.n_logE_bins):
            #Injection in each ebin is signal fraction * total nsig
            n_inj = np.rint(self.sig_acc_frac[e] * n_sig).astype(int)
            
            #Mask the MC events in this ebin
            sig_ebin_mask = sig_ebin_inds == e
            sig = self.sig_evs[sig_ebin_mask]
        
            #Injection bins are choice from within dec min/max bounds with replacement
            dec_mask = (self.bin_decs<=np.radians(self.max_dec_deg)) & (self.bin_decs>=np.radians(self.min_dec_deg))
            inj_choice = np.arange(hp.nside2npix(self.nside))[dec_mask]
            
            #The probability of injection for each bin within the ON region includes acceptance but no smoothing
            inj_probs = self.template_acc[e, dec_mask]
            inj_bins = rng_inj.choice(inj_choice, size=n_inj, p=inj_probs/np.sum(inj_probs))
            
            inj_bin_decs = self.bin_decs[inj_bins] # (test)
            
            inj_evs = np.array([], dtype=self.sig_evs.dtype)
            ps = np.array([])
            #Loop through chosen injection pixels with the number of injections in that pixel
            #bin_edges = np.unique(self.bin_decs)[:-1] + np.diff(np.unique(self.bin_decs))/2
            #bin_edges = np.r_[-np.pi/2, bin_edges, np.pi/2]
            #bin_sindec_bin_inds = np.digitize(np.sin(self.bin_decs[inj_bins]), np.sin(bin_edges))
            #for i in np.unique(bin_sindec_bin_inds):
            d, n = np.unique(inj_bin_decs, return_counts=True) # (test)
            for d, n in zip(d,n): # (test)
                #mask MC events that are in the same sindec band as the pixel chosen for injection
                #sig_dec_mask = np.digitize(np.sin(sig['true_dec']), np.sin(bin_edges)) == i
                #sig_in_dec = sig[sig_dec_mask]
                sig_dec_mask = (sig['true_dec'] > d-np.radians(3)) & (sig['true_dec'] < d+np.radians(3)) # (test)
                sig_in_dec = sig[sig_dec_mask]
                
                #pick random events from that sindec band selection with size equal to occurrences (weighted choice)
                choice_weights = self.sig_relweights[sig_ebin_mask][sig_dec_mask]
                #n = np.sum(bin_sindec_bin_inds==i)
                
                #At full scale, with enough MC, replace should be set to False (so as to not select the same MC event twice)
                inj_evs = np.concatenate((inj_evs, rng_inj.choice(sig_in_dec, size=n, replace=False, p=choice_weights/np.sum(choice_weights), shuffle=False)))
                #Keep track of injection pixels from each dec band
                #ps = np.concatenate((ps,inj_bins[bin_sindec_bin_inds==i]))
                ps = np.concatenate((ps, inj_bins[self.bin_decs[inj_bins]==d])) # (test)
                
                #For pixels in the same sindec band, it doesnt matter which events go to which pixel
                #    since it is a random assignment. So the events in inj_evs are assigned to the 
                #    pixels in ps, which are the pixels in this sindec band ordered as they were in 
                #    the random choice inj_bins above.
                
            ps = ps.astype(int)

            #Get shift polar angle and spherical distance between chosen events and the respective injection pixels
            true_ev_coord = ap.coordinates.SkyCoord(inj_evs['true_ra'], inj_evs['true_dec'], unit='rad', frame='icrs')
            true_bin_coord = ap.coordinates.SkyCoord(hp.pix2ang(self.nside, ps)[1], np.pi/2-hp.pix2ang(self.nside, ps)[0], unit='rad', frame='icrs')
            pos_angle = true_ev_coord.position_angle(true_bin_coord).radian
            sep = true_ev_coord.separation(true_bin_coord).radian

            #Move the events reco direction the same amount
            start_coord = ap.coordinates.SkyCoord(inj_evs['ra'], inj_evs['dec'], unit='rad', frame='icrs')
            stop_coord = start_coord.directional_offset_by(pos_angle * ap.units.rad, sep * ap.units.rad)
            inj_ra, inj_dec = stop_coord.ra.radian, stop_coord.dec.radian
                
            #Actually inject the events in the pixel corresponding to the moved reco direction
            reco_inj_bins = hp.ang2pix(self.nside, np.pi/2-inj_dec, inj_ra)
            #Get unique injection pixels and their number of occurrences 
            reco_bin_nums, reco_bin_injs = np.unique(reco_inj_bins, return_counts=True)
            #Increase counts of injection pixels by their number of occurrences
            self.counts[e,reco_bin_nums] += reco_bin_injs                
            
            self.inj_bins[e] = reco_inj_bins
        
        if verbose:
            print('--> Injections Done.')
                
        return          
        
