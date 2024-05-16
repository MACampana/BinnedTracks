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

from glob import glob

class BinnedTemplateAnalysis:
    """For conducting binned calculations using maximum likelihood statistical methods. 
    For binned sky map of IceCube event data, specifically for template analyses.
    
    (Now with energy!)
    
    
    """
    def __init__(self, data, sig, grl, is_binned=False, savedir=None, name='BinnedTemplateAnalysis', 
                 template=None, gamma=2.7, cutoff=None, ebins=None,
                 nside=128, min_dec_deg=-80, max_dec_deg=80, qtot=True,
                 verbose=False, force=False):
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
                                                 ##('angErr', '<f8')])
                                                 
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
                                                 ('true_angErr', float), 
                                                 ('oneweight', float),     
                                                 ('true_energy', float)]
            
            grl: Path to numpy array with GRL runs
                                     
            is_binned: boolean, True if argument data is an array of binned_data, otherwise False (data will be binned)
            
            savedir: path to directory to save binned data. Default: None (don't save)
            
            name: unique name to identify analysis (used in file names when saving)
            
            template: path to template array or None
                TO DO:
                    * Rescale template nside for differing energy ranges?
                                                            
            gamma: spectral index for detector acceptance and injections. 
                    
            cutoff: minimum template value to be considered an On bin
                TO DO (???):
                    * Implementation without need for cutoff???
                    
            ebins: Choice of log10(energy) bins. If None, use default (hardcoded) bins. If a single integer,
                use `ebins` evenly spaced bins in log10(energy). If a list of numbers, assumed to be 
                log10(energy) bin edges. If `is_binned` is True, `ebins` will be determined from the loaded data.

            nside: integer for healpy nside (Default: 128)
                TO DO: 
                    * Allow nside to differ for different energy ranges?
                        
            min_/max_dec_deg: min and max declination in degrees for likelihood calculations.
            
            qtot: If True, use total charge instead of reconstructed energy for "energy" binning of events
            
            verbose: True for lots of output (Default: False)
            
            force: boolean, if `is_binned` is True and the loaded data does not match the provided `nside`, set this
                to True to resize the loaded, binned data to the give nside. If False, will raise ValueError (Default: False)
            
            
        """
        '''
        self.bb_bins = np.array([np.array([-1.        , -0.89950881, -0.83661275, -0.80065442, -0.76553614,
                  -0.69690706, -0.63707182, -0.44462304, -0.32370584, -0.18437429,
                  -0.14481433, -0.04738862,  0.03614851,  0.07055893,  0.10370578,
                   0.14263341,  0.18036763,  0.23877537,  0.32903122,  0.67796469,
                   0.76397449,  0.81928373,  0.85810175,  0.90456654,  0.95961778,
                   1.        ])                                                   ,
           np.array([-1.        , -0.78476815, -0.69908404, -0.62853187, -0.32848784,
                  -0.1782143 , -0.14411293, -0.08989597, -0.03075927,  0.03954887,
                   0.12793786,  0.22504095,  0.34646798,  0.74268888,  0.82304583,
                   0.88710902,  0.97455871,   1.])                                 ,
           np.array([-1.        , -0.94508251, -0.79744124, -0.64653098, -0.33665038,
                  -0.23981953, -0.17296853, -0.1127156 , -0.05814776,  0.02298234,
                   0.10693383,  0.21212381,  0.68803068,  0.7595799 ,  0.82960466,
                   0.87462507,  0.94333963,  1.        ])                         ,
           np.array([-1.        , -0.93228461, -0.6887116 , -0.30945155, -0.19231385,
                  -0.02493273,  0.05556296,  0.14551715,  0.29527479,  0.66134755,
                   0.72700736,  0.83179921,  0.88889625,  1.        ])            ,
           np.array([-1.        , -0.89587741, -0.63983012, -0.32088205, -0.27587493,
                  -0.18331951, -0.0625932 ,  0.03519046,  0.17708531,  0.66425052,
                   0.74350907,  0.84517586,  1.        ])                         ,
           np.array([-1.        , -0.94980334, -0.89446603, -0.35797063, -0.30623653,
                  -0.23750648, -0.20005087, -0.14912647, -0.04772366,  0.02935457,
                   0.14264125,  0.27074083,  0.64923804,  0.73057599,  0.83079406,
                   1.        ])                                                   ,
           np.array([-1.        , -0.95915644, -0.91785802, -0.42934792, -0.32111846,
                  -0.23508843, -0.08848357,  0.03313759,  0.23282553,  0.58526565,
                   0.73127259,  0.81134119,  0.91122624,  1.        ])            ,
           np.array([-1.        , -0.63988874, -0.44057553, -0.34487885, -0.26602035,
                  -0.22448619, -0.11700594, -0.06483343,  0.00651613,  0.21425199,
                   0.55912025,  0.75801123,  0.88280736,  1.        ])            ,
           np.array([-1.        , -0.80537918, -0.37382054, -0.30461851, -0.2642643 ,
                  -0.01181373,  0.23229368,  0.58831011,  0.75657851,  0.91848494,
                   1.        ])                                                   ,
           np.array([-1.        , -0.78862124, -0.42719774, -0.33711707, -0.27909552,
                  -0.10108909, -0.00173871,  0.23913102,  0.59890677,  0.74766866,
                   1.        ])                                                   ,
           np.array([-1.        , -0.85109453, -0.43724261, -0.35357553, -0.30242805,
                  -0.09572638,  0.00568298,  0.14312297,  0.58453204,  0.76307897,
                   1.        ])                                                   ,
           np.array([-1.        , -0.85375089, -0.49444223, -0.41086568, -0.34493369,
                  -0.25914617, -0.12793911, -0.03857407,  0.3792026 ,  0.70109617,
                   1.        ])                                                   ,
           np.array([-1.        , -0.47993628, -0.42031807, -0.35668521, -0.2772424 ,
                  -0.12123379,  0.062789  ,  0.61076675,  1.        ])            ,
           np.array([-1.        , -0.65886077, -0.49294887, -0.41700345, -0.38278879,
                  -0.30697841, -0.14299963, -0.10765568,  0.05474702,  0.41822243,
                   1.        ])                                                   ,
           np.array([-1.        , -0.69577917, -0.58995407, -0.50162113, -0.44592239,
                  -0.14051574,  0.0290701 ,  0.40718245,  1.        ])            ,
           np.array([-1.        , -0.78487765, -0.57440096, -0.52616888, -0.3933832 ,
                  -0.34047809, -0.03850109,  0.41001694,  1.        ])            ,
           np.array([-1.        , -0.89003459, -0.67152018, -0.62820324, -0.53483758,
                  -0.45742703, -0.4173029 , -0.27119512, -0.22923853, -0.15156628,
                   0.21022644,  1.        ])                                      ,
           np.array([-1.        , -0.80362332, -0.7466046 , -0.54636915, -0.50492262,
                  -0.44541832, -0.41016856, -0.31881564, -0.22966573, -0.19571998,
                  -0.14148578,  0.07246615,  0.51337711,  1.        ])            ,
           np.array([-1.        , -0.90266999, -0.6508549 , -0.60722661, -0.56719411,
                  -0.50679383, -0.46906437, -0.43093971, -0.39770393, -0.35257006,
                  -0.29883854, -0.24346951, -0.19202965, -0.1170732 ,  0.2987159 ,
                   1.        ])                                                   ,
           np.array([-1.        , -0.82227933, -0.73916067, -0.64192097, -0.60591511,
                  -0.57344868, -0.49612254, -0.39820569, -0.3463813 , -0.29245966,
                  -0.25805732, -0.21943134, -0.12267165,  0.24823844,  1.        ]),
           np.array([-1.        , -0.93314273, -0.82412113, -0.7503662 , -0.68656827,
                  -0.60980959, -0.55463197, -0.47093148, -0.39804144, -0.35490189,
                  -0.31054281, -0.26578856, -0.20762202, -0.09619029,  1.        ]),
           np.array([-1.        , -0.93783505, -0.88733081, -0.79611619, -0.72912984,
                  -0.68260349, -0.61084009, -0.5376294 , -0.48168417, -0.42919966,
                  -0.3832681 , -0.33826319, -0.27812385, -0.1796548 ,  0.16011418,
                   1.        ])                                                   ,
           np.array([-1.        , -0.78947119, -0.68900664, -0.60469798, -0.54282587,
                  -0.48234702, -0.41529246, -0.34189329, -0.27095458, -0.20294518,
                  -0.09790813,  1.        ])                                      ,
           np.array([-1.        , -0.85487801, -0.75778255, -0.66065389, -0.59035808,
                  -0.52120806, -0.44588417, -0.36064746, -0.276493  , -0.17064716,
                   1.        ])                                                   ,
           np.array([-1.        , -0.95070853, -0.86310485, -0.79336687, -0.6910255 ,
                  -0.63622918, -0.56348131, -0.4964505 , -0.45227271, -0.37778964,
                  -0.32488907, -0.27234281, -0.18201988,  1.        ])            ],
          dtype=object)
        '''
        
        if template is None:
            raise NotImplementedError('Current implementation can only perform template analysis!')
        elif cutoff is None:
            raise NotImplementedError('Current implementation requires a cutoff to define on/off bins!')
        
        print('Setting up:')
        
        self.name = name
        self.savedir = savedir
        ensure_dir(savedir)
        
        self.verbose = verbose
        
        self.gamma = gamma
        self.cutoff = cutoff
        self.nside = nside
        self.min_dec_deg = min_dec_deg
        self.max_dec_deg = max_dec_deg
        
        #sindec band edges: default here taken from PSTracks in csky
        self.sindec_bins = np.unique(np.concatenate([
                             np.linspace(-1, -0.93, 4 + 1),
                             np.linspace(-0.93, -0.3, 10 + 1),
                             np.linspace(-0.3, 0.05, 9 + 1),
                             np.linspace(0.05, 1, 18 + 1) ]) )
        
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

            ## (test) subset of bins
            #new_sig_mask = self.sig_evs[self.logE_name]>=binned_dict.item()['logE_bins'][4]
            #self.sig_evs = self.sig_evs[new_sig_mask]
            #self.sig_relweights = self.sig_relweights[new_sig_mask]
            #self.logE_bins = binned_dict.item()['logE_bins'][5:]
            ##Get binned data from dictionary
            #self.binned_data = binned_dict.item()['binned_data'][5:] 
            #self.n_logE_bins = len(self.binned_data)
            
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
#=======================================((TEST))===================================================
        '''
        if kde is not None: # (test)
            print(f'Load KDE PDFs <-- {kde}')
            kde_dict = np.load(kde, allow_pickle=True)
            self.bg_accs = kde_dict.item()['bg']
            self.signal_accs = kde_dict.item()['signal']

            #Load template, create versions with acceptance and/or smoothing
            print(f'Load template <-- {template}')
            template = np.load(template, allow_pickle=True)         
            self.template = template.copy()
            self.create_template_pdf(med_sigs)
            
        else:
            self.create_signal_acc_kde()
            
            #Load template, create versions with acceptance and/or smoothing
            print(f'Load template <-- {template}')
            template = np.load(template, allow_pickle=True)         
            self.template = template.copy()
            self.create_template_pdf(med_sigs)
            
            self.create_bg_acc_kde()

            if self.savedir is not None:
                ensure_dir(f'{self.savedir}/kdes')
                kde_dict = {'bg': self.bg_accs,
                            'signal': self.signal_accs}
                savefile = f'{self.savedir}/kdes/{self.name}.kde_pdfs.nside{self.nside}.npy'
                i = 0
                #if save file already exists, append a number to the file name
                if os.path.exists(savefile):
                    print('Saved file of chosen name already exists!')
                    while os.path.exists(savefile):
                        savefile = f'{self.savedir}/kdes/{self.name}.kde_pdfs_{i}.nside{self.nside}.npy'
                        i += 1
                np.save(savefile, kde_dict)
                print(f'KDE PDF value arrays saved to --> {savefile}')
        '''
#=======================================((TEST))===================================================

        #Load template, create versions with acceptance and/or smoothing
        print(f'Load template <-- {template}')
        template = np.load(template, allow_pickle=True)         
        self.template = template.copy()
        self.create_template_pdf(med_sigs)

        #Get S and B PDFs for likelihood (properly normalized, I hope)
        #self.create_bg_acc_without_sindec()
        self.get_pdfs()

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
            
            #Mask events from GRL
            mask = np.isin(data['run'], self.grl)
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
        del new_arr
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

#    def create_scramble(self, path, verbose=None, seed=0, savedir=None):
#        """
#        Loads data, scrambles it, and bins it.
#        
#        Args:
#            path: path to directory containing data files or path to a data file
#            
#            verbose: True to show more output (Defaults to class's initited value)
#
#            seed: For RNG in scrambling.
#
#            savedir: directory to save binned scrambles (Default: None, which sets to self.savedir)
#            
#        """
#        if verbose is None:
#            verbose = self.verbose
#        if savedir is None:
#            savedir = self.savedir
#        ensure_dir(savedir)
#        #Make sure path is a directory or a file
#        assert (os.path.isdir(path) or os.path.isfile(path)), f"Expected path to directory or file, got: {path}"
#        
#        print(f'Loading, scrambling, and binning data from {path}')
#        if os.path.isdir(path):
#            #If path is a directory, make list of numpy files from that directory
#            files_like = path+'/*.npy'
#            file_list = sorted(glob(files_like))
#            
#        else: #path is one file
#            #If just one file, make it into a list
#            file_list = [path]
#            
#        ra_rng = np.random.default_rng(seed=seed)
#        binned_scramble = np.zeros((self.n_logE_bins, hp.nside2npix(self.nside)))
#        #Loop through files    
#        for file in file_list:
#            #Load the data
#            data = np.load(file)
#            #If using qtot instead of energy, add the logQtot field to the data array (just like for MC above)
#            if self.logE_name == 'logQtot':
#                if 'qtot_wdc' in data.dtype.names:
#                    q_name = 'qtot_wdc'
#                else:
#                    q_name = 'qtot'
#                    
#                #If binning data, make the same addition of logQtot field to the data array
#                #Add log10 of QTot to the sig array for consistency with logE
#                new_dtype = np.dtype(data.dtype.descr + [(self.logE_name, '<f8')])
#                new_arr = np.empty(data.shape, dtype=new_dtype)
#                for n in data.dtype.names:
#                    new_arr[n] = data[n]
#                new_arr[self.logE_name] = np.log10(data[q_name])
#
#                data = new_arr.copy()
#            
#            #Mask events from GRL
#            mask = np.isin(data['run'], self.grl)
#            data = data[mask]
#
#            #Randomize RAs
#            data['ra'] = ra_rng.random(size=len(data['ra'])) * 2.0 * np.pi
#            
#            #Get logE bin indices for each event (if only one bin, all zeros)
#            if self.n_logE_bins > 1:
#                e_inds = np.digitize(data[self.logE_name], self.logE_bins)
#            elif self.n_logE_bins == 1:
#                e_inds = np.zeros_like(data[self.logE_name], dtype=int)
#            else:
#                raise ValueError("Something is wrong with the number of energy bins!")
#
#            if verbose:
#                print(f'    {file} : ')
#            
#            #Loop through unique energy bin indices
#            for e in np.unique(e_inds):
#                if verbose:
#                    print(f'        Energy Bin {e+1}/{self.n_logE_bins}...', end=' ')
#                #Mask events in this energy bin
#                e_mask = (e_inds == e)
#                #bin the events from this file and add to the binned_data (initialized above as all zeros)
#                binned_scramble[e] = binned_scramble[e] + self.bin_data(data[e_mask])
#                
#                if verbose:
#                    print(' --> Done.')
#        
#        #Free up memory
#        del new_arr
#        del data
#        gc.collect()
#        
#        print('--> Data Loading Done. \n')
#        
#        #If loading and binning data, and if savedir is given, save
#        if savedir is not None:
#            binned_dict = {'logE_bins': self.logE_bins,
#                           'binned_scramble': binned_scramble}
#            savefile = f'{savedir}/{self.name}.binned_scramble.seed{seed}.nside{self.nside}.npy'
#            i = 0
#            #if save file already exists, append a number to the file name
#            if os.path.exists(savefile):
#                print('Saved file of chosen name already exists!')
#                while os.path.exists(savefile):
#                    if scramble:
#                        savefile = f'{savedir}/{self.name}.binned_scramble_{i}.seed{seed}.nside{self.nside}.npy'
#                    else:
#                        savefile = f'{savedir}/{self.name}.binned_data_{i}.nside{self.nside}.npy'
#                    i += 1
#            np.save(savefile, binned_dict)
#            print(f'Binned scramble saved to --> {savefile}')
#            
#        return
        
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
        #This is really just the declination rows of pixels, but avoiding rows with no background pixels near the poles
        start = 0
        stop = len(np.unique(self.bin_decs))-1
        for e in range(self.n_logE_bins):
            mask = (self.template_acc_smoothed[e] <= self.cutoff)
            num_bg_pix_per_dec = np.array([np.sum(mask & (self.bin_decs==d)) for d in np.unique(self.bin_decs)])
            start = np.maximum(start, np.nonzero(num_bg_pix_per_dec)[0][0])
            stop = np.minimum(stop, np.nonzero(num_bg_pix_per_dec)[0][-1])

        #Get edges of dec bands for selecting pixels below
        bin_edges = np.unique(self.bin_decs)[:-1] + np.diff(np.unique(self.bin_decs))/2
        bin_edges = np.r_[-np.pi/2, bin_edges[start:stop], np.pi/2]
    
        bg_acc_spline = np.empty(self.n_logE_bins, dtype=object)
        bg_hist = np.empty(self.n_logE_bins, dtype=object) # (test)
        for e in range(self.n_logE_bins):

            #bin_edges = np.arcsin(self.bb_bins[e]) # (test)
            #BG pdf using only "off" pixels as defined by cutoff
            mask = (self.template_acc_smoothed[e] <= self.cutoff)

            dOmega_corr = []
            dec_npix = []
            for i in np.arange(len(bin_edges)-1):
            #for d in np.unique(self.bin_decs):
                #Get pixel numbers from band of sindec
                pixels_in_band = hp.query_strip(self.nside, np.pi/2-bin_edges[i+1], np.pi/2-bin_edges[i])
                #Get boolean array: True if pixel from this sindec band is in signal region
                bool_array = np.isin(pixels_in_band, np.arange(hp.nside2npix(self.nside))[~mask])
                #Count number of signal pixels in this sindec band
                number_true = np.count_nonzero(bool_array)
                #Get correction factor: number of pixels in band / number of bg pixels in band
                corr = float(len(pixels_in_band)/float((len(pixels_in_band)-number_true)))
                
                dec_npix.append((pixels_in_band))
                dOmega_corr.append(corr)

            #Make hist
            h_counts_nocorr = hl.hist(np.sin(self.bin_decs[mask]), weights=self.binned_data[e, mask], bins=np.sin(bin_edges))
            #Correct hist counts
            counts_corr = h_counts_nocorr.values * np.array(dOmega_corr) #/ np.array(dec_npix)
            counts_corr = np.clip(counts_corr, a_min=np.min(counts_corr[np.nonzero(counts_corr)]), a_max=None) # (test)
            #counts_corr = np.where(counts_corr==0, 1, counts_corr) # (test)
            #New hist
            h_counts = hl.Hist(values=counts_corr, bins=h_counts_nocorr.bins)
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
        
    '''
    def create_bg_acc_without_sindec(self):
        """
        testing
        """
        #self.bg_perpix_nohist = np.zeros_like(self.binned_data)
        self.bg_nohist = np.zeros_like(self.binned_data)
        self.bg_perpix_bydec_nohist = np.zeros((self.n_logE_bins, len(np.unique(self.bin_decs))))
        for e in range(self.n_logE_bins):
            bg_mask = self.template_acc_smoothed[e]<=self.cutoff
            for i,d in enumerate(np.unique(self.bin_decs)):
                row_mask = self.bin_decs==d
                row_bg_sum = np.sum(self.binned_data[e,row_mask & bg_mask])
                n_pix = np.sum(row_mask)
                n_bg_pix = np.sum(row_mask & bg_mask)
                if n_bg_pix==0:
                    row_bg_perpix = 1.0
                else:
                    row_bg_perpix = row_bg_sum / n_bg_pix #n bg counts per pix in row ...Maybe should be median?
                    #row_bg_perpix = np.median(self.binned_data[e,row_mask & bg_mask])
                    
                    row_bg_perpix = np.maximum(np.round(row_bg_perpix), 1.0)

                #self.bg_perpix_nohist[e,row_mask] = row_bg_perpix
                self.bg_perpix_bydec_nohist[e,i] = row_bg_perpix
                num_bg_events = np.sum(self.binned_data[e, bg_mask]) * hp.nside2npix(self.nside) / np.sum(bg_mask) 
                self.bg_nohist[e, row_mask] = row_bg_perpix / num_bg_events / hp.nside2pixarea(self.nside)

        return
    '''

    '''
    def create_bg_acc_kde(self):
        """
        test
        """
        print('Creating background PDF with KDE...')
        self.bg_acc_kde = np.empty(self.n_logE_bins, dtype=object)
        self.bg_accs = np.zeros((self.n_logE_bins, hp.nside2npix(self.nside)))
        for e in range(self.n_logE_bins):
            mask = (self.template_acc_smoothed[e] <= self.cutoff)
            k = sp.stats.gaussian_kde(np.sin(self.bin_decs[mask]), weights=self.binned_data[e, mask])
            self.bg_acc_kde[e] = k
            self.bg_accs[e] = k(np.sin(self.bin_decs))           

        return
    '''
    
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

    '''
    def create_signal_acc_kde(self):
        """
        test
        """
        print('Creating signal PDF with KDE...')
        if self.n_logE_bins > 1:
            sig_ebin_inds = np.digitize(self.sig_evs[self.logE_name], self.logE_bins)
        elif self.n_logE_bins == 1:
            #sig_ebin_inds = np.zeros_like(self.sig_evs[self.logE_name], dtype=int)
            sig_ebin_inds = np.digitize(self.sig_evs[self.logE_name], self.logE_bins)
            sig_ebin_inds = np.where(sig_ebin_inds==1, 0, np.nan)
        else:
            raise ValueError("Something is wrong with the number of energy bins!")
            
        self.signal_acc_kde = np.empty(self.n_logE_bins, dtype=object)
        self.signal_accs = np.zeros((self.n_logE_bins, hp.nside2npix(self.nside)))
        for e in range(self.n_logE_bins):
            sig_ebin_mask = sig_ebin_inds == e
            k = sp.stats.gaussian_kde(np.sin(self.sig_evs['true_dec'][sig_ebin_mask]), weights=self.sig_relweights[sig_ebin_mask])
            self.signal_acc_kde[e] = k
            self.signal_accs[e] = k(np.sin(self.bin_decs))

        return
    '''
    
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
                #Get acceptance from spline using energy bin index and sindec
                #out = np.exp(self.signal_acc_spline[e](sindec))
                out = self.sig_hists[e].get_values(sindec) # (test)
                
            #If the signal_acc_spline has not been created...
            except AttributeError:
                print('Signal acceptance spline not yet created. Creating now... \n')
                self.create_signal_acc_spline()
                #out = np.exp(self.signal_acc_spline[e](sindec))
                out = self.sig_hists[e].get_values(sindec) # (test)
                
            return out
        
        #Same for background spline...
        elif acc == 'bg':
            try:
                #out = np.exp(self.bg_acc_spline[e](sindec))
                out = self.binned_bg_hists[e].get_values(sindec) # (test)
                
            except AttributeError:
                print('Background acceptance spline not yet created. Creating now... \n')
                self.create_bg_acc_spline()
                #out = np.exp(self.bg_acc_spline[e](sindec))
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
            #template_acc = template * self.signal_accs[e] # (test)  
            
            #Do the normalization (and smoothing)
            self.template_acc[e] = normalize_pdf(template_acc, sig=None)
            self.template_acc_smoothed[e] = normalize_pdf(template_acc, sig=smooth_sigs[e])
            #self.template_acc[e] = template.copy() # (test)
            #self.template_acc_smoothed[e] = template.copy() # (test)
            #self.template_acc_smoothed[e] = self.template_acc[e].copy() # (test)
        
        print('--> Template PDF-ization: Done. \n')
        
        return

    def multinomial_TS(self, n_sig, n, p_s, p_b, frac=1.0):
        """
        This function is used to calculate the multinomial TS:
        TS = 2 \sum_k \sum_i n_ik \ln\left( frac{f_k*n_s}{N_k}\left( frac{s_ik}{b_ik} - 1\right) + 1 \right)
        
        It is minimized for n_sig in the fitting functions.
        
        Args:
            n_sig: number of (signal) events
            
            n: array of event counts in pixels (via healpy)
            
            p_s: array of pixel-wise signal probabilities using signal/template PDf
            
            p_b: array of pixel-wise background probabilities using background PDF
            
            frac: fraction of signal events (for different energy bins)
            
        Returns: TS as calculated in the above equation.
        
        """
        #Here, N has length of ebins
        N = np.sum(n, axis=1)[:, np.newaxis]
        TS = 2.0 * np.sum( n * np.log( (frac*n_sig / N ) * (p_s / p_b - 1.0) + 1.0 ) )
        
        return TS

#I think the multinomial TS makes more theoretical sense for this purpose, so poisson is unused
#Both TSs produce the same results
#    def poisson_TS(self, n_sig, n, p_s, p_b, frac=1.0):
#        """
#        This function is used to calculate the poisson TS.
#        
#        It is minimized for n_sig in the fitting functions.
#        
#        Args:
#            n_sig: number of (signal) events
#            
#            n: array of event counts in pixels (via healpy)
#            
#            p_s: array of pixel-wise signal probabilities using signal/template PDf
#            
#            p_b: array of pixel-wise background probabilities using background PDF
#            
#        Returns: TS as calculated in the above equation.
#        
#        """
#        N = np.sum(n, axis=1)[:, np.newaxis]  
#        TS = 2.0 * np.sum( (frac*n_sig / N) * (p_b - p_s) + n * np.log( (frac*n_sig / N) * (p_s / p_b - 1.0) + 1.0 ) )
#        
#        return TS
    
    def get_pdfs(self, verbose=None):
        """
        Creates signal and background pdfs used in the test statistic calculation/minimization.
        
        `p_s` is the signal PDF; i.e., the template_pdf with pixels that do not pass the cutoff set to 0 and renormalized within dec bounds
        
        `p_b` is the background PDF; i.e., using the bg spline of 'off' bin counts and renormalized within dec bounds
        
        
        """
        if verbose is None:
            verbose = self.verbose
        
        if verbose:
            print('Creating signal and background PDFs for TS calculations...')
            
        self.p_s = np.zeros((self.n_logE_bins, hp.nside2npix(self.nside)))
        self.p_b = np.zeros((self.n_logE_bins, hp.nside2npix(self.nside)))

        #Get mask for signal pixels and dec bounds
        mask = (self.template_acc_smoothed > self.cutoff)
        dec_mask = (self.bin_decs<=np.radians(self.max_dec_deg)) & (self.bin_decs>=np.radians(self.min_dec_deg))
        
        #Signal PDF is template with acceptance and smoothing
        #Pixels below the cutoff are set to 0
        p_s = np.where(mask, self.template_acc_smoothed, 0.0)
        p_s /= np.sum(p_s[:,dec_mask], axis=1)[:, np.newaxis] * hp.nside2pixarea(self.nside) # [:,dec_mask]
        
        #Background PDF comes straight from the spline
        p_b = np.array([self.get_acc_from_spline(np.sin(self.bin_decs), b, acc='bg') for b in range(self.n_logE_bins)])
        #p_b = self.bg_accs.copy() # (test)
        #p_b = np.array([self.bg_nohist[b] for b in range(self.n_logE_bins)]) 
        #p_b = self.bg_perpix_nohist.copy()
        #p_b = np.clip(p_b, np.min(p_b[p_b>0]), None)
        p_b /= np.sum(p_b[:,dec_mask], axis=1)[:, np.newaxis] * hp.nside2pixarea(self.nside) # [:,dec_mask]
        
        self.p_s = p_s
        self.p_b = p_b
        
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
    
        #TS calculation uses pixels within dec bounds only (e.g., to exclude poles)
        n = self.counts[:,dec_mask].copy()
        p_s = self.p_s[:,dec_mask].copy()
        #p_s /= np.sum(p_s, axis=1)[:, np.newaxis]
        p_b = self.p_b[:,dec_mask].copy()
        #p_b /= np.sum(p_b, axis=1)[:, np.newaxis]
        frac = self.sig_acc_frac[:, np.newaxis]
        
        #Convenience function is of one variable ns and is the negative of the TS
        #Then, minimize...
        def min_neg_TS(ns):
            return -1.0 * self.multinomial_TS(ns, n, p_s, p_b, frac=frac)
        
        if verbose:
            print('Minimizing -TS...')
            
        #Using iminuit.minimize scipy-like interface
        #res = iminuit.minimize(min_neg_TS, 1, bounds=[(0,np.sum(n))])
        #fit_ns = res.x
        #fit_TS = -1.0 * res.minuit.fval
        
        #Using scipy
        #res = sp.optimize.minimize(min_neg_TS, 1, bounds=[(0,np.sum(n))])
        #fit_ns = res.x
        #fit_TS = -1.0 * res.fun
        
        #Using iminuit brute force scan to find minimum, then Migrad for better accuracy
        res = iminuit.Minuit(min_neg_TS, 1e2)
        res.print_level = 0
        #Limit ns to between 0 and number of events
        res.limits['ns'] = (0.0, np.sum(self.counts))
        res.scan()
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
            #For background, only consider "off" region
            mask = self.template_acc_smoothed[e] <= self.cutoff
            #Loop through unique pixel decs
            #for dec in unique_decs:
            for i,dec in enumerate(unique_decs):
                #Mask pixels in a row of dec
                dec_mask = (self.bin_decs == dec)
                #Get number of bg events in whole sky (for ebin e)
                num_bg_events = np.sum(self.binned_data[e, mask]) * hp.nside2npix(self.nside) / np.sum(mask) 
                #Get per pixel bg events for dec
                val = self.get_acc_from_spline(np.sin(dec), e, acc='bg') * num_bg_events * hp.nside2pixarea(self.nside) 
                #val = self.bg_perpix_bydec_nohist[e,i]
                
                #uvals, uinds = np.unique(self.bg_accs[e], return_index=True)
                #ordered_uvals = self.bg_accs[e].copy()[np.sort(uinds)]
                #val = ordered_uvals[np.argwhere(np.unique(self.bin_decs)[::-1] == dec)[0,0]]
                #val = val * num_bg_events * hp.nside2pixarea(self.nside) # (test)
                
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
        
            #Injection bins are choice from ON region (defined by template w/ acc and smoothing) within dec min/max bounds with replacement
            mask = (self.template_acc_smoothed[e] > self.cutoff)
            dec_mask = (self.bin_decs<=np.radians(self.max_dec_deg)) & (self.bin_decs>=np.radians(self.min_dec_deg))
            inj_choice = np.arange(hp.nside2npix(self.nside))[mask & dec_mask]
            
            #The probability of injection for each bin within the ON region includes acceptance but no smoothing
            inj_probs = self.template_acc[e,mask & dec_mask]
            inj_bins = rng_inj.choice(inj_choice, size=n_inj, p=inj_probs/np.sum(inj_probs))
            
            inj_evs = np.array([], dtype=self.sig_evs.dtype)
            ps = np.array([])
            #Loop through chosen injection pixels with the number of injections in that pixel
            bin_sindec_bin_inds = np.digitize(np.sin(self.bin_decs[inj_bins]), self.sindec_bins)
            for i in np.unique(bin_sindec_bin_inds):
                #mask MC events that are in the same sindec band as the pixel chosen for injection
                sig_dec_mask = np.digitize(np.sin(sig['true_dec']), self.sindec_bins) == i
                sig_in_dec = sig[sig_dec_mask]
                
                #pick random events from that sindec band selection with size equal to occurrences (weighted choice)
                choice_weights = self.sig_relweights[sig_ebin_mask][sig_dec_mask]
                n = np.sum(bin_sindec_bin_inds==i)
                #At full scale, with enough MC, replace should be set to False (so as to not select the same MC event twice)
                inj_evs = np.concatenate((inj_evs, rng_inj.choice(sig_in_dec, size=n, replace=False, p=choice_weights/np.sum(choice_weights), shuffle=False)))
                #Keep track of injection pixels from each dec band
                ps = np.concatenate((ps,inj_bins[bin_sindec_bin_inds==i]))
                
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
        
