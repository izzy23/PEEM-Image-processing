import logging
import numpy as np
import scipy.ndimage as ndi
import scipy.fftpack
import skimage
from skimage.util import crop
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class ImageStack:
    """ImageStack basis class"""
    #I COMMENTED THIS - MAYBE REMOVE
    #CollectionViewer = skimage.viewer.CollectionViewer
    _attr_numpy_list = []  #list for attribute should be loaded as numpy
    _attr_save_list = []  # list of attribute to be saved in a json

    def __init__(self, frames, type_cast=None):
        """
        Initialise an ImageStack intance by a sequence of frames
        or (N,P,Q) array
        """
        self.frames = np.asarray(frames, dtype=type_cast)
        self.mask = np.ones(self.frames.shape, dtype=bool)
        self._viewer = None

    @property
    def nimgs(self):
        """Number of images stored"""
        return len(self.frames)

    def resize(self, shape, order=1):
        """Resize the entire array"""
        frames = np.moveaxis(self.frames, 0, 2)
        resized = tf.resize(frames,
                            shape,
                            order,
                            mode='reflect',
                            preserve_range=True)
        self._backup()
        self.frames = np.rollaxis(resized, 2)

    def save(self, name):
        """Save array into numpy binary file"""

        print("Saving data to file {}.npy".format(name))
        np.save(name + '.npy', self.frames)
        self._save_attrs(name)

    def load(self, name):
        """"Load frames from numpy binary file"""

        print("Loading data from file {}.npy".format(name))
        self._load_attrs(name)
        self.frames = np.load(name + '.npy')

    @property
    def imshape(self):
        """Shape of each frame"""
        return self.frames[0].shape

    def set_cicular_mask(self, rel, mode="union"):
        """Apply a circular mask to the existing mask
        Parameters
        ----------
        rel: float
            relative radius

        mode: string
            mode of operation. "union" or "replace"
        """
        mask = get_circular_mask(self.imshape, rel)
        if mode == "union":
            self.mask = mask & self.mask
        if mode == "replace":
            self.mask = mask

    @staticmethod
    #i commented this !!!! - maybe don't
    #def _view(data):
    #    """Show all images using skimage.imshow"""
    #
    #    viewer = skimage.viewer.CollectionViewer(data)
    #    viewer.show()

    def view(self):
        """View frames"""

        self._view(self)

    def rotate(self, ang):
        """
        Rotate the stack

        Parameters:
            angs int or list angle of rotations. If a integer is passed
            rotate all frames
        """
        self._backup()
        if isinstance(ang, int):
            frames = np.moveaxis(self.frames, 0, 2)
            self.frames = np.rollaxis(
                tf.rotate(frames, ang, preserve_range=True), 2)
        else:
            for i, a in enumerate(ang):
                self.frames[i] = tf.rotate(self.frames[i],
                                           a,
                                           preserve_range=True)

    def _backup(self):
        """Copy data to self._frames"""
        self._frames = self.frames.copy()

    def _restore(self):
        """Restore from self._frames"""
        self.frames = self._frames

    def __getitem__(self, index):
        """Allow direct slicing"""
        if isinstance(index, tuple) and len(index) == 2:
            return ImageStack(
                self.frames.__getitem__((slice(None, None, None), ) + index))
        else:
            return self.frames.__getitem__(index)

    def __setitem__(self, index, value):
        self.frames.__setitem__(index, value)

    def __len__(self):
        return len(self.frames)

    def __repr__(self):
        string = "<ImageStack at {} with frames: \n".format(id(self))
        string += self.frames.__repr__()
        string += "\n>"
        return string

    def _save_attrs(self, name):
        """Seralize with json. Save np array as lists"""
        json_dict = {}
        for attr in self._attr_save_list:
            value = self.__getattribute__(attr)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            json_dict[attr] = value

        if json_dict:
            with open(name + '.json', "w") as fp:
                json.dump(json_dict, fp)

    def _load_attrs(self, name):
        """Load list as np array"""
        try:
            with open(name + '.json') as fp:
                json_dict = json.load(fp)
        except FileNotFoundError:
            print("No json file found")
            return

        for key, value in json_dict.items():
            if isinstance(value, list) and key in self._attr_numpy_list:
                value = np.array(value)
            self.__setattr__(key, value)

    def _clear_data(self):
        """
        Clear the underlying data
        """
        del self.frames
        del self.mask
        self.frames = None
        self.mask = None

class DriftCorrector(ImageStack):
    """A class for correcting drift between frames"""

    DEFAULT_DRIFT_MODE = 'one-pass'  # Defult mode of operation
    INPLACE_DRIFT = False

    def __init__(self,
                 frames,
                 ref=0,
                 crop_setting=((0, 0), (0, 0)),
                 super_sample=4):
        """
        Initialise an instance of DriftCorrector

        :pararm frames: list of arrays of frames to be corrected

        :param ref: Index of the reference frame

        :param crop_setting: Setting of crop of axis 0 and aixs 1. 
          See skimages' crop function.

        :param super_sample: Ratio of super sampling during correction.

        We define the drift being *the tranlation to move the image
        back to the reference*.

        Axis orders are numpy standard (Y, X) for images
        """
        super().__init__(frames)
        self.super_sample = super_sample
        self.corrected_frames = None
        self.ref = ref
        self._refimg = None  # Reference image with improved qaulity
        self.crop_setting = crop_setting
        self._flag_drift_calculated = False
        self.drifts = np.zeros((len(self.frames), 2))
        self.drifts_applied = np.zeros((len(self.frames), 2))

        # Storeage for errors
        # See doc of register_translation
        self.phase_diffs = np.zeros(len(self.frames))
        self.errors = np.zeros(len(self.frames))  # See register_translation

    def view_cropped(self, block=False, **kwargs):
        """
        View cropped frames for calculating translations.
        Calls the get_croped_single method with given kwargs
        """
        cpd = np.array(
            [self.get_croped_single(i, **kwargs) for i in range(len(self))])

        show_images_series(cpd, block=block)

    def get_croped_single(self,
                          index,
                          ignore_drift=False,
                          use_corrected=False,
                          copy=False):
        """
        Get a cropped image given by its index.
        This takes acount any drift it has in self.drifts.

        :param index: Index of the frame to be cropped
        :param ignore_drift: Ignore any existin drft
        :param use_corrected: If true will use corrected image.
          e.g to check the quality of drft correction
        """

        # If requesting reference image use self.refimg
        if use_corrected:
            ignore_drift = True
            frame = self.corrected_frames[index]
        else:
            frame = self.frames[index]

        roi = CropRegion(self.crop_setting, image=frame)

        # Do not take the current drift of the frame into account
        if not ignore_drift:
            # Offset the corp region be the negative mount of the drift
            # The drift vector is from the drifted image to the reference
            # Hence applying the negative will move the crop region to
            # centre onto the freature
            roi.offset = -self.drifts[index]
        res = roi.get_cropped_image(copy=copy)
        return res

    def get_mean_cropped(self):
        """
        Note that the image is linkely to be an array of unit16
        hence we need to convert the refernece image to float
        for averaing
        Generate mean of the cropped image.
        This can be used as the reference for iterative drift unit
        things are converged
        """

        refimg = self.get_croped_single(self.ref).astype(np.float64,
                                                         casting='unsafe',
                                                         copy=True)
        for n, i in enumerate(range(self.nimgs)):
            if n == self.ref:
                continue
            img = self.get_croped_single(i)
            refimg += img

        mean = (refimg / self.nimgs)
        mean.astype(self.frames.dtype)
        return mean

    def calc_drifts(self, refimg=None, sigma=None, ref_otf_update=False):
        """
        Compute the drift vectors
        """

        if refimg is None:
            refimg = self.get_croped_single(self.ref)
            indices = range(self.ref + 1, len(self))
        else:
            indices = range(self.ref, len(self))

        # If we update the refernce energy, make sure that it is a copy
        if ref_otf_update is True:
            refimg = refimg.copy()

        refimg = filter_image(refimg)
        dc = 0  # Counter for drift direction
        for i in indices:
            img = self.get_croped_single(i, copy=False)
            img = filter_image(img, sigma=sigma)
            #show_images_series([img])
            # Registor translation takes the current drift into account
            #dft, error, phase_diff = register_translation(
            dft, error, phase_diff = skimage.registration.phase_cross_correlation(
                refimg, img, self.super_sample)
            

            # Drift is a accumulation effect so we update the future frames
            self.drifts[i:] += dft
            self.errors[i] = error
            self.phase_diffs[i] = phase_diff

            # Update the reference by taking weighted average
            # Only if it is requested
            dc += 1
            if ref_otf_update is True:
                img = self.get_croped_single(i)
                refimg = dc / (dc + 1) * refimg + img / (dc + 1)

        indices = reversed(range(self.ref))
        for i in indices:
            img = self.get_croped_single(i)
            img = filter_image(img, sigma=sigma)
            # Registor translation takes the current drift into account
            #dft, error, phase_diff = register_translation(
            dft, error, phase_diff = skimage.registration.phase_cross_correlation(
                refimg, img, self.super_sample)
            self.drifts[:i + 1] += dft
            self.errors[i] = error
            self.phase_diffs[i] = phase_diff
            dc += 1
            if ref_otf_update is True:
                img = self.get_croped_single(i)
                refimg = dc / (dc + 1) * refimg + img / (dc + 1)

        # Make sure the reference image has drift 0
        self.drifts -= self.drifts[self.ref]

        self._refimg = refimg
        self._flag_drift_calculated = True
        return self.drifts

    def apply_correction(self, inplace=None):
        """
        Apply drifts to the frames.
        Corrected frames are saved in self.corrected_frames
        """
        if inplace is None:
            inplace = self.INPLACE_DRIFT

        drifts = self.drifts
        fm = self.frames
        if inplace is False:
            res = []
            for d, im in zip(drifts, fm):
                res.append(ndi.shift(im, d))

            res = np.array(res)
            self.corrected_frames = res
        else:
            for d, im in zip(drifts, fm):
                # Apply in place
                ndi.shift(im, d, output=im)
            # Dump the drift to the delta array then reset to zero
            # Since the image has been shifted
            self.drifts_applied += self.drifts
            self.drifts[...] = 0
            res = self.frames
            self.corrected_frames = self.frames

        return res

    @property
    def drifts_from_initial(self):
        """Return the correct drift form the initial image"""
        return self.drifts_applied + self.drifts

    apply_drift_corr = apply_correction

    def correct(self, mode=None, sigma=None):
        """Calculate drifts and apply correction.
        Corrected frames are returned.
        """
        if mode is None:
            mode = self.DEFAULT_DRIFT_MODE
        # Do two passes
        if mode != 'iter':
            self.calc_drifts(sigma=sigma)
            if mode == 'two-pass':
                ref = self.get_mean_cropped()
                self.calc_drifts(refimg=ref, sigma=sigma)
        elif mode == 'iter':
            self.calc_drift_iter()

        self.apply_correction()
        return self.corrected_frames

    def calc_drift_iter(self, tol=0.5, verbose=True, max_iter=10):
        """
        Iteratively correct unit the change of the drift vector
        dimishes
        """

        def _drift_residual(d1, d2):
            return np.linalg.norm(d1 - d2, axis=-1).max()

        last_drifts = np.ones((self.nimgs, 2))
        count = 1
        refimg = None
        # Compute the drift until it is converged
        while True:
            drifts = self.calc_drifts(refimg=refimg).copy()
            refimg = self._refimg
            residual = _drift_residual(last_drifts, drifts)
            if residual < tol:
                print("# {}/{} residual: {:.3f}, converged".format(
                    count, max_iter, residual))
                break
            if count > max_iter:
                break
            else:
                if verbose:
                    print("# {}/{} residual: {:.3f},"
                          " unconverged".format(count, max_iter, residual))
                last_drifts = drifts
            count += 1
        if verbose:
            print("Correction completed")

    def _clear_data(self):
        super(DriftCorrector, self)._clear_data()
        del self.corrected_frames
        self.corrected_frames = None  