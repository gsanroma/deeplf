import SimpleITK as sitk
import os
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
# import timeit
from copy import copy

float_type = np.float16
int_type = np.uint8

#
# Class image reader
#

class ImageReader(object):

    def __init__(self, train_dir, val_dir, img_suffix, lab_suffix, labels_superlist, patch_rad, search_rad):

        self.dist_max = search_rad
        self.patch_rad = patch_rad
        self.search_rad = search_rad
        self.labels_superlist = labels_superlist

        self.TrainImagesList, self.TrainLabelsList = [], []
        self.ValImagesList, self.ValLabelsList = [], []

        for mode in ('train', 'val'):

            mode_dir = train_dir if mode == 'train' else val_dir

            files = os.listdir(mode_dir)
            self.img_list = [f for f in files if f.endswith(img_suffix)]
            assert self.img_list, 'No image found'
            self.name_list = [f.split(img_suffix)[0] for f in self.img_list]
            self.lab_list = [f + lab_suffix for f in self.name_list]
            assert False not in [os.path.exists(os.path.join(mode_dir, f)) for f in self.lab_list]

            for idx_img, (img_file, lab_file) in enumerate(zip(self.img_list, self.lab_list)):
                print('READING SUBJECT: %s' % img_file)
                # read img & lab
                img_sitk = sitk.ReadImage(os.path.join(mode_dir, img_file))
                img = sitk.GetArrayFromImage(img_sitk).astype(float_type)
                lab_sitk = sitk.ReadImage(os.path.join(mode_dir, lab_file))
                lab = sitk.GetArrayFromImage(lab_sitk).astype(int_type)
                # compute cropping
                mask = np.ones(img.shape, dtype=np.bool)
                for label_id in [label_id_aux for labels_list in labels_superlist for label_id_aux in labels_list]:
                    if label_id == 0: continue
                    mask[lab == label_id] = False
                lab[mask] = 0
                # compute bboxes
                coord_axis = np.where(lab != 0)
                orig_min, orig_max = img.shape, [0, 0, 0]
                bbox_min = [min(orig_min[i], np.min(c_ax) - self.dist_max - patch_rad - search_rad - 1) for i, c_ax in enumerate(coord_axis)]
                bbox_max = [max(orig_max[i], np.max(c_ax) + self.dist_max + patch_rad + search_rad + 1) for i, c_ax in enumerate(coord_axis)]
                # crop
                if mode == 'train':
                    self.TrainImagesList.append(copy(img[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]]))
                    self.TrainLabelsList.append(copy(lab[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]]))
                else:
                    self.ValImagesList.append(copy(img[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]]))
                    self.ValLabelsList.append(copy(lab[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]]))
                del img, lab

#
# Class patch sampler
#

class PatchSampler(object):

    def __init__(self, rng, img_reader, labels_list):

        self.rng = rng
        self.TrainImagesList = img_reader.TrainImagesList
        self.TrainLabelsList = img_reader.TrainLabelsList
        self.ValImagesList = img_reader.ValImagesList
        self.ValLabelsList = img_reader.ValLabelsList
        self.patch_rad = img_reader.patch_rad
        self.search_rad = img_reader.search_rad
        self.dist_max = img_reader.dist_max

        self.n_lab = len(labels_list)
        self.labels_list = list(set(labels_list) - {0})

        assert self.n_lab >=1, 'labels list must be non-empty'

        labels_aux = [label_id for labels_list_aux in img_reader.labels_superlist for label_id in labels_list_aux]
        label_intersect = list(set(labels_list) & set(labels_aux))
        assert len(label_intersect) == len(labels_list), 'some labels not included in image reader'

        self.TrainDistList, self.ValDistList = [], []
        self.TrainLocBorder, self.TrainIdxBorder = np.array([], dtype=np.int), np.array([], dtype=np.int)
        self.ValLocBorder, self.ValIdxBorder = np.array([], dtype=np.int), np.array([], dtype=np.int)
        self.TrainIdxInside, self.TrainLocInside, self.TrainProbInside = np.array([], dtype=np.int), np.array([], dtype=np.int), np.array([], dtype=np.int)
        self.ValIdxInside, self.ValLocInside, self.ValProbInside = np.array([], dtype=np.int), np.array([], dtype=np.int), np.array([], dtype=np.int)

        for mode in ('train', 'val'):

            LabelsList = self.TrainLabelsList if mode == 'train' else self.ValLabelsList

            print('Sampling locations %s' % mode)
            for i, labmap in enumerate(LabelsList):
                aux = np.zeros((self.n_lab + 1,) + labmap.shape, np.float64)
                aux2 = np.zeros(labmap.shape, dtype=np.bool)
                for j in range(self.n_lab):
                    aux3 = labmap == self.labels_list[j]
                    distance_transform_edt(aux3, distances=aux[j])
                    np.bitwise_or(aux3, aux2, aux2)
                    del aux3
                distance_transform_edt(np.invert(aux2), distances=aux[-1])
                del aux2
                aux[aux == 0.] = np.inf
                if mode == 'train':
                    self.TrainDistList += [np.min(aux, axis=0).astype(float_type)]
                else:
                    self.ValDistList += [np.min(aux, axis=0).astype(float_type)]
                del aux

            DistList = self.TrainDistList if mode == 'train' else self.ValDistList

            # Sampling border locations
            for i, distmap in enumerate(DistList):
                Iaux = np.where(distmap.ravel() == 1.)[0]
                if mode == 'train':
                    self.TrainLocBorder = np.hstack((self.TrainLocBorder, Iaux))
                    self.TrainIdxBorder = np.hstack((self.TrainIdxBorder, i * np.ones(Iaux.shape, dtype=np.int)))
                else:
                    self.ValLocBorder = np.hstack((self.ValLocBorder, Iaux))
                    self.ValIdxBorder = np.hstack((self.ValIdxBorder, i * np.ones(Iaux.shape, dtype=np.int)))
                del Iaux

            # Sampling inside locations
            for i, distmap in enumerate(DistList):
                Iaux = np.where(np.logical_and(distmap.ravel() > 1., distmap.ravel() <= float(self.dist_max)))[0]
                if mode == 'train':
                    self.TrainLocInside = np.hstack((self.TrainLocInside, Iaux))
                    self.TrainIdxInside = np.hstack((self.TrainIdxInside, i * np.ones(Iaux.shape, dtype=np.int)))
                    self.TrainProbInside = np.hstack((self.TrainProbInside, 1. - ((distmap.ravel()[Iaux] - 1.) / float(self.dist_max) )))
                else:
                    self.ValLocInside = np.hstack((self.ValLocInside, Iaux))
                    self.ValIdxInside = np.hstack((self.ValIdxInside, i * np.ones(Iaux.shape, dtype=np.int)))
                    self.ValProbInside = np.hstack((self.ValProbInside, 1. - ((distmap.ravel()[Iaux] - 1.) / float(self.dist_max) )))
                del Iaux

            #
            # keeps track of epoch which is composed of border samples in training set
            if mode == 'train':
                self.n_total = self.TrainLocBorder.size
                self.remaining_samples = np.arange(self.n_total)
                self.epoch = 0.


    #
    # Sample a mini-batch
    #

    def sample(self, num_samples, num_neighbors0, fract_inside, xvalset='train', update_epoch=True):

        ImagesList = self.TrainImagesList
        LabelsList = self.TrainLabelsList
        DistList = self.TrainDistList
        LocBorder, LocInside = self.TrainLocBorder, self.TrainLocInside
        IdxBorder, IdxInside = self.TrainIdxBorder, self.TrainIdxInside
        ProbInside = self.TrainProbInside
        if xvalset == 'val':
            ImagesList = self.ValImagesList
            LabelsList = self.ValLabelsList
            DistList = self.ValDistList
            LocBorder, LocInside = self.ValLocBorder, self.ValLocInside
            IdxBorder, IdxInside = self.ValIdxBorder, self.ValIdxInside
            ProbInside = self.ValProbInside

        n_inside = np.int(np.rint(num_samples * fract_inside))
        n_border = np.int(num_samples - n_inside)

        # choose samples
        if xvalset == 'train' and update_epoch:

            # in case training set make sure not to repeat samples within an epoch
            if n_border > self.remaining_samples.size:  # if done with epoch, start over
                self.epoch = np.floor(self.epoch) + 1.
                self.remaining_samples = np.arange(self.n_total)

            Irem = self.rng.choice(range(self.remaining_samples.size), size=n_border, replace=False)
            Iborder = self.remaining_samples[Irem]
            self.remaining_samples = np.delete(self.remaining_samples, Irem)

            self.epoch = np.floor(self.epoch) + (np.float(self.n_total - self.remaining_samples.size) / np.float(self.n_total))

        else:  # we do not mind repeating samples in validation set
            Iborder = self.rng.choice(range(LocBorder.size), size=n_border, replace=False)

        # sample inside samples according to probability Pinside
        Iinside = self.rng.choice(range(LocInside.size), size=n_inside, replace=False, p=ProbInside / ProbInside.sum()) if n_inside > 0 else []
        # Iinside = self.rng.choice(range(LocInside.size), size=n_inside, replace=False) if n_inside > 0 else []

        #
        # sample patches

        p0, p1, p2 = np.meshgrid(np.arange(-self.patch_rad, self.patch_rad + 1),
                                 np.arange(-self.patch_rad, self.patch_rad + 1),
                                 np.arange(-self.patch_rad, self.patch_rad + 1))
        s0, s1, s2 = np.meshgrid(range(-self.search_rad, self.search_rad+1),
                                 range(-self.search_rad, self.search_rad+1),
                                 range(-self.search_rad, self.search_rad+1))
        aux = np.vstack((s0.ravel(), s1.ravel(), s2.ravel()))
        Iaux = np.logical_not(np.all(aux == 0, axis=0))
        s0, s1, s2 = s0.ravel()[Iaux], s1.ravel()[Iaux], s2.ravel()[Iaux]

        patch_len = p0.size
        search_len = s0.size

        num_neighbors = min(search_len, num_neighbors0)

        TargetPatches = np.zeros((num_samples, 1, patch_len))
        TargetVotes = np.zeros((num_samples, 1))
        AtlasPatches = np.zeros((num_samples, search_len, patch_len))
        AtlasVotes = np.zeros((num_samples, search_len))

        isamp = 0
        ibase = 0

        # time_aux = timeit.default_timer()

        for mode in ('border', 'inside'):

            Isamp = Iborder if mode == 'border' else Iinside
            Idx = IdxBorder if mode == 'border' else IdxInside
            Loc = LocBorder if mode == 'border' else LocInside
            n_samp = n_border if mode == 'border' else n_inside

            if n_samp == 0: continue
            Itrack = np.array([])  # keep track of correspondences between Isamp and Patch data

            for i, (Image, Label) in enumerate(zip(ImagesList, LabelsList)):

                Iimg = np.where(Idx[Isamp] == i)[0]

                for loc in Loc[Isamp[Iimg]]:

                    l0, l1, l2 = np.unravel_index(loc, Image.shape)
                    p_coord = np.array([l0 + p0.ravel(), l1 + p1.ravel(), l2 + p2.ravel()])

                    Ipatch = np.ravel_multi_index(p_coord, Image.shape)
                    TargetPatches[isamp, 0] = Image.ravel()[Ipatch]
                    TargetVotes[isamp, 0] = Label.ravel()[loc]

                    for j, s in enumerate(np.array(zip(s0.ravel(), s1.ravel(), s2.ravel()))):
                        p_coord_aux = p_coord + s[:, np.newaxis]

                        Ipatch = np.ravel_multi_index(p_coord_aux, Image.shape)
                        AtlasPatches[isamp, j] = Image.ravel()[Ipatch]
                        AtlasVotes[isamp, j] = Label[l0 + s[0], l1 + s[1], l2 + s[2]]

                    isamp += 1

                Itrack = np.hstack((Itrack, Iimg))

            # errcheck
            rs, rn, rp = self.rng.randint(n_samp), self.rng.randint(search_len), self.rng.randint(patch_len)
            img = Idx[Isamp[rs]]
            itrack = np.where(Itrack == rs)[0][0]
            loc = np.array(np.unravel_index(Loc[Isamp[rs]], ImagesList[img].shape)) + np.array([s0.ravel()[rn], s1.ravel()[rn], s2.ravel()[rn]]) + \
                  np.array([p0.ravel()[rp], p1.ravel()[rp], p2.ravel()[rp]])
            assert AtlasPatches[ibase + itrack, rn, rp] == ImagesList[img][loc[0], loc[1], loc[2]], 'error sampling neighboring patches'
            loc = np.array(np.unravel_index(Loc[Isamp[rs]], ImagesList[img].shape)) + np.array([s0.ravel()[rn], s1.ravel()[rn], s2.ravel()[rn]])
            assert AtlasVotes[ibase + itrack, rn] == LabelsList[img][loc[0], loc[1], loc[2]], 'error sampling neighboring votes'
            loc = np.array(np.unravel_index(Loc[Isamp[rs]], ImagesList[img].shape)) + np.array([p0.ravel()[rp], p1.ravel()[rp], p2.ravel()[rp]])
            assert TargetPatches[ibase + itrack, 0, rp] == ImagesList[img][loc[0], loc[1], loc[2]], 'error sampling central patches'
            loc = np.array(np.unravel_index(Loc[Isamp[rs]], ImagesList[img].shape))
            assert TargetVotes[ibase + itrack, 0] == LabelsList[img][loc[0], loc[1], loc[2]], 'error sampling central votes'
            if mode == 'inside':
                assert np.allclose(ProbInside[Isamp[rs]], 1. - ((DistList[img][loc[0], loc[1], loc[2]] - 1.) / float(self.dist_max)), 1e-3), 'error sampling probabilities'
            ibase = isamp

        # print('(sampler) spent %0.3f secs.' % (timeit.default_timer() - time_aux))

        AtlasPatches2 = np.zeros((num_samples, num_neighbors, patch_len))
        AtlasVotes2 = np.zeros((num_samples, num_neighbors))

        for isamp in range(num_samples):

            I1 = np.where(AtlasVotes[isamp] == TargetVotes[isamp])[0]
            I0 = np.array(list(set(range(search_len)) - set(I1)))

            if I1.size == 0:
                print('this should not happen')

            if I0.size > num_neighbors // 2 and I1.size > num_neighbors // 2:
                r = num_neighbors % 2
                n0 = num_neighbors // 2 + r * int(self.rng.rand() > 0.5)
                n1 = num_neighbors - n0
            elif I0.size >= I1.size:
                n1 = I1.size
                n0 = num_neighbors - n1
            else:
                n0 = I0.size
                n1 = num_neighbors - n0

            Ineigh0 = self.rng.choice(range(I0.size), size=n0, replace=False) if I0.size > 0 else []
            Pat0, Vot0 = (AtlasPatches[isamp, I0[Ineigh0]], AtlasVotes[isamp, I0[Ineigh0]]) if I0.size > 0 else (
                np.zeros((0, patch_len)), np.zeros((0,)))

            Ineigh1 = self.rng.choice(range(I1.size), size=n1, replace=False) if I1.size > 0 else []
            Pat1, Vot1 = (AtlasPatches[isamp, I1[Ineigh1]], AtlasVotes[isamp, I1[Ineigh1]]) if I1.size > 0 else (
                np.zeros((0, patch_len)), np.zeros((0,)))

            AtlasPatches2[isamp] = np.concatenate((Pat0, Pat1), axis=0)
            AtlasVotes2[isamp] = np.concatenate((Vot0, Vot1), axis=0)

        return (TargetPatches, TargetVotes, AtlasPatches2, AtlasVotes2)

