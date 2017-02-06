import numpy as np
from scipy.stats import zscore
from sklearn import linear_model
from copy import copy
# from skimage.transform import downscale_local_mean
# from time import sleep

float_type = np.float16
int_type = np.uint8

def ms_coord(l, dim_ms, dim_orig):
    return [int(round(float(dim_ms[0] * l[0]) / float(dim_orig[0]))),
            int(round(float(dim_ms[1] * l[1]) / float(dim_orig[1]))),
            int(round(float(dim_ms[2] * l[2]) / float(dim_orig[2])))]


def preliminary_checks(target, AtlasImages, mask, patch_rad, search_rad, fusion_rad):

    assert len(target.shape) == 3 and len(mask.shape) == 3 and len(AtlasImages.shape) == 4, 'incorrect dimensions'
    assert len(patch_rad) == 3 and len(search_rad) == 3 and len(fusion_rad) == 3, 'incorrect radius vector size'
    assert mask.dtype == np.bool, 'mask must be boolean'
    assert all([a == b for a, b in zip(target.shape, mask.shape)]), 'target and mask must be same size'
    assert all([a == b for a, b in zip(target.shape, AtlasImages.shape[1:])]), 'target and atlas must be same size'

def labfus(target, AtlasImages, AtlasLabels, mask, patch_rad, search_rad, fusion_rad, ss, method, reg_param, normType, embedder=None):#, MSTarget=[], MSAtlas=[]):#, atlas_is_target=False):

    preliminary_checks(target, AtlasImages, mask, patch_rad, search_rad, fusion_rad)

    uniqueLabelsList = np.unique(AtlasLabels[0])

    p0, p1, p2 = np.meshgrid(np.arange(-patch_rad[0], patch_rad[0]+1), np.arange(-patch_rad[1], patch_rad[1]+1), np.arange(-patch_rad[2], patch_rad[2]+1))
    s0, s1, s2 = np.meshgrid(np.arange(-search_rad[0], search_rad[0]+1), np.arange(-search_rad[1], search_rad[1]+1), np.arange(-search_rad[2], search_rad[2]+1))
    f0, f1, f2 = np.meshgrid(np.arange(-fusion_rad[0], fusion_rad[0]+1), np.arange(-fusion_rad[1], fusion_rad[1]+1), np.arange(-fusion_rad[2], fusion_rad[2]+1))

    # if atlas_is_target:
    #     # REMOVE CENTRAL PATCH WHEN ATLAS == TARGET
    #     aux = np.vstack((s0.ravel(), s1.ravel(), s2.ravel()))
    #     Iaux = np.logical_not(np.all(aux == 0, axis=0))
    #     s0, s1, s2 = s0.ravel()[Iaux], s1.ravel()[Iaux], s2.ravel()[Iaux]

    n_lab = uniqueLabelsList.size
    img_shape = target.shape
    patch_len = p0.size
    search_len = s0.size
    fusion_len = f0.size
    n_atlas = AtlasImages.shape[0]

    I = np.argwhere(mask)

    # I = np.array([I[9999]])

    TargetProbs = np.zeros(img_shape + (n_lab,), dtype=np.float32)

    for step, (i0, i1, i2) in enumerate(I):

        if step % 1000 == 0:
            print 'step %d of %d' % (step // 1000, I.shape[0] // 1000)

        p_coord = np.array([i0 + p0.ravel(), i1 + p1.ravel(), i2 + p2.ravel()])
        f_coord = np.array([i0 + f0.ravel(), i1 + f1.ravel(), i2 + f2.ravel()])

        Iaux = np.ravel_multi_index(p_coord, img_shape)
        target_patch = target.ravel()[Iaux]

        # target_ms_patch = np.zeros((len(MSTarget), patch_len), dtype=np.float32)

        # print("going to get ms target patches")

        # # Get multi-scale patches
        # for j, MSImage in enumerate(MSTarget):
        #     lms = ms_coord([i0, i1, i2], MSImage.shape, target.shape)
        #     p_coord_ms = np.array([lms[0] + p0.ravel(), lms[1] + p1.ravel(), lms[2] + p2.ravel()])
        #     Ipatch = np.ravel_multi_index(p_coord_ms, MSImage.shape)
        #     target_ms_patch[j] = MSImage.ravel()[Ipatch]

        AtlasPatches = np.zeros((n_atlas, search_len, patch_len), dtype=np.float32)
        AtlasVotes = np.zeros((n_atlas, search_len, fusion_len), dtype=int_type)
        # AtlasMSPatches = np.zeros((n_atlas, len(MSAtlas), patch_len), dtype=np.float32)

        # print("going to get atlas patches")

        for i in range(n_atlas):

            atlas_img_i = AtlasImages[i].view()
            atlas_lab_i = AtlasLabels[i].view()

            for j, s in enumerate(np.array(zip(s0.ravel(), s1.ravel(), s2.ravel()))):

                p_coord_aux = p_coord + s[:, np.newaxis]
                Iaux = np.ravel_multi_index(p_coord_aux, img_shape)
                AtlasPatches[i, j] = atlas_img_i.ravel()[Iaux]

                f_coord_aux = f_coord + s[:, np.newaxis]
                Iaux = np.ravel_multi_index(f_coord_aux, img_shape)
                AtlasVotes[i, j] = atlas_lab_i.ravel()[Iaux]

            # # Get multi-scale patches
            # for j, MSImage in enumerate(MSAtlas):
            #     lms = ms_coord([i0, i1, i2], MSImage[i].shape, target.shape)
            #     p_coord_ms = np.array([lms[0] + p0.ravel(), lms[1] + p1.ravel(), lms[2] + p2.ravel()])
            #     Ipatch = np.ravel_multi_index(p_coord_ms, MSImage[i].shape)
            #     AtlasMSPatches[i, j] = MSImage[i].ravel()[Ipatch]

        # print("going to select atlas patches")

        AtlasPatches2 = AtlasPatches.view()
        AtlasPatches2.shape = (n_atlas * search_len, patch_len)
        AtlasVotes2 = AtlasVotes.view()
        AtlasVotes2.shape = (n_atlas * search_len, fusion_len)

        # print("going to check errors")

        # errcheck
        ra, rs, rp, rf = np.random.randint(n_atlas), np.random.randint(search_len), np.random.randint(patch_len), np.random.randint(fusion_len)
        assert target[p_coord[0, rp], p_coord[1, rp], p_coord[2, rp]] == target_patch[rp], 'target patch not correct at coord (%d, %d, %d)' % (i0, i1, i2)
        aux = AtlasImages[ra].view()
        aux2 = AtlasLabels[ra].view()
        assert AtlasPatches2[ra * search_len + rs, rp] == aux[p_coord[0, rp] + s0.ravel()[rs], p_coord[1, rp] + s1.ravel()[rs], p_coord[2, rp] + s2.ravel()[rs]], \
            'atlas patch not correct at coord (%d, %d, %d)' % (i0, i1, i2)
        assert AtlasVotes2[ra * search_len + rs, rf] == aux2[f_coord[0, rf] + s0.ravel()[rs], f_coord[1, rf] + s1.ravel()[rs], f_coord[2, rf] + s2.ravel()[rs]], \
            'atlas vote not correct at coord (%d, %d, %d)' % (i0, i1, i2)
        # assert StructSim2[ra * search_len + rs] == StructSim[ra, rs], 'struct sim not correct at coord (%d, %d, %d)' % (i0, i1, i2)
        # for i in range(len(MSTarget)):
        #     lms = ms_coord([i0, i1, i2], MSTarget[i].shape, target.shape)
        #     lms += np.array([p0.ravel()[rp], p1.ravel()[rp], p2.ravel()[rp]])
        #     assert target_ms_patch[i, rp] == MSTarget[i][lms[0], lms[1], lms[2]], 'error in target multiscale patches'
        #     assert AtlasMSPatches[ra, i, rp] == MSAtlas[i][ra][lms[0], lms[1], lms[2]], 'error in atlas multiscale patches'

        # print("going to normalize patches")

        # # Normalize multiscale patches
        # if normType == 'zscore':
        #     AtlasMSPatches = zscore(AtlasMSPatches, axis=2)
        #     target_ms_patch= zscore(target_ms_patch, axis=1)
        # elif normType == 'l2':
        #     AtlasMSPatches = AtlasMSPatches / np.linalg.norm(AtlasMSPatches, ord=2, axis=2)[..., np.newaxis]
        #     target_ms_patch = target_ms_patch / np.linalg.norm(target_ms_patch, ord=2, axis=1)[..., np.newaxis]
        # AtlasMSPatches[np.logical_not(np.isfinite(AtlasMSPatches))] = 0.
        # target_ms_patch[np.logical_not(np.isfinite(target_ms_patch))] = 0.

        # print("goingto reshape ms patches")

        # # multiscale patches
        # target_ms_patch = np.reshape(target_ms_patch, (len(MSTarget) * patch_len, ))
        # AtlasMSPatches = np.reshape(AtlasMSPatches, (n_atlas, 1, len(MSAtlas) * patch_len))
        # AtlasMSPatches = np.tile(AtlasMSPatches, (1, search_len, 1))
        # AtlasMSPatches = np.reshape(AtlasMSPatches, (n_atlas * search_len, len(MSAtlas) * patch_len))

        # print("select best atlas patches with ss %f, n_atlas %d" % (ss, n_atlas))

        #
        # Patch pre-selection

        # WHY NOT ATLASPATCHES2 ??

        mu_PATCHES = np.mean(AtlasPatches, axis=2, dtype=np.float32)
        std_PATCHES = np.std(AtlasPatches, axis=2, dtype=np.float32)

        mu_patch = target_patch.mean(dtype=np.float32)
        std_patch = target_patch.std(dtype=np.float32)

        StructSim = ((2. * mu_patch * mu_PATCHES) / (mu_patch ** 2. + mu_PATCHES ** 2. + np.finfo(np.float32).eps)) * \
                    ((2. * std_patch * std_PATCHES) / (std_patch ** 2. + std_PATCHES ** 2. + np.finfo(np.float32).eps))

        StructSim2 = StructSim.view()
        StructSim2.shape = (n_atlas * search_len,)

        # Choose best patches
        Iok = []
        saux = ss
        while len(Iok) < n_atlas:
            Iok = np.argwhere(StructSim2 > saux).ravel()
            saux -= 0.2

        # print("going to normalize patches")

        # Normalize best patches
        if normType == 'l2':
            AtlasPatchesNorm = AtlasPatches2[Iok] / np.linalg.norm(AtlasPatches2[Iok], ord=2, axis=1)[:, np.newaxis]
            target_patch_norm = (target_patch / np.linalg.norm(target_patch, ord=2, axis=0))
        elif normType == 'zscore':
            AtlasPatchesNorm = zscore(AtlasPatches2[Iok], axis=1)
            target_patch_norm = zscore(target_patch)
        else:# normType == 'none':
            AtlasPatchesNorm = AtlasPatches2[Iok]
            target_patch_norm = target_patch
        AtlasPatchesNorm[np.logical_not(np.isfinite(AtlasPatchesNorm))] = 0.
        target_patch_norm[np.logical_not(np.isfinite(target_patch_norm))] = 0.

        # print("going to concatenate ms patches")

        # # Concatenate multiscale patches
        # AtlasPatchesNorm = np.concatenate((AtlasPatchesNorm, AtlasMSPatches[Iok]), axis=1)
        # target_patch_norm = np.concatenate((target_patch_norm, target_ms_patch), axis=0)

        # print("going to compute weights")

        # here we should apply learned transforms

        if method == 'nlwv':
            if embedder is None:
                l2_norms = np.sum((AtlasPatchesNorm - target_patch_norm[np.newaxis, :])**2, axis=1)
            else:
                l2_norms = embedder.get_l2n(target_patch_norm[np.newaxis, np.newaxis, ...], AtlasPatchesNorm[np.newaxis, ...])
            weights = np.exp(- l2_norms / (l2_norms.min() + reg_param))
            if embedder is not None:
                weights = weights.squeeze()
        elif method == 'nlbeta':
            if embedder is None:
                l2_norms = np.sum((AtlasPatchesNorm - target_patch_norm[np.newaxis, :]) ** 2, axis=1)
                weights = np.exp(- l2_norms * reg_param)
            else:
                weights = embedder.get_w(target_patch_norm[np.newaxis, np.newaxis, ...], AtlasPatchesNorm[np.newaxis, ...], reg_param)
                weights = weights.squeeze()
        elif method == 'deeplf':
            weights = embedder.get_w_val(target_patch_norm[np.newaxis, np.newaxis, ...], AtlasPatchesNorm[np.newaxis, ...])
            weights = weights.squeeze()
        elif method == 'lasso':
            clf = linear_model.Lasso(alpha=reg_param, fit_intercept=False)
            clf.fit(AtlasPatchesNorm.T, target_patch_norm)
            weights = clf.coef_

        # print("going to fuse labels")

        for i, (ff0, ff1, ff2) in enumerate(f_coord.T):
            probs = [np.sum(weights * (AtlasVotes2[Iok, i] == cur_lab).astype(np.float32)) for cur_lab in uniqueLabelsList]
            probs /= (np.sum(probs) + np.finfo(float).eps)
            TargetProbs[ff0, ff1, ff2, :] += np.float32(probs)

    TargetLabels = uniqueLabelsList[np.argmax(TargetProbs, axis=3)]
    TargetProbs /= np.expand_dims(np.sum(TargetProbs, axis=3) + np.finfo(TargetProbs.dtype).eps, axis=3)

    # Majority voting
    LabelStats = np.zeros((len(uniqueLabelsList),) + target.shape)
    for i, cur_lab in enumerate(uniqueLabelsList):
        LabelStats[i] = np.sum((AtlasLabels == cur_lab).astype(np.int16), axis=0)
    TargetMV = uniqueLabelsList[np.argmax(LabelStats, axis=0)]

    TargetLabels[~mask] = TargetMV[~mask]

    return (TargetLabels, TargetProbs, uniqueLabelsList)


if __name__ == "__main__":

    import SimpleITK as sitk
    import argparse
    # from sys import exit
    # import os

    parser = argparse.ArgumentParser(description='Performs label fusion using one of the weighted voting methods')
    parser.add_argument("--target_img", type=str, nargs=1, required=True, help="target image")
    parser.add_argument("--atlas_img_list", type=str, nargs='+', required=True, help="list of atlas images")
    parser.add_argument("--atlas_lab_list", type=str, nargs='+', required=True, help="list of atlas labelmaps")
    parser.add_argument("--out_file", type=str, nargs=1, required=True, help="output fusion results")
    parser.add_argument("--probabilities", action="store_true", help="compute segmentation probabilities")
    parser.add_argument("--patch_radius", type=str, nargs=1, help="image patch radius (default 2x2x2)")
    parser.add_argument("--search_radius", type=str, nargs=1, help="search neighborhood radius (default 3x3x3)")
    parser.add_argument("--fusion_radius", type=str, nargs=1, help="neighborhood fusion radius (default 1x1x1)")
    parser.add_argument("--struct_sim", type=float, nargs=1, default=[0.9], help="structural similarity threshold (default 0.9)")
    parser.add_argument("--normalization", type=str, nargs=1, help="patch normalization type [L2 | zscore | none] (default zscore)")
    parser.add_argument("--method", type=str, nargs=1, required=True, help="nlwv, nlbeta, deeplf, lasso")
    parser.add_argument("--regularization", type=float, nargs=1, default=[0.001], help="(nlwv, lasso, nlbeta) regularization parameter for label fusion method")
    parser.add_argument("--load_net", type=str, nargs=1, help="(deeplf) file with the deep neural network")
    parser.add_argument("--label_grp", type=int, nargs='+', help="(optional) list of label ids to segment")
    parser.add_argument("--consensus_thr", type=float, nargs=1, default=[0.9], help="(optional) consensus threshold for creating segmentation mask (default 0.9)")
    parser.add_argument("--classification_metrics", type=str, nargs=1, help="compute classification metrics in non-consensus region (needs target labelmap)")
    # parser.add_argument("--downscale_factors", type=int, nargs='+', default=[], help="(deeplf) list of downscale factors for multiscale patches")
    # parser.add_argument("--num_threads", type=int, nargs=1, default=[1], help="(optional) number of threads (default 1)")

    args = parser.parse_args()
    # args = parser.parse_args('--atlas_img_list /Users/gsanroma/DATA/deeplf/models_sata/registrations/1002_3/1000_3_brainWarped.nii.gz /Users/gsanroma/DATA/deeplf/models_sata/registrations/1002_3/1001_3_brainWarped.nii.gz --target_img /Users/gsanroma/DATA/deeplf/data/sata_mini_val/1002_3_brain.nii.gz --classification_metrics /Users/gsanroma/DATA/deeplf/data/sata_mini_val/1002_3_glm.nii.gz --atlas_lab_list /Users/gsanroma/DATA/deeplf/models_sata/registrations/1002_3/1000_3_glmWarped.nii.gz /Users/gsanroma/DATA/deeplf/models_sata/registrations/1002_3/1001_3_glmWarped.nii.gz --method deeplf --search_radius 1x1x1 --fusion_radius 1x1x1 --struct_sim 0.900000 --label_grp 23 30 --load_net /Users/gsanroma/DATA/deeplf/models_sata/Group0/kk/grp0_epch0_007.dat --out_file /Users/gsanroma/DATA/deeplf/models_sata/Labfus/kk/1002_3_dlf_grp0.nii.gz'.split())
    # args = parser.parse_args('--atlas_img_list /Users/gsanroma/DATA/deeplf/models_adni/registrations/013_S_4731/006_S_4192_brainWarped.nii.gz /Users/gsanroma/DATA/deeplf/models_adni/registrations/013_S_4731/020_S_0213_brainWarped.nii.gz /Users/gsanroma/DATA/deeplf/models_adni/registrations/013_S_4731/136_S_0579_brainWarped.nii.gz --target_img /Users/gsanroma/DATA/deeplf/data/mini_val7/013_S_4731_brain.nii.gz --atlas_lab_list /Users/gsanroma/DATA/deeplf/models_adni/registrations/013_S_4731/006_S_4192_labelsWarped.nii.gz /Users/gsanroma/DATA/deeplf/models_adni/registrations/013_S_4731/020_S_0213_labelsWarped.nii.gz /Users/gsanroma/DATA/deeplf/models_adni/registrations/013_S_4731/136_S_0579_labelsWarped.nii.gz --method deeplf --search_radius 1x1x1 --fusion_radius 1x1x1 --struct_sim 0.900000 --label_id 1 --load_net /Users/gsanroma/DATA/deeplf/models_adni/Label0/kk/l0_e0_002.dat --out_file /Users/gsanroma/DATA/deeplf/models_adni/Labfus/kk/013_S_4731_dlf_l1.nii.gz'.split())
    #                          '/Users/gsanroma/DATA/deeplf/est_deeplf_S4_val0/regs/002_S_0729/002_S_0295_brainWarped.nii.gz '
    #                          '/Users/gsanroma/DATA/deeplf/est_deeplf_S4_val0/regs/002_S_0729/002_S_0413_brainWarped.nii.gz '
    #                          '/Users/gsanroma/DATA/deeplf/est_deeplf_S4_val0/regs/002_S_0729/002_S_0782_brainWarped.nii.gz '
    #                          '/Users/gsanroma/DATA/deeplf/est_deeplf_S4_val0/regs/002_S_0729/002_S_0816_brainWarped.nii.gz '
    #                          '/Users/gsanroma/DATA/deeplf/est_deeplf_S4_val0/regs/002_S_0729/002_S_1070_brainWarped.nii.gz '
    #                          '/Users/gsanroma/DATA/deeplf/est_deeplf_S4_val0/regs/002_S_0729/002_S_1155_brainWarped.nii.gz '
    #                          '/Users/gsanroma/DATA/deeplf/est_deeplf_S4_val0/regs/002_S_0729/002_S_1261_brainWarped.nii.gz '
    #                          '--target_img /Users/gsanroma/DATA/deeplf/data/adni_val0/002_S_0729_brain.nii.gz '
    #                          '--atlas_lab_list '
    #                          '/Users/gsanroma/DATA/deeplf/est_deeplf_S4_val0/regs/002_S_0729/002_S_0295_labelsWarped.nii.gz '
    #                          '/Users/gsanroma/DATA/deeplf/est_deeplf_S4_val0/regs/002_S_0729/002_S_0413_labelsWarped.nii.gz '
    #                          '/Users/gsanroma/DATA/deeplf/est_deeplf_S4_val0/regs/002_S_0729/002_S_0782_labelsWarped.nii.gz '
    #                          '/Users/gsanroma/DATA/deeplf/est_deeplf_S4_val0/regs/002_S_0729/002_S_0816_labelsWarped.nii.gz '
    #                          '/Users/gsanroma/DATA/deeplf/est_deeplf_S4_val0/regs/002_S_0729/002_S_1070_labelsWarped.nii.gz '
    #                          '/Users/gsanroma/DATA/deeplf/est_deeplf_S4_val0/regs/002_S_0729/002_S_1155_labelsWarped.nii.gz '
    #                          '/Users/gsanroma/DATA/deeplf/est_deeplf_S4_val0/regs/002_S_0729/002_S_1261_labelsWarped.nii.gz '
    #                          '--method deeplf 0.01 '
    #                          '--patch_radius 2x2x2 '
    #                          '--search_radius 1x1x1 '
    #                          '--fusion_radius 1x1x1 '
    #                          '--struct_sim 0.9 '
    #                          '--normalization zscore '
    #                          '--out_file /Users/gsanroma/DATA/deeplf/est_deeplf_S4_val0/002_S_0729_dlf.nii.gz '
    #                          '--load_net /Users/gsanroma/DATA/deeplf/models_fold1/model001.model '.split())

    patch_rad = [2, 2, 2]
    if args.patch_radius is not None:
        patch_rad = [int(f) for f in args.patch_radius[0].split('x')]

    search_rad = [3, 3, 3]
    if args.search_radius is not None:
        search_rad = [int(f) for f in args.search_radius[0].split('x')]

    fusion_rad = [1, 1, 1]
    if args.fusion_radius is not None:
        fusion_rad = [int(f) for f in args.fusion_radius[0].split('x')]

    norm_type = 'zscore'
    if args.normalization is not None:
        norm_type = args.normalization[0]

    downscale_factors = []

    embedder = None
    if args.method[0] == 'deeplf':
        from DeepMetricLearning import DeepML
        numpy_rng = np.random.RandomState(1234)
        print('loading network from file ' + args.load_net[0])
        embedder = DeepML.fromfile(numpy_rng, args.load_net[0])
        # read label fusion parameters from net
        patch_rad = [embedder.patch_rad] * 3
        norm_type = embedder.patch_norm
        downscale_factors = embedder.ds_fact_list
        # check label id is included in net's labels list
        if args.label_grp is not None:
            assert len(set(args.label_grp) & set(embedder.labels_list)) == len(set(args.label_grp)), 'label id not included in nets labels list'
    # else:
    #     print("reading IDNET!!")
    #     from DeepMetricLearning import IdNet
    #     embedder = IdNet()

    AtlasImages_list, AtlasLabels_list = [], []
    # MSAtlas_superlist = [[] for i in range(len(downscale_factors))]
    for i, (atlas_img, atlas_lab) in enumerate(zip(args.atlas_img_list, args.atlas_lab_list)):
        print('Reading imgs and labs of atlas %d (out of %d)' % (i+1, len(args.atlas_img_list)))
        aux_sitk = sitk.ReadImage(atlas_img)
        AtlasImages_list += [sitk.GetArrayFromImage(aux_sitk).astype(float_type)]
        aux_sitk = sitk.ReadImage(atlas_lab)
        AtlasLabels_list += [sitk.GetArrayFromImage(aux_sitk).astype(int_type)]
        # for i, ds_factor in enumerate(downscale_factors):
        #     MSAtlas_superlist[i].append(downscale_local_mean(AtlasImages_list[-1], (int(ds_factor), int(ds_factor), int(ds_factor))).astype(float_type))
    n_atlases = len(AtlasImages_list)

    target_sitk = sitk.ReadImage(args.target_img[0])
    target_img = sitk.GetArrayFromImage(target_sitk).astype(float_type)
    img_shape = target_img.shape
    # MSTarget_list = []
    # msimg_shape_list = []
    # for i, ds_factor in enumerate(downscale_factors):
    #     MSTarget_list.append(downscale_local_mean(target_img, (int(ds_factor), int(ds_factor), int(ds_factor))).astype(float_type))
    #     msimg_shape_list.append(MSTarget_list[-1].shape)

    AtlasImages = copy(np.array(AtlasImages_list))
    AtlasLabels = copy(np.array(AtlasLabels_list))

    # free memory
    for img, lab in zip(AtlasImages_list, AtlasLabels_list): del img, lab
    del AtlasImages_list, AtlasLabels_list

    # If label_grp given -> set to zero the rest of labels
    if args.label_grp is not None:
        mask = np.ones(AtlasLabels.shape, dtype=np.bool)
        for label_id in args.label_grp:
            mask[AtlasLabels == label_id] = False
        AtlasLabels[mask] = 0

    #
    # crop areas with no labels

    # compute bboxes
    coord_axis = np.where(np.any(AtlasLabels != 0, axis=0))
    aux_min = [np.min(c_ax) for c_ax in coord_axis]
    aux_max = [np.max(c_ax)  for c_ax in coord_axis]
    orig_min, orig_max = img_shape, [0, 0, 0]
    # if downscale_factors:
    #     ms_min = [c - patch_rad[i] - 1 for i, c in enumerate(ms_coord(aux_min, msimg_shape_list[-1], img_shape))]
    #     ms_max = [c + patch_rad[i] + 1 for i, c in enumerate(ms_coord(aux_max, msimg_shape_list[-1], img_shape))]
    #     orig_min = ms_coord(ms_min, img_shape, msimg_shape_list[-1])
    #     orig_max = ms_coord(ms_max, img_shape, msimg_shape_list[-1])
    bbox_min = [min(orig_min[i], np.min(c_ax) - patch_rad[i] - search_rad[i] - 1) for i, c_ax in enumerate(coord_axis)]
    bbox_max = [max(orig_max[i], np.max(c_ax) + patch_rad[i] + search_rad[i] + 1) for i, c_ax in enumerate(coord_axis)]
    # msbbox_min_list, msbbox_max_list = [], []
    # for msimg_shape in msimg_shape_list:
    #     msbbox_min_list.append(ms_coord(bbox_min, msimg_shape, img_shape))
    #     msbbox_max_list.append(ms_coord(bbox_max, msimg_shape, img_shape))

    # crop single scale images
    AtlasImages_crop = copy(AtlasImages[:, bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]])
    AtlasLabels_crop = copy(AtlasLabels[:, bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]])
    del AtlasImages, AtlasLabels
    target_img_crop = copy(target_img[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]])
    del target_img
    img_crop_shape = target_img_crop.shape

    # # crop multiscale images
    # msimg_crop_shape_list = []
    # MSTarget_crop_list, MSAtlas_crop_superlist = [], []
    # for i, (MSTarget, MSAtlas_list) in enumerate(zip(MSTarget_list, MSAtlas_superlist)):
    #     msbbox_min, msbbox_max = msbbox_min_list[i], msbbox_max_list[i]
    #     MSTarget_crop_list.append(MSTarget[msbbox_min[0]:msbbox_max[0], msbbox_min[1]:msbbox_max[1], msbbox_min[2]:msbbox_max[2]])
    #     del MSTarget
    #     msimg_crop_shape_list.append(MSTarget_crop_list[-1].shape)
    #     MSAtlas_crop_superlist.append([])
    #     for MSAtlas in MSAtlas_list:
    #         MSAtlas_crop = MSAtlas[msbbox_min[0]:msbbox_max[0], msbbox_min[1]:msbbox_max[1], msbbox_min[2]:msbbox_max[2]]
    #         MSAtlas_crop_superlist[-1].append(MSAtlas_crop)
    #         del MSAtlas
    #     del MSAtlas_list
    # del MSAtlas_superlist, MSTarget_list

    # Set segmentation mask according to consensus threshold
    u_lab = np.unique(AtlasLabels_crop[0])
    Freq_maps = np.zeros((u_lab.size,) + img_crop_shape, dtype=np.float32)
    for i, l in enumerate(u_lab):
        Freq_maps[i] += np.sum((AtlasLabels_crop == l).astype(np.float32), axis=0) / float(n_atlases)
    # assert np.allclose(Freq_maps.sum(0), 1.0)  ### CHECK WHAT HAPPENS!!!
    mask = np.max(Freq_maps, axis=0) < args.consensus_thr[0]

    # atlas_is_target = False
    # if args.atlas_img_list[0] == args.target_img[0]:
    #     from scipy.ndimage.morphology import distance_transform_edt
    #     atlas_is_target = True
    #     # THIS IS FOR LABEL FUSION WITH ATLAS = TARGET
    #     aux = np.array([distance_transform_edt(AtlasLabels == ulab) for ulab in np.unique(AtlasLabels)])
    #     aux[aux == 0.] = np.inf
    #     distmap = np.min(aux, axis=0)
    #     mask = distmap.squeeze() < 2.
    # else:
    #     AtlasMasks = AtlasLabels > 0
    #     map_a = np.logical_not(np.all(AtlasMasks, axis=0))  # not all ones
    #     map_b = np.all(np.logical_not(AtlasMasks), axis=0)  # all zeros
    #     mask = np.logical_and(map_a, np.logical_not(map_b))  # not all ones and not all zeros

    print('Performing label fusion')

    TargetLabels_crop, TargetProbs_crop, uniqueLabelsList = labfus(target_img_crop, AtlasImages_crop, AtlasLabels_crop, mask, patch_rad, search_rad,
                                                                   fusion_rad, args.struct_sim[0], args.method[0], args.regularization[0], norm_type,
                                                                   embedder)#, MSTarget_crop_list, MSAtlas_crop_superlist)

    print('Writing results')

    # pad cropped image
    TargetLabels = np.zeros(img_shape, dtype=int_type)
    TargetLabels[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]] = TargetLabels_crop

    aux_sitk = sitk.GetImageFromArray(TargetLabels)
    aux_sitk.CopyInformation(target_sitk)
    sitk.WriteImage(aux_sitk, args.out_file[0])

    if args.classification_metrics is not None:

        from os import extsep

        print('Reading target labels for classification metrics')
        aux_sitk = sitk.ReadImage(args.classification_metrics[0])
        gtr_lab = sitk.GetArrayFromImage(aux_sitk).astype(int_type)
        gtr_lab_crop = gtr_lab[bbox_min[0]:bbox_max[0], bbox_min[1]:bbox_max[1], bbox_min[2]:bbox_max[2]]

        f = open(args.out_file[0].split(extsep, 1)[0] + '.txt', 'w')

        labels_list = args.label_grp if args.label_grp is not None else list(set(np.unique(TargetLabels)) - {0})
        tp, tn, fp, fn = 0., 0., 0., 0.
        for label in labels_list:
            est = TargetLabels_crop[mask] == label
            gtr = gtr_lab_crop[mask] == label
            tp += np.logical_and(est, gtr).mean()
            tn += np.logical_and(np.logical_not(est), np.logical_not(gtr)).mean()
            fp += np.logical_and(est, np.logical_not(gtr)).mean()
            fn += np.logical_and(np.logical_not(est), gtr).mean()

        sens = float(tp) / (float(tp) + float(fn))
        spec = float(tn) / (float(tn) + float(fp))
        acc = float(tp + tn) / (tp + fp + tn + fn)
        f.write('%f,%f,%f\n' % (sens, spec, acc))

        f.close()


                # if args.probabilities:
    #
    #     out_dir = os.path.dirname(args.out_file[0])
    #     out_name = os.path.basename(args.out_file[0]).split(os.extsep, 1)[0]
    #     prob_dir = os.path.join(out_dir, out_name)
    #
    #     if not os.path.exists(prob_dir):
    #         os.makedirs(prob_dir)
    #
    #     for i in range(len(uniqueLabelsList)):
    #
    #         prob_path = os.path.join(prob_dir, 'prob%d.nii.gz' % (uniqueLabelsList[i]))
    #         targetprobs = TargetProbs[:, :, :, i]
    #         aux_sitk = sitk.GetImageFromArray(targetprobs)
    #         aux_sitk.CopyInformation(target_sitk)
    #         sitk.WriteImage(aux_sitk, prob_path)



