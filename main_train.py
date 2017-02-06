
import timeit
from time import sleep
import argparse
import numpy
# import PatchSamplerAtlas
import PatchSampler as ps
from scipy.optimize import minimize
from DeepMetricLearning import DeepML, IdNet, normalize_patches
from deeplf_display import write_stats4
import os
from label_fusion import label_fusion, get_label_fusion_params
from subprocess import call, check_output
from sys import platform, exit
from shutil import rmtree
import csv
from warnings import warn

scale_vars = []  # [l2n, y]: global variables for scale estimation

sigmoid = lambda x: 1 / (1 + numpy.exp(-x))

def softmax(w):

    e = numpy.exp(w)
    return e / numpy.sum(e, axis=1, keepdims=True)

def xentropy(beta):

    y = scale_vars[1]
    smax = softmax(-beta * scale_vars[0])
    true_dist = y / numpy.sum(y, axis=1, keepdims=True)
    entr = numpy.sum(true_dist * numpy.log(smax), axis=1)
    return -numpy.mean(entr)

def labfus(beta):

    y = scale_vars[1]
    w = -beta * scale_vars[0]
    e = numpy.exp(w)
    lf = numpy.sum(e * y, axis=1) / numpy.sum(e, axis=1)
    return -numpy.mean(numpy.log(lf))

def labfus_acc(beta, l2n, y):

    w = -beta * l2n
    e = numpy.exp(w)
    lf = numpy.sum(e * y, axis=1) / numpy.sum(e, axis=1)
    return numpy.mean(lf > 0.5)


parser = argparse.ArgumentParser(description='Trains deep model for manifold learning')

# dataset arguments
parser.add_argument("--train_dir", type=str, nargs=1, required=True, help="directory of training images")
parser.add_argument("--val_dir", type=str, nargs=1, required=True, help="directory of validation images")
parser.add_argument("--img_suffix", type=str, nargs=1, required=True, help="suffix of images")
parser.add_argument("--lab_suffix", type=str, nargs=1, required=True, help="suffix of labelmaps")

# segment arguments
parser.add_argument("--num_threads_deeplf", type=int, nargs=1, default=[4], help='number of threads for deep labfus (default 4)')
parser.add_argument("--no_competing_labfus", action='store_true', help="do not perform nlwv nlbeta labfus")
parser.add_argument("--parallel_labfus", action='store_true', help="each label segmented individually (otherwise grouped as per label_group)")
parser.add_argument("--reg_dir", type=str, nargs=1, required=True, help='directory with registrations from train to train/validation images')
parser.add_argument("--reg_img_suffix", type=str, nargs=1, required=True, help='suffix of registered images')
parser.add_argument("--reg_lab_suffix", type=str, nargs=1, required=True, help='suffix of registered labelmaps')

# sampler argument
parser.add_argument("--fract_inside", type=float, nargs=1, default=[.5], help="(optional) fraction of inside patches to sample (default .5)")
parser.add_argument("--num_neighbors", type=int, nargs=1, default=[50], help="(optional) number of neighbors for each target patch (default 50)")

# store arguments
parser.add_argument("--model_name", type=str, nargs=1, required=True, help="name of the model (used to name output files)")
parser.add_argument("--store_folder", type=str, nargs=1, required=True, help="folder to store model to")
parser.add_argument("--label_group", type=int, nargs='+', action='append', required=True, help="group of labels defining a model")

# training arguments
parser.add_argument("--num_epochs", type=float, nargs=1, required=True, help="number of epochs")
parser.add_argument("--train_batch_size", type=int, nargs=1, required=True, help="size of a training batch")
parser.add_argument("--est_batch_size", type=int, nargs=1, required=True, help="size of an estimation batch")
parser.add_argument("--display_frequency", type=int, nargs=1, required=True, help="frequency (# iterations) for display")
parser.add_argument("--segment_frequency", type=int, nargs=1, required=True, help="frequency (# iterations) for segmenting")

# optimization arguments
parser.add_argument("--learning_rate", type=float, nargs=1, required=True, help="learning rate")
parser.add_argument("--L1_reg", type=float, nargs=1, default=[0.0], help="(optional) L1 regularization strength (default 0.0)")
parser.add_argument("--L2_reg", type=float, nargs=1, default=[0.0], help="(optional) L2 regularization strength (default 0.0)")
parser.add_argument("--sparse_reg", type=float, nargs=1, default=[0.0], help="(optional) sparsity regularization strength (default 0.0)")
parser.add_argument("--kl_rho", type=float, nargs=1, default=[0.05], help="(optional) reference value for KL divergence for sparsity (default 0.05)")

# network arguments
parser.add_argument("--num_units", type=int, nargs=1, help="(optional) number of units")
parser.add_argument("--num_hidden_layers", type=int, nargs=1, help="(optional) number of layers")
parser.add_argument("--activation", type=str, nargs=1, default=['relu'], help="(optional) [relu|tanh|sigmoid|none] activation of hidden layers (default relu)")
parser.add_argument("--batch_norm", action='store_true', help="batch normalization")
# net params from file
parser.add_argument("--load_net", type=str, nargs=1, help="(optional) file with the initial model parameters")

# method arguments
parser.add_argument("--patch_rad", type=int, nargs=1, default=[2], help="(optional) image patch radius (default 2)")
parser.add_argument("--patch_norm", type=str, nargs=1, default=['zscore'], help="(optional) patch normalization type [zscore|l2|none]")
parser.add_argument("--search_rad", type=int, nargs=1, default=[3], help="(optional) search neighborhood radius (default 3)")

args = parser.parse_args()
# args = parser.parse_args(''
#                          '--train_dir /Users/gsanroma/DATA/deeplf/data/sata_mini_train '
#                          # '--train_dir /Users/gsanroma/DATA/deeplf/data/mini_train7 '
#                          '--val_dir /Users/gsanroma/DATA/deeplf/data/sata_mini_val '
#                          # '--val_dir /Users/gsanroma/DATA/deeplf/data/mini_val7 '
#                          '--img_suffix _brain.nii.gz --lab_suffix _glm.nii.gz '
#                          # '--img_suffix _brain.nii.gz --lab_suffix _labels.nii.gz '
#                          #
#                          '--reg_dir /Users/gsanroma/DATA/deeplf/sata20/registrations '
#                          # '--reg_dir /Users/gsanroma/DATA/deeplf/adni35/registrations '
#                          '--reg_img_suffix _brainWarped.nii.gz '
#                          '--reg_lab_suffix _glmWarped.nii.gz '
#                          # '--reg_lab_suffix _labelsWarped.nii.gz '
#                          # '--parallel_labfus '
#                          #
#                          '--fract_inside 0.5 '
#                          '--model_name kk '
#                          '--store_folder /Users/gsanroma/DATA/deeplf/models_sata '
#                          # '--store_folder /Users/gsanroma/DATA/deeplf/models_adni '
#                          # '--label_group 1 2 '
#                          '--label_group 31 --label_group 32 --label_group 36 --label_group 37 '
#                          # '--label_group 23 30 '
#                          '--num_epochs 30 '
#                          '--train_batch_size 50 '
#                          '--est_batch_size 500 '
#                          '--display_frequency 1 '
#                          '--segment_frequency 1 '
#                          '--learning_rate 0.0002 '
#                          '--L2_reg 0. '
#                          '--sparse_reg 2e-3 '
#                          #
#                          # '--sim exp '
#                          '--num_units 100 '
#                          '--num_hidden_layers 1 '
#                          '--activation relu '
#                          '--batch_norm '
#                          #
#                          # '--load_net /Users/gsanroma/DATA/deeplf/models/model000.model '
#                          #
#                          '--num_neighbors 50 '
#                          '--patch_rad 2 '
#                          '--search_rad 3 '
#                          '--patch_norm zscore '.split())

if platform == 'darwin':
    is_hpc = False
else:
    is_hpc = True

python_path = os.path.join(os.environ['HOME'], 'anaconda', 'envs', 'sitkpy', 'bin', 'python')
code_path = os.path.join(os.environ['HOME'], 'CODE')
evalseg_path = os.path.join(code_path, 'scripts_py', 'evaluate_segmentations.py')

numpy_rng = numpy.random.RandomState(1234)

n_in = ((args.patch_rad[0] * 2 + 1) ** 3)

kl_rho = numpy.asarray(args.kl_rho[0], dtype='float32')  # for training and obtaining kl_div
kl_rho.shape = (1, 1)

# correct batch_norm if no hidden layers
batch_norm_flag = args.batch_norm if args.num_hidden_layers[0] > 0 else False

# label fusion parameters
aux = get_label_fusion_params(args.val_dir[0], args.img_suffix[0], args.reg_dir[0], args.reg_img_suffix[0], args.reg_lab_suffix[0])
val_name_list, val_path_list, train_img_path_superlist, train_lab_path_superlist = aux

print('... building idnet')

id_net = IdNet()

#
# Create sampler and embedder list

img_reader = ps.ImageReader(args.train_dir[0], args.val_dir[0], args.img_suffix[0], args.lab_suffix[0],
                            args.label_group, args.patch_rad[0], args.search_rad[0])

# patch_sampler = PatchSamplerAtlas.PatchSamplerAtlas(numpy_rng, args.train_dir[0], args.val_dir[0], args.img_suffix[0], args.lab_suffix[0],
#                                                     args.reg_dir[0], args.reg_img_suffix[0], args.reg_lab_suffix[0],
#                                                     args.label_group, args.patch_rad[0], args.search_rad[0], args.cert_threshold[0])
# labels_superlist = patch_sampler.labels_superlist

labels_superlist = args.label_group
N_models = len(labels_superlist)

sampler_list = []
embedder_list = []
train_model_list = []

for i, labels_list in enumerate(labels_superlist):

    print("Creating sampler %d (of %d)" % (i + 1, len(labels_superlist)))
    sampler_list.append(ps.PatchSampler(numpy_rng, img_reader, labels_list))

    print("... building network %d (of %d)" % (i+1, len(labels_superlist)))

    if args.load_net is None:
        embedder_list.append(DeepML(numpy_rng, n_in, args.num_units[0], args.num_units * args.num_hidden_layers[0],
                                    batch_norm_flag, args.activation[0], labels_list, args.patch_rad[0], args.patch_norm[0]))
    else:
        # Read network from file
        embedder_list.append(DeepML.fromfile(numpy_rng, args.load_net[0]))

    # train function

    train_model_list.append(embedder_list[-1].get_train_function(args.learning_rate[0], args.L1_reg[0], args.L2_reg[0], args.sparse_reg[0]))

#
# Sample training estimation batch (for initial scale and batchnorm parameters)

beta0_list = []

for i, (sampler, embedder) in enumerate(zip(sampler_list, embedder_list)):

    # Tp_est, Tv_est, Ap_est, Av_est = patch_sampler.sample(i, args.est_batch_size[0], args.num_neighbors[0], xvalset='train', update_epoch=False)
    Tp_est, Tv_est, Ap_est, Av_est = sampler.sample(args.est_batch_size[0], numpy.inf, args.fract_inside[0], xvalset='train', update_epoch=False)

    Tp_est = normalize_patches(args.patch_norm[0], Tp_est)
    Ap_est = normalize_patches(args.patch_norm[0], Ap_est)
    Y_est = numpy.double(Av_est == Tv_est)

    #
    # Estimate initial scale parameter

    print('estimating initial scale parameter')

    global scale_vars

    # scale for the original method
    l2n = id_net.get_l2n(Tp_est, Ap_est)
    scale_vars = (l2n, Y_est)
    # if args.cost[0] == 'xent': beta0_list.append(minimize(xentropy, 0.001, method='Nelder-Mead', tol=1e-6).x)
    beta0_list.append(minimize(labfus, 0.001, method='Nelder-Mead', tol=1e-6).x)

    embedder.beta0 = beta0_list[-1]  # add beta0 to model

    # scale for the last layer of the network
    l2n = embedder.get_l2n(Tp_est, Ap_est)
    scale_vars = (l2n, Y_est)
    # if args.cost[0] == 'xent': beta_ini_nn = minimize(xentropy, 0.001, method='Nelder-Mead', tol=1e-6).x
    beta_ini_nn = minimize(labfus, 0.001, method='Nelder-Mead', tol=1e-6).x
    embedder.rescale_metric_layer_weights(beta_ini_nn)

    print("Scale param lab %d: beta0 %0.4f, beta_nn %0.4f" % (i, beta0_list[-1], beta_ini_nn))

if args.load_net is not None:
    args.num_units = [embedder_list[-1].metric_layer.n_out]
    args.num_hidden_layers = [len(embedder_list[-1].hidden_layers_list)]
    args.activation = [embedder_list[-1].activation_str]
    args.bn_flag = embedder_list[-1].bn_flag

# Create output model dir

if not os.path.exists(args.store_folder[0]): os.makedirs(args.store_folder[0])

labfus_dir = os.path.join(args.store_folder[0], 'Labfus', args.model_name[0])

group_dir_list = [os.path.join(args.store_folder[0], 'Group%d' % i) for i in range(N_models)]
stats_fig_list = [os.path.join(group_dir_list[i], 'grp%d_%s.png' % (i, args.model_name[0])) for i in range(N_models)]

model_dir_list = []
for group_dir in group_dir_list:
    model_dir_list.append(os.path.join(group_dir, args.model_name[0]))
    if not os.path.exists(model_dir_list[-1]): os.makedirs(model_dir_list[-1])

wait_jobs = [os.path.join(os.environ['ANTSSCRIPTS'], "waitForSGEQJobs.pl"), '0', '30']
check_jobs = [os.path.join(code_path, 'deeplf', 'checkForSGEQJobs.pl'), '1']

#
# PIPELINE
#

print('... training')

start_time = timeit.default_timer()

iter_num = 0
first_iter = True
if args.no_competing_labfus:
    first_iter = False


epoch_superlist = [[0.0] for _ in range(N_models)]
hours_list = [0.0]
cost_tr_full = [[] for _ in range(N_models)]
cost_noreg = [[] for _ in range(N_models)]
acc_labfus = [[] for _ in range(N_models)]
seg_stats = [[] for _ in range(N_models)]
class_stats = [[] for _ in range(N_models)]
sparsity = [[] for _ in range(N_models)]
invalid_updates = [0 for _ in range(N_models)]
# results
segment_epoch_list = [0.0 for _ in range(N_models)]
dice_nl = [0.0 for _ in range(N_models)]
dice_nlb = [0.0 for _ in range(N_models)]
dice_dlf = [0.0 for _ in range(N_models)]
sens_nl = [0.0 for _ in range(N_models)]
sens_nlb = [0.0 for _ in range(N_models)]
sens_dlf = [0.0 for _ in range(N_models)]
spec_nl = [0.0 for _ in range(N_models)]
spec_nlb = [0.0 for _ in range(N_models)]
spec_dlf = [0.0 for _ in range(N_models)]
acc_nl = [0.0 for _ in range(N_models)]
acc_nlb = [0.0 for _ in range(N_models)]
acc_dlf = [0.0 for _ in range(N_models)]

segment_jobs = []
no_final = True

while no_final:

    #
    # Train model (until valid update)
    #

    time_aux = timeit.default_timer()
    # train_time, sample_time = 0., 0.

    for i, (sampler, embedder, train_model) in enumerate(zip(sampler_list, embedder_list, train_model_list)):

        # print('sampler %d' % i)

        while True:

            # Sample training data

            # time_aux2 = timeit.default_timer()

            Tp_tr, Tv_tr, Ap_tr, Av_tr = sampler.sample(args.train_batch_size[0], args.num_neighbors[0], args.fract_inside[0],
                                                        xvalset='train', update_epoch=True)
            # Tp_tr, Tv_tr, Ap_tr, Av_tr = patch_sampler.sample(i, args.train_batch_size[0], args.num_neighbors[0], xvalset='train', update_epoch=True)

            # sample_time = timeit.default_timer() - time_aux2
            #
            # print('Sample time: %0.3f' % sample_time)

            Tp_tr = normalize_patches(args.patch_norm[0], Tp_tr)
            Ap_tr = normalize_patches(args.patch_norm[0], Ap_tr)
            Y_tr = numpy.double(Av_tr == Tv_tr)

            embedder.backup_multilayer_params()

            # train model

            # time_aux2 = timeit.default_timer()

            cost = train_model(numpy.float32(Tp_tr), numpy.float32(Ap_tr), numpy.float32(Y_tr), kl_rho)

            # train_time = timeit.default_timer() - time_aux2
            #
            # print('Train time: %0.3f' % train_time)

            if numpy.isfinite(cost): break

            print('Invalid update. Restore previous params and resample new patches')

            embedder.restore_multilayer_params()

            sleep(5)

            invalid_updates[i] += 1

        cost_tr_full[i].append(cost)

    total_time = timeit.default_timer() - time_aux

    print('Avg. train cost: %0.3f. Total time: %0.3f' % (numpy.asarray([cost_tr_full[i][-1] for i in range(N_models)]).mean(), total_time))


    #
    # Display statistics
    #

    if iter_num % args.display_frequency[0] == 0:

        print('computing display statistics...')
        time_aux = timeit.default_timer()

        for i, (sampler, embedder, model_dir) in enumerate(zip(sampler_list, embedder_list, model_dir_list)):

            if sampler.epoch > args.num_epochs[0]: continue
            # if patch_sampler.epoch[i] > args.num_epochs[0]: continue

            # Store latest model

            Tp_est, Ap_est = None, None

            if batch_norm_flag:  # if bn layers then sample estimation batch (for saving bn statistics)

                Tp_est, Tv_est, Ap_est, Av_est = sampler.sample(args.est_batch_size[0], numpy.inf, args.fract_inside[0],
                                                                xvalset='train', update_epoch=False)
                # Tp_est, Tv_est, Ap_est, Av_est = patch_sampler.sample(i, args.est_batch_size[0], numpy.inf, xvalset='train', update_epoch=False)

                Tp_est = normalize_patches(args.patch_norm[0], Tp_est)
                Ap_est = normalize_patches(args.patch_norm[0], Ap_est)

            model_name = os.path.join(model_dir, 'grp%d_latest.dat' % i)
            embedder.write_multilayer(model_name, Tp_est, Ap_est, sampler.epoch)
            # embedder.write_multilayer(model_name, Tp_est, Ap_est, patch_sampler.epoch[i])

            # Performance on train set

            Tp_tr, Tv_tr, Ap_tr, Av_tr = sampler.sample(args.train_batch_size[0], args.num_neighbors[0], args.fract_inside[0],
                                                        xvalset='train', update_epoch=False)
            # Tp_tr, Tv_tr, Ap_tr, Av_tr = patch_sampler.sample(i, args.train_batch_size[0], args.num_neighbors[0], xvalset='train', update_epoch=False)

            Tp_tr = normalize_patches(args.patch_norm[0], Tp_tr)
            Ap_tr = normalize_patches(args.patch_norm[0], Ap_tr)
            Y_tr = numpy.double(Av_tr == Tv_tr)

            c_tr_aux = embedder.get_cost(Tp_tr, Ap_tr, Y_tr)

            # Performance on validation set

            Tp_val, Tv_val, Ap_val, Av_val = sampler.sample(args.est_batch_size[0], numpy.inf, args.fract_inside[0], xvalset='val')
            # Tp_val, Tv_val, Ap_val, Av_val = patch_sampler.sample(i, args.est_batch_size[0], numpy.inf, xvalset='val')

            Tp_val = normalize_patches(args.patch_norm[0], Tp_val)
            Ap_val = normalize_patches(args.patch_norm[0], Ap_val)
            Y_val = numpy.double(Av_val == Tv_val)

            c_val_aux = embedder.get_cost(Tp_val, Ap_val, Y_val)#, args.cost[0])
            c_val0_aux = id_net.get_cost(Tp_val, Ap_val, Y_val, beta0_list[i])#, args.cost[0])

            acc_val_aux = embedder.get_acc(Tp_val, Ap_val, Y_val)
            acc_val0_aux = id_net.get_acc(Tp_val, Ap_val, Y_val, beta0_list[i])

            # kl-divergence
            kl_val = embedder.get_kl(Tp_val, Ap_val, kl_rho)
            kl_val0 = id_net.get_kl(Tp_val, Ap_val, beta0_list[i], kl_rho)

            cost_noreg[i].append((c_tr_aux, c_val_aux, c_val0_aux))
            acc_labfus[i].append((acc_val_aux, acc_val0_aux))
            sparsity[i].append((kl_val, kl_val0))

            print('Group %d: tr cost %0.3f, val cost %0.3f, val0 cost %0.3f' % (i, c_tr_aux, c_val_aux, c_val0_aux))
            print('Group %d: val acc %0.3f, val0 acc %0.3f' % (i, acc_val_aux, acc_val0_aux))
            print('Group %d: val sparse %0.3f, val0 sparse %0.3f' % (i, kl_val, kl_val0))

            try:
                # write statistics
                write_stats4(stats_fig_list[i], args, epoch_superlist[i], hours_list, cost_tr_full[i][::args.display_frequency[0]],
                             cost_noreg[i], acc_labfus[i], sparsity[i], seg_stats[i], class_stats[i], invalid_updates[i],
                             (labels_superlist[i], beta0_list[i]))
            except:
                warn('Failed to write statistics after display')

            epoch_superlist[i].append(sampler.epoch)
            # epoch_superlist[i].append(patch_sampler.epoch[i])

        hours_list.append((timeit.default_timer() - start_time) / 3600.)

        display_time = timeit.default_timer() - time_aux

        print('display time: %0.3f' % display_time)

        # Update stopping condition
        no_final = False
        for epoch_list in epoch_superlist:
            if epoch_list[-1] < args.num_epochs[0]:
                no_final = True
                break


    #
    # Segment statistics
    #

    if iter_num % args.segment_frequency[0] == 0:# and is_hpc:

        if check_output(check_jobs + segment_jobs).find('Still waiting') == -1:

            print('Not waiting for any segmentation job')

    # segment_jobs = 3
    # aux_segment = 3

            if segment_jobs:

                print('Previous segmentation jobs finished')

                segment_time = timeit.default_timer() - aux_segment
                print('Segment time: %0.3f' % segment_time)
                print('Going to evaluate')

                # Evaluate segmentations

                if args.parallel_labfus:
                    # pairs of id group, id model
                    grp_idx_list = []
                    lab_ini = 0
                    for id_model, labels_list in enumerate(labels_superlist):
                        len_list = len(labels_list)
                        grp_idx_list += list(zip(range(lab_ini, lab_ini + len_list), [id_model]*len_list))
                        lab_ini += len_list
                else:
                    grp_idx_list = [(i, i) for i in range(N_models)]

                seg_err_list = [False for _ in range(N_models)]
                n_lab_list = [0.0 for _ in range(N_models)]  # number of labels per model

                for grp, i in grp_idx_list:

                    dice_dlf[i] = 0.0  # initialize dice accumulator

                    if seg_err_list[i]: continue  # if got segmentation errors in some label of the same model then skip

                    n_lab_list[i] += 1.0  # increase number of labels of current model

                    suffix_list = ['_dlf_grp%d.nii.gz' % grp]
                    if first_iter:
                        suffix_list.append('_nl_grp%d.nii.gz' % grp)
                        suffix_list.append('_nlb_grp%d.nii.gz' % grp)

                    try:
                        for suffix in suffix_list:
                            cmdline = [python_path, '-u', evalseg_path]
                            cmdline.extend(['--est_dir'] + [labfus_dir])
                            cmdline.extend(['--est_suffix'] + [suffix])
                            cmdline.extend(['--gtr_dir'] + args.val_dir)
                            cmdline.extend(['--gtr_suffix'] + args.lab_suffix)

                            call(cmdline)

                            with open(os.path.join(labfus_dir, 'label_dice.csv'), 'rb') as f:
                                reader = csv.reader(f)
                                label_dices_str = list(reader)[1]
                                label_dices = numpy.asarray([float(f) for f in label_dices_str])
                                if suffix == '_nl_grp%d.nii.gz' % grp: dice_nl[i] += label_dices.mean()
                                if suffix == '_nlb_grp%d.nii.gz' % grp: dice_nlb[i] += label_dices.mean()
                                if suffix == '_dlf_grp%d.nii.gz' % grp: dice_dlf[i] += label_dices.mean()
                                os.remove(os.path.join(labfus_dir, 'label_dice.csv'))

                    except:
                        seg_err_list[i] = True  # indicate segmentation errors in i-th model
                        warn('There were some segmentation errors')
                        continue

                    # SENSITIVITY, SPECIFICITY, ACCURACY

                    sens_dlf[i] = 0.0  # initialize sens accumulator
                    spec_dlf[i] = 0.0  # initialize spec accumulator
                    acc_dlf[i] = 0.0  # initialize acc accumulator

                    suffix2_list = ['_dlf_grp%d.txt' % grp]
                    if first_iter:
                        suffix2_list.append('_nl_grp%d.txt' % grp)
                        suffix2_list.append('_nlb_grp%d.txt' % grp)
                    try:
                        for suffix in suffix2_list:
                            files_list = os.listdir(labfus_dir)
                            est_files = [f for f in files_list if f.endswith(suffix)]
                            assert est_files, "No estimated segmentation found"
                            sens, spec, acc = 0., 0., 0.
                            for est_file in est_files:
                                with open(os.path.join(labfus_dir, est_file), 'r') as f:
                                    for line in f:
                                        values = line.split(',')
                                        sens += float(values[0])
                                        spec += float(values[1])
                                        acc += float(values[2])
                                        break
                            if suffix == '_nl_grp%d.txt' % grp:
                                sens_nl[i] += sens / float(len(est_files))
                                spec_nl[i] += spec / float(len(est_files))
                                acc_nl[i] += acc / float(len(est_files))
                            if suffix == '_nlb_grp%d.txt' % grp:
                                sens_nlb[i] += sens / float(len(est_files))
                                spec_nlb[i] += spec / float(len(est_files))
                                acc_nlb[i] += acc / float(len(est_files))
                            if suffix == '_dlf_grp%d.txt' % grp:
                                sens_dlf[i] += sens / float(len(est_files))
                                spec_dlf[i] += spec / float(len(est_files))
                                acc_dlf[i] += acc / float(len(est_files))
                    except:
                        seg_err_list[i] = True  # indicate segmentation errors in i-th model
                        warn('There were some segmentation errors')
                        continue

                for i in range(N_models):

                    if seg_err_list[i]: continue

                    # average dices across labels
                    dice_dlf[i] /= n_lab_list[i]
                    if first_iter:
                        dice_nl[i] /= n_lab_list[i]
                        dice_nlb[i] /= n_lab_list[i]

                    sens_dlf[i] /= n_lab_list[i]
                    spec_dlf[i] /= n_lab_list[i]
                    acc_dlf[i] /= n_lab_list[i]
                    if first_iter:
                        sens_nl[i] /= n_lab_list[i]
                        spec_nl[i] /= n_lab_list[i]
                        acc_nl[i] /= n_lab_list[i]
                        sens_nlb[i] /= n_lab_list[i]
                        spec_nlb[i] /= n_lab_list[i]
                        acc_nlb[i] /= n_lab_list[i]

                    seg_stats[i].append((dice_nl[i], dice_nlb[i], dice_dlf[i], segment_epoch_list[i]))
                    class_stats[i].append((sens_nl[i], spec_nl[i], acc_nl[i],
                                           sens_nlb[i], spec_nlb[i], acc_nlb[i],
                                           sens_dlf[i], spec_dlf[i], acc_dlf[i], segment_epoch_list[i]))

                    print("Dices group %d (epoch %0.3f): nlwv %0.4f, nlbeta %0.4f, deeplf %0.4f" %
                          (i, segment_epoch_list[i], dice_nl[i], dice_nlb[i], dice_dlf[i]))
                    print("Acc group %d (epoch %0.3f): nlwv %0.4f, nlbeta %0.4f, deeplf %0.4f" %
                          (i, segment_epoch_list[i], acc_nl[i], acc_nlb[i], acc_dlf[i]))

                    try:
                        # write statistics
                        write_stats4(stats_fig_list[i], args, epoch_superlist[i], hours_list, cost_tr_full[i][::args.display_frequency[0]],
                                     cost_noreg[i], acc_labfus[i], sparsity[i], seg_stats[i], class_stats[i], invalid_updates[i],
                                     (labels_superlist[i], beta0_list[i]))
                    except:
                        warn('Failed to write statistics after segmentation')

                print("Avg. dices: nlwv %0.4f, nlbeta %0.4f, deeplf %0.4f" %
                      (numpy.asarray(dice_nl).mean(), numpy.asarray(dice_nlb).mean(), numpy.asarray(dice_dlf).mean()))

                print("Avg. acc: nlwv %0.4f, nlbeta %0.4f, deeplf %0.4f" %
                      (numpy.asarray(acc_nl).mean(), numpy.asarray(acc_nlb).mean(), numpy.asarray(acc_dlf).mean()))

                if os.path.exists(labfus_dir): rmtree(labfus_dir)

                first_iter = False

            #
            # Segment

            aux_segment = timeit.default_timer()

            # Store models

            model_files_list = []

            for i, (sampler, embedder, model_dir) in enumerate(zip(sampler_list, embedder_list, model_dir_list)):

                Tp_est, Ap_est = None, None

                if batch_norm_flag:  # if bn layers then sample estimation batch (for saving bn statistics)

                    Tp_est, Tv_est, Ap_est, Av_est = sampler.sample(args.est_batch_size[0], numpy.inf, args.fract_inside[0],
                                                                    xvalset='train', update_epoch=False)
                    # Tp_est, Tv_est, Ap_est, Av_est = patch_sampler.sample(i, args.est_batch_size[0], numpy.inf, xvalset='train', update_epoch=False)

                    Tp_est = normalize_patches(args.patch_norm[0], Tp_est)
                    Ap_est = normalize_patches(args.patch_norm[0], Ap_est)

                model_files_list.append(os.path.join(model_dir, ('grp%d_epch' % i) + ('%0.3f' % sampler.epoch).replace('.', '_') + '.dat'))
                embedder.write_multilayer(model_files_list[-1], Tp_est, Ap_est, sampler.epoch)
                # model_files_list.append(os.path.join(model_dir, ('grp%d_epch' % i) + ('%0.3f' % patch_sampler.epoch[i]).replace('.', '_') + '.dat'))
                # embedder.write_multilayer(model_files_list[-1], Tp_est, Ap_est, patch_sampler.epoch[i])

                # keep segmentation epochs
                segment_epoch_list[i] = sampler.epoch

            if not os.path.exists(labfus_dir): os.makedirs(labfus_dir)

            # Segment

            print('Going to launch segmentation jobs')

            try:

                segment_jobs = []
                for i, (val_name, val_path, train_img_path_list, train_lab_path_list) in \
                        enumerate(zip(val_name_list, val_path_list, train_img_path_superlist, train_lab_path_superlist)):

                    # entangle list of labels with their corresponding params (for parallel label fusion)
                    aux = [([label], model_file) for labels_list, model_file in zip(labels_superlist, model_files_list) for label in labels_list]
                    label_params_list = zip(*aux) if args.parallel_labfus else [labels_superlist, model_files_list]

                    methods_list, label_params_superlist, suffix_list = ['deeplf'], [label_params_list], ['_dlf.nii.gz']

                    if first_iter:
                        # nlbeta
                        methods_list.append('nlbeta')
                        # entangle list of labels with their corresponding params (for parallel label fusion)
                        aux = [([label], beta0) for labels_list, beta0 in zip(labels_superlist, beta0_list) for label in labels_list]
                        label_params_list = zip(*aux) if args.parallel_labfus else [labels_superlist, beta0_list]
                        label_params_superlist.append(label_params_list)
                        suffix_list.append('_nlb.nii.gz')
                        # nlwv
                        methods_list.append('nlwv')
                        # entangle list of labels with their corresponding params (for parallel label fusion)
                        aux = [([label], 0.0) for labels_list in labels_superlist for label in labels_list]
                        label_params_list = zip(*aux) if args.parallel_labfus else [labels_superlist, [0.0]*N_models]
                        label_params_superlist.append(label_params_list)
                        suffix_list.append('_nl.nii.gz')

                    for method, label_params_list, suffix in zip(methods_list, label_params_superlist, suffix_list):

                        out_file = os.path.join(labfus_dir, val_name + suffix)

                        # REMEMBER TO SET SEARCH_RAD TO '1' IF USING SINGLE IMAGE FOR LEARNING

                        patch_rad = '%dx%dx%d' % (args.patch_rad[0], args.patch_rad[0], args.patch_rad[0])
                        # search_rad = '%dx%dx%d' % (args.search_rad[0], args.search_rad[0], args.search_rad[0])
                        search_rad = '1x1x1'
                        fusion_rad = '1x1x1'
                        struct_sim = 0.9
                        patch_norm = args.patch_norm[0]

                        # if actual threads is given, then num_threads becomes a list [num_threads, actual_num_threads]
                        num_threads = None
                        if method == 'deeplf':
                            num_threads = args.num_threads_deeplf[0]

                        _, job_id_list = label_fusion(val_path, train_img_path_list, train_lab_path_list, out_file, False, method, patch_rad, search_rad,
                                                      fusion_rad, struct_sim, patch_norm, label_params_list[1], label_params_list[0], num_threads,
                                                      val_path.split(args.img_suffix[0])[0] + args.lab_suffix[0])

                        if is_hpc: segment_jobs.extend(job_id_list)

            except:
                warn("Failed to launch segmentation jobs")

            print('Launched seg jobs: %s' % (' '.join(segment_jobs)))

    iter_num += 1

