
import timeit
from time import sleep
import argparse
import numpy
import PatchSampler as ps
from scipy.optimize import minimize
from DeepMetricLearning import DeepML, IdNet, normalize_patches
from deeplf_display import write_stats4
import os
from warnings import warn

scale_vars = []  # [l2n, y]: global variables for scale estimation

sigmoid = lambda x: 1 / (1 + numpy.exp(-x))

def softmax(w):

    e = numpy.exp(w)
    return e / numpy.sum(e, axis=1, keepdims=True)

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


parser = argparse.ArgumentParser(description='Learns deep embeddings for patch-based label fusion')

# dataset arguments
parser.add_argument("--train_dir", type=str, nargs=1, required=True, help="directory of training images")
parser.add_argument("--val_dir", type=str, nargs=1, required=True, help="directory of validation images")
parser.add_argument("--img_suffix", type=str, nargs=1, required=True, help="suffix of images")
parser.add_argument("--lab_suffix", type=str, nargs=1, required=True, help="suffix of labelmaps")

# sampler argument
parser.add_argument("--fract_inside", type=float, nargs=1, default=[.5], help="(optional) fraction of inside patches to sample (default .5)")
parser.add_argument("--num_neighbors", type=int, nargs=1, default=[50], help="(optional) number of neighbors for each target patch (default 50)")

# store arguments
parser.add_argument("--model_name", type=str, nargs=1, required=True, help="name of the model (used to name output files)")
parser.add_argument("--store_folder", type=str, nargs=1, required=True, help="folder to store model to")
parser.add_argument("--label_group", type=int, nargs='+', action='append', required=True, help="Group of labels defining a model."
                                                                                               "A separate model will be learnt from each group of labels."
                                                                                               "Typically a group of labels is defined as a pair of labels for"
                                                                                               "the left and right parts of a structure")

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
parser.add_argument("--search_rad", type=int, nargs=1, default=[3], help="(optional) neighborhood radius for sampling voting patches (default 3)")

args = parser.parse_args()
# EXAMPLE:
# args = parser.parse_args(''
#                          '--train_dir /Users/gsanroma/DATA/deeplf/data/sata_mini_train '
#                          '--val_dir /Users/gsanroma/DATA/deeplf/data/sata_mini_val '
#                          '--img_suffix _brain.nii.gz '
#                          '--lab_suffix _glm.nii.gz '
#                          #
#                          '--fract_inside 0.5 '
#                          '--model_name kk '
#                          '--store_folder /Users/gsanroma/DATA/deeplf/models_sata '
#                          '--label_group 31 32 '
#                          '--label_group 36 37 '
#                          '--num_epochs 30 '
#                          '--train_batch_size 50 '
#                          '--est_batch_size 500 '
#                          '--display_frequency 1 '
#                          '--learning_rate 0.0002 '
#                          '--L2_reg 0. '
#                          '--sparse_reg 2e-3 '
#                          #
#                          '--num_units 100 '
#                          '--num_hidden_layers 1 '
#                          '--activation relu '
#                          '--batch_norm '
#                          #
#                          '--num_neighbors 50 '
#                          '--patch_rad 3 '
#                          '--search_rad 3 '
#                          '--patch_norm zscore '.split())

numpy_rng = numpy.random.RandomState(1234)

n_in = ((args.patch_rad[0] * 2 + 1) ** 3)

kl_rho = numpy.asarray(args.kl_rho[0], dtype='float32')  # for training and obtaining kl_div
kl_rho.shape = (1, 1)

# correct batch_norm if no hidden layers
batch_norm_flag = args.batch_norm if args.num_hidden_layers[0] > 0 else False

print('... building idnet')

id_net = IdNet()

#
# Create sampler and embedder list

img_reader = ps.ImageReader(args.train_dir[0], args.val_dir[0], args.img_suffix[0], args.lab_suffix[0],
                            args.label_group, args.patch_rad[0], args.search_rad[0])

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

    Tp_est, Tv_est, Ap_est, Av_est = sampler.sample(args.est_batch_size[0], numpy.inf, args.fract_inside[0], xvalset='train', update_epoch=False)  # get all neighbors for estimation

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
    beta0_list.append(minimize(labfus, 0.001, method='Nelder-Mead', tol=1e-6).x)

    embedder.beta0 = beta0_list[-1]  # add beta0 to model

    # scale for the last layer of the network
    l2n = embedder.get_l2n(Tp_est, Ap_est)
    scale_vars = (l2n, Y_est)
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

#
# PIPELINE
#

print('... training')

start_time = timeit.default_timer()

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
# these are variables to store segmentation results on validation images
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

iter_num = 0
no_final = True

while no_final:

    #
    # Train model (until valid update)
    #

    time_aux = timeit.default_timer()

    for i, (sampler, embedder, train_model) in enumerate(zip(sampler_list, embedder_list, train_model_list)):

        while True:

            # Sample training data

            Tp_tr, Tv_tr, Ap_tr, Av_tr = sampler.sample(args.train_batch_size[0], args.num_neighbors[0], args.fract_inside[0],
                                                        xvalset='train', update_epoch=True)

            Tp_tr = normalize_patches(args.patch_norm[0], Tp_tr)
            Ap_tr = normalize_patches(args.patch_norm[0], Ap_tr)
            Y_tr = numpy.double(Av_tr == Tv_tr)

            embedder.backup_multilayer_params()

            # train model

            cost = train_model(numpy.float32(Tp_tr), numpy.float32(Ap_tr), numpy.float32(Y_tr), kl_rho)

            if numpy.isfinite(cost): break

            print('Invalid update. Restore previous params and resample new patches')

            embedder.restore_multilayer_params()

            sleep(5)  # in case of bad update

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

            # Store latest model

            Tp_est, Ap_est = None, None

            if batch_norm_flag:  # if bn layers then sample estimation batch (for saving bn statistics)

                Tp_est, Tv_est, Ap_est, Av_est = sampler.sample(args.est_batch_size[0], numpy.inf, args.fract_inside[0],
                                                                xvalset='train', update_epoch=False)

                Tp_est = normalize_patches(args.patch_norm[0], Tp_est)
                Ap_est = normalize_patches(args.patch_norm[0], Ap_est)

            model_name = os.path.join(model_dir, 'grp%d_latest.dat' % i)
            embedder.write_multilayer(model_name, Tp_est, Ap_est, sampler.epoch)

            # Performance on train set

            Tp_tr, Tv_tr, Ap_tr, Av_tr = sampler.sample(args.train_batch_size[0], args.num_neighbors[0], args.fract_inside[0],
                                                        xvalset='train', update_epoch=False)

            Tp_tr = normalize_patches(args.patch_norm[0], Tp_tr)
            Ap_tr = normalize_patches(args.patch_norm[0], Ap_tr)
            Y_tr = numpy.double(Av_tr == Tv_tr)

            c_tr_aux = embedder.get_cost(Tp_tr, Ap_tr, Y_tr)

            # Performance on validation set

            Tp_val, Tv_val, Ap_val, Av_val = sampler.sample(args.est_batch_size[0], numpy.inf, args.fract_inside[0], xvalset='val')

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

    # if iter_num % args.segment_frequency[0] == 0:  # launch segmentation jobs once each args.segment_frequency[0] iterations

    # Here segmentation jobs should be launched to segment the validation images in val_dir and results should be evaluated when segmentation jobs finished

    # First of all, results should be evaluated if previous segmentation jobs already finished
    # Dice scores for each model should be kept in lists: dice_dlf (deeplf), dice_nl (nlwv), dice_nlb (nlbeta)
    # Alternatively, sensitivities and specificities can be kept in lists: sens_dfl, spec_dlf, ...

    # Next, if previous segmentation jobs finished then save the current models (with epoch identifier) and launch new jobs.
    # Keeping the current networks with epoch identifier can be done with something along the lines:

    # model_files_list = []
    # for i, (sampler, embedder, model_dir) in enumerate(zip(sampler_list, embedder_list, model_dir_list)):
    #     Tp_est, Ap_est = None, None
    #     if batch_norm_flag:  # if bn layers then sample estimation batch (for saving bn statistics)
    #         Tp_est, Tv_est, Ap_est, Av_est = sampler.sample(args.est_batch_size[0], numpy.inf, args.fract_inside[0],
    #                                                         xvalset='train', update_epoch=False)
    #         Tp_est = normalize_patches(args.patch_norm[0], Tp_est)
    #         Ap_est = normalize_patches(args.patch_norm[0], Ap_est)
    #
    #     model_files_list.append(os.path.join(model_dir, ('grp%d_epch' % i) + ('%0.3f' % sampler.epoch).replace('.', '_') + '.dat'))
    #     embedder.write_multilayer(model_files_list[-1], Tp_est, Ap_est, sampler.epoch)
    #     # keep segmentation epochs
    #     segment_epoch_list[i] = sampler.epoch


    # Segmentation jobs should be launched using script pblf_py to segment the validation images using the saved models
    # Segmentation jobs should be ideally asyncronous so that training does not need to stop while segmenting validation images


    iter_num += 1

