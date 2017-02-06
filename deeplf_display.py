import numpy
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import matplotlib.pyplot as plt


bigfont = 12
medfont = 10  # fontsize
smallfont = 7

def write_stats4(file, args, epochs, hours, cost_tr_full, cost_noreg, acc_labfus, sparsity, seg_stats, class_stats, invalid_updates, label_beta0_pair):

    distinct_colors = plt.get_cmap('jet')(numpy.linspace(0, 1.0, args.num_hidden_layers[0] + 1))[:-1]
    fig = plt.figure(figsize=(8.27, 11.69))
    plt.clf()

    plt.subplot(421)
    plt.plot(cost_tr_full)
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.title('cost train (w regularization)', fontsize=medfont)

    plt.subplot(422)
    aux = numpy.asarray(cost_noreg)
    c_tr, c_val, c_val0 = aux.T[0], aux.T[1], aux.T[2]
    plt.plot(c_tr, 'b', label='tr set')
    plt.plot(c_val, 'g', label='val set')
    plt.plot(c_val0, 'r', label='val set 0')
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.legend(fontsize=medfont, loc=4)
    plt.title('cost w/o regularization', fontsize=medfont)

    plt.subplot(423)
    aux = numpy.asarray(acc_labfus)
    acc_val, acc_val0 = aux.T[0], aux.T[1]
    plt.plot(acc_val, 'b', label='deeplf')
    plt.plot(acc_val0, 'r', label='original')
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.legend(fontsize=medfont, loc=4)
    plt.title('label fusion accuracy', fontsize=medfont)

    plt.subplot(424)
    aux = numpy.asarray(sparsity)
    sprs_val, sprs_val0 = aux.T[0], aux.T[1]
    plt.plot(sprs_val, 'b', label='deeplf')
    plt.plot(sprs_val0, 'r', label='original')
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.legend(fontsize=medfont, loc=1)
    plt.title('sparsity (KL divergence)', fontsize=medfont)

    plt.subplot(425)
    aux = numpy.asarray(seg_stats)
    if aux.size > 0:
        dice_nl, dice_nlb, dice_dlf, epochs_dice = aux.T[0], aux.T[1], aux.T[2], aux.T[3]
        plt.plot(epochs_dice, dice_dlf, 'b', label='deeplf')
        plt.plot(epochs_dice, dice_nl, 'g', ls='dashed', label='nlwv')
        plt.plot(epochs_dice, dice_nlb, 'r', label='nlbeta')
        plt.ylim([0.65, 1.])
        plt.xticks(fontsize=smallfont)
        plt.yticks(fontsize=smallfont)
        plt.legend(fontsize=medfont, loc=4)
        plt.title('MULTI-ATLAS segmentation dice', fontsize=medfont)

    plt.subplot(426)
    aux = numpy.asarray(class_stats)
    if aux.size > 0:
        sens_nl, spec_nl, acc_nl, sens_nlb, spec_nlb, acc_nlb, sens_dlf, spec_dlf, acc_dlf, epochs_dice = \
            aux.T[0], aux.T[1], aux.T[2], aux.T[3], aux.T[4], aux.T[5], aux.T[6], aux.T[7], aux.T[8], aux.T[9]
        plt.plot(epochs_dice, acc_dlf, 'b', ls='solid', label='ACC dlf')
        plt.plot(epochs_dice, acc_nl, 'g', ls='solid', label='ACC nl')
        plt.plot(epochs_dice, acc_nlb, 'r', ls='solid', label='ACC nlb')
        plt.plot(epochs_dice, sens_dlf, 'b', ls='dashed', label='SENS dlf')
        plt.plot(epochs_dice, sens_nl, 'g', ls='dashed', label='SENS nl')
        plt.plot(epochs_dice, sens_nlb, 'r', ls='dashed', label='SENS nlb')
        plt.plot(epochs_dice, spec_dlf, 'b', ls='dotted', label='SPEC dlf')
        plt.plot(epochs_dice, spec_nl, 'g', ls='dotted', label='SPEC nl')
        plt.plot(epochs_dice, spec_nlb, 'r', ls='dotted', label='SPEC nlb')
        # plt.ylim([0.65, 1.])
        plt.xticks(fontsize=smallfont)
        plt.yticks(fontsize=smallfont)
        # plt.legend(fontsize=smallfont, loc=4)
        plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1, fontsize=smallfont)
        plt.title('Classification metrics', fontsize=medfont)

    plt.subplot(428)
    plt.plot(epochs, hours)
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.title('epoch vs. time', fontsize=medfont)

    plt.subplot(427)
    plt.axis('off')
    plt.text(0, .9, 'learn rate: %g' % (args.learning_rate[0]), fontsize=medfont, fontweight='bold')
    plt.text(0, .8, 'hidden layers: %d' % (args.num_hidden_layers[0]), fontsize=medfont, fontweight='bold')
    plt.text(0, .7, 'num units: %d' % (args.num_units[0]), fontsize=medfont, fontweight='bold')
    plt.text(0, .6, 'activation: %s' % (args.activation[0]), fontsize=medfont, fontweight='bold')
    plt.text(0, .5, 'L2 reg: %g' % (args.L2_reg[0]), fontsize=medfont, fontweight='bold')

    plt.text(0, .4, 'train batch size: %d' % (args.train_batch_size[0]), fontsize=medfont)
    plt.text(0, .3, 'num neighbors: %d' % (args.num_neighbors[0]), fontsize=medfont)
    plt.text(0, .2, 'patch radius: %d' % (args.patch_rad[0]), fontsize=medfont)
    plt.text(0, .1, 'patch norm: %s' % (args.patch_norm[0]), fontsize=medfont)
    plt.text(0, .0, 'search radius: %d' % (args.search_rad[0]), fontsize=medfont)
    plt.text(0, -.1, 'invalid updates: %d' % (invalid_updates), fontsize=medfont)

    # plt.text(0.6, .9, 'fract inside: %0.3f' % (args.fract_inside[0]), fontsize=medfont)
    # plt.text(0.6, .8, 'cert or dist max: %0.3f' % (args.cert_or_dist_max[0]), fontsize=medfont)
    plt.text(0.6, .7, 'sparse reg: %g' % (args.sparse_reg[0]), fontsize=medfont)

    plt.text(0.6, 0.6, 'Labs: %s' % (label_beta0_pair[0]), fontsize=smallfont)
    plt.text(0.6, 0.5, 'beta0 %0.5f' % (label_beta0_pair[1]), fontsize=medfont)

    plt.savefig(file)
    plt.close(fig)



def write_stats(file, args, epochs, hours, cost_tr_full, cost_noreg, dist_val, activ, bn_params, invalid_updates):

    distinct_colors = plt.get_cmap('jet')(numpy.linspace(0, 1.0, args.num_hidden_layers[0] + 1))[:-1]
    fig = plt.figure(figsize=(8.27, 11.69))
    plt.clf()

    plt.subplot(421)
    plt.plot(cost_tr_full)
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.title('cost train (w regularization)', fontsize=medfont)

    plt.subplot(422)
    aux = numpy.asarray(cost_noreg)
    c_tr, c_val, c_val0 = aux.T[0], aux.T[1], aux.T[2]
    plt.plot(c_tr, 'b', label='tr set')
    plt.plot(c_val, 'g', label='val set')
    plt.plot(c_val0, 'r', label='val set 0')
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.legend(fontsize=medfont)
    plt.title('cost w/o regularization', fontsize=medfont)

    plt.subplot(423)
    aux = numpy.asarray(dist_val)
    m_d1, s_d1, m_d0, s_d0 = aux.T[0], aux.T[1], aux.T[2], aux.T[3]
    plt.plot(m_d1, 'b', label='same class')
    plt.plot(numpy.array(m_d1) - numpy.array(s_d1), 'b--')
    plt.plot(numpy.array(m_d1) + numpy.array(s_d1), 'b--')
    plt.plot(m_d0, 'r', label='diff class')
    plt.plot(numpy.array(m_d0) - numpy.array(s_d0), 'r--')
    plt.plot(numpy.array(m_d0) + numpy.array(s_d0), 'r--')
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.legend(fontsize=medfont)
    plt.title('Euclidean distances', fontsize=medfont)

    plt.subplot(424)
    aux = numpy.asarray(activ)
    act_list = aux.T
    for i, act_mu_std_layer in enumerate(act_list):
        mu_act, std_act = act_mu_std_layer
        plt.plot(mu_act, c=distinct_colors[i], label='layer %d' % (i))
        plt.plot(numpy.array(mu_act) - numpy.array(std_act), ls='dashed', c=distinct_colors[i])
        plt.plot(numpy.array(mu_act) + numpy.array(std_act), ls='dashed', c=distinct_colors[i])
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.legend(fontsize=medfont)
    plt.title('mean (std) activations', fontsize=medfont)

    # GAmma & Beta
    aux = numpy.asarray(bn_params)
    bn_list = aux.T
    gamma_list, beta_list = [], []
    for i, gamma_beta_layer in enumerate(bn_list):
        gamma_layer, beta_layer = gamma_beta_layer
        gamma_list.append(gamma_layer)
        beta_list.append(beta_layer)

    plt.subplot(425)
    for i, gamma_layer in enumerate(gamma_list):
        plt.plot(gamma_layer, c=distinct_colors[i], label='layer %d' % (i))
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.legend(fontsize=medfont)
    plt.title('gamma', fontsize=medfont)

    plt.subplot(426)
    for i, beta_layer in enumerate(beta_list):
        plt.plot(beta_layer, c=distinct_colors[i], label='layer %d' % (i))
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.legend(fontsize=medfont)
    plt.title('beta', fontsize=medfont)

    plt.subplot(427)
    plt.plot(hours, epochs)
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.title('epoch vs. time', fontsize=medfont)

    plt.subplot(428)
    plt.axis('off')
    plt.text(0, .9, 'learn rate: %g' % (args.learning_rate[0]), fontsize=medfont, fontweight='bold')
    plt.text(0, .8, 'hidden layers: %d' % (args.num_hidden_layers[0]), fontsize=medfont, fontweight='bold')
    plt.text(0, .7, 'num units: %d' % (args.num_units[0]), fontsize=medfont, fontweight='bold')
    plt.text(0, .6, 'activation: %s' % (args.activation[0]), fontsize=medfont, fontweight='bold')
    plt.text(0, .5, 'L2 reg: %g' % (args.L2_reg[0]), fontsize=medfont, fontweight='bold')
    plt.text(0, .4, 'train batch size: %d' % (args.train_batch_size[0]), fontsize=medfont)
    plt.text(0, .3, 'num neighbors: %d' % (args.num_neighbors[0]), fontsize=medfont)
    plt.text(0, .2, 'patch radius: %d' % (args.patch_radius[0]), fontsize=medfont)
    plt.text(0, .1, 'patch norm: %s' % (args.patch_norm[0]), fontsize=medfont)
    plt.text(0, .0, 'invalid updates: %d' % (invalid_updates), fontsize=medfont)

    plt.text(0.6, .9, 'fract inside: %0.3f' % (args.fract_inside[0]), fontsize=medfont)
    plt.text(0.6, .8, 'est batch size: %d' % (args.est_batch_size[0]), fontsize=medfont)
    plt.text(0.6, .7, 'cert or dist max: %0.3f' % (args.cert_or_dist_max[0]), fontsize=medfont)

    plt.savefig(file)
    plt.close(fig)


def write_stats3(file, args, epochs, hours, cost_tr_full, cost_noreg, activ, bn_params, acc_labfus, seg_stats, invalid_updates, beta0):

    distinct_colors = plt.get_cmap('jet')(numpy.linspace(0, 1.0, args.num_hidden_layers[0] + 1))[:-1]
    fig = plt.figure(figsize=(8.27, 11.69))
    plt.clf()

    plt.subplot(421)
    plt.plot(cost_tr_full)
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.title('cost train (w regularization)', fontsize=medfont)

    plt.subplot(422)
    aux = numpy.asarray(cost_noreg)
    c_tr, c_val, c_val0 = aux.T[0], aux.T[1], aux.T[2]
    plt.plot(c_tr, 'b', label='tr set')
    plt.plot(c_val, 'g', label='val set')
    plt.plot(c_val0, 'r', label='val set 0')
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.legend(fontsize=medfont, loc=4)
    plt.title('cost w/o regularization', fontsize=medfont)

    plt.subplot(423)
    aux = numpy.asarray(acc_labfus)
    acc_val, acc_val0 = aux.T[0], aux.T[1]
    plt.plot(acc_val, 'b', label='deeplf')
    plt.plot(acc_val0, 'r', label='original')
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.legend(fontsize=medfont, loc=4)
    plt.title('label fusion accuracy', fontsize=medfont)

    plt.subplot(424)
    aux = numpy.asarray(activ)
    act_list = aux.T
    for i, act_mu_std_layer in enumerate(act_list):
        mu_act, std_act = act_mu_std_layer
        plt.plot(mu_act, c=distinct_colors[i], label='layer %d' % (i))
        plt.plot(numpy.array(mu_act) - numpy.array(std_act), ls='dashed', c=distinct_colors[i])
        plt.plot(numpy.array(mu_act) + numpy.array(std_act), ls='dashed', c=distinct_colors[i])
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.legend(fontsize=medfont, loc=4)
    plt.title('mean (std) activations', fontsize=medfont)

    # GAmma & Beta
    aux = numpy.asarray(bn_params)
    bn_list = aux.T
    gamma_list, beta_list = [], []
    for i, gamma_beta_layer in enumerate(bn_list):
        gamma_layer, beta_layer = gamma_beta_layer
        gamma_list.append(gamma_layer)
        beta_list.append(beta_layer)

    plt.subplot(425)
    for i, gamma_layer in enumerate(gamma_list):
        plt.plot(gamma_layer, c=distinct_colors[i], label='layer %d' % (i))
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.legend(fontsize=medfont)
    plt.title('gamma', fontsize=medfont)

    plt.subplot(426)
    aux = numpy.asarray(seg_stats)
    if aux.size > 0:
        dice_nl, dice_nlb, dice_dlf, epochs_dice = aux.T[0], aux.T[1], aux.T[2], aux.T[3]
        plt.plot(epochs_dice, dice_dlf, 'b', label='deeplf')
        plt.plot(epochs_dice, dice_nl, 'g', ls='dashed', label='nlwv')
        plt.plot(epochs_dice, dice_nlb, 'r', label='nlbeta')
        plt.xticks(fontsize=smallfont)
        plt.yticks(fontsize=smallfont)
        plt.legend(fontsize=medfont, loc=4)
        plt.title('MULTI-ATLAS segmentation dice', fontsize=medfont)

    plt.subplot(427)
    plt.plot(epochs, hours)
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.title('epoch vs. time', fontsize=medfont)

    plt.subplot(428)
    plt.axis('off')
    plt.text(0, .9, 'learn rate: %g' % (args.learning_rate[0]), fontsize=medfont, fontweight='bold')
    plt.text(0, .8, 'hidden layers: %d' % (args.num_hidden_layers[0]), fontsize=medfont, fontweight='bold')
    plt.text(0, .7, 'num units: %d' % (args.num_units[0]), fontsize=medfont, fontweight='bold')
    plt.text(0, .6, 'activation: %s' % (args.activation[0]), fontsize=medfont, fontweight='bold')
    plt.text(0, .5, 'L2 reg: %g' % (args.L2_reg[0]), fontsize=medfont, fontweight='bold')

    plt.text(0, .4, 'train batch size: %d' % (args.train_batch_size[0]), fontsize=medfont)
    plt.text(0, .3, 'num neighbors: %d' % (args.num_neighbors[0]), fontsize=medfont)
    plt.text(0, .2, 'patch radius: %d' % (args.patch_radius[0]), fontsize=medfont)
    plt.text(0, .1, 'patch norm: %s' % (args.patch_norm[0]), fontsize=medfont)
    plt.text(0, .0, 'search radius: %d' % (args.search_radius[0]), fontsize=medfont)
    plt.text(0, -.1, 'invalid updates: %d' % (invalid_updates), fontsize=medfont)

    plt.text(0.6, .9, 'fract inside: %0.3f' % (args.fract_inside[0]), fontsize=medfont)
    plt.text(0.6, .8, 'cert or dist max: %0.3f' % (args.cert_or_dist_max[0]), fontsize=medfont)
    plt.text(0.6, .7, 'cost: %s' % (args.cost[0]), fontsize=medfont)
    plt.text(0.6, .6, 'beta0: %0.5f' % (beta0), fontsize=medfont)

    plt.savefig(file)
    plt.close(fig)



def write_stats2(stats_fig, args, cost_u, epoch_u, cost_s, epoch_s, err_train, err_val, finetune_errs, invalid_updates):

    distinct_colors = plt.get_cmap('jet')(numpy.linspace(0, 1.0, len(cost_s) + 1))[:-1]

    fig = plt.figure(figsize=(8.27, 11.69))
    plt.clf()

    plt.subplot(321)
    plt.plot(epoch_u, cost_u)
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.title('PRE-TRAINING Reconstruction error (w reg)', fontsize=medfont)

    plt.subplot(322)
    for i, (cost, epoch, color) in enumerate(zip(cost_s, epoch_s, distinct_colors)):
        plt.plot(epoch, cost, c=color, label='%d' % i)
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.legend(fontsize=medfont)
    plt.title('FINE-TUNING Training cost', fontsize=medfont)

    plt.subplot(323)
    for i, (err_t, err_v, epoch, color) in enumerate(zip(err_train, err_val, epoch_s, distinct_colors)):
        plt.plot(epoch, err_t, c=color)
        plt.plot(epoch, err_v, c=color, ls='dashed')
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.title('FINE-TUNING Errors (training & validation)', fontsize=medfont)

    plt.subplot(324)
    bar_width = 0.65
    for i, (ft_err, color) in enumerate(zip(finetune_errs, distinct_colors)):
        plt.bar(i - bar_width/2., ft_err, bar_width, color=color)
    plt.xticks(fontsize=smallfont)
    plt.yticks(fontsize=smallfont)
    plt.title('Best errors', fontsize=medfont)

    plt.subplot(325)
    plt.axis('off')
    plt.text(0, .9, 'Num units: %d' % (args.num_units[0]), fontsize=medfont)
    plt.text(0, .8, 'Batch normalization' if args.batch_norm else ' ', fontsize=medfont)
    plt.text(0, .7, '%s' % (args.ae_type[0]), fontsize=medfont)
    plt.text(0, .6, 'reg_level: %0.3f' % (args.reg_level[0]), fontsize=medfont)
    plt.text(0, .5, 'invalid updates: %d' % (invalid_updates), fontsize=medfont)
    if finetune_errs:
        plt.text(0, .4, 'Best error: %0.3f (stage %d)' % (min(finetune_errs), finetune_errs.index(min(finetune_errs))), fontsize=medfont, fontweight='bold')

    plt.savefig(stats_fig)
    plt.close(fig)


