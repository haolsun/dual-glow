from utils import *
import horovod.tensorflow as hvd

# n_l = 1
'''
f_loss: function with as input the (x,y,reuse=False), and as output a list/tuple whose first element is the loss.
'''


def abstract_model_xy(sess, hps, feeds, train_iterator, test_iterator, data_init, lr, f_loss):

    # == Create class with static fields and methods
    class m(object):
        pass
    m.sess = sess
    m.feeds = feeds
    m.lr = lr

    # === Loss and optimizer
    loss_train, stats_train = f_loss(train_iterator, True)
    all_params = tf.trainable_variables()

    if not hps.inference:  # computing gradients during training time
        with tf.device('/gpu:0'):
            if hps.gradient_checkpointing == 1:
                from memory_saving_gradients import gradients
                gs = gradients(loss_train, all_params)
            else:
                gs = tf.gradients(loss_train, all_params)

            optimizer = {'adam': optim.adam, 'adamax': optim.adamax,
                         'adam2': optim.adam2}[hps.optimizer]

            train_op, polyak_swap_op, ema = optimizer(
                all_params, gs, alpha=lr, hps=hps)



        if hps.direct_iterator:
            m.train = lambda _lr: sess.run([train_op, stats_train], {lr: _lr})[1]
        else:
            def _train(_lr):
                _x_m, _x_p, _y = train_iterator()
                return sess.run([train_op, stats_train], {feeds['x_m']: _x_m,
                                                          feeds['x_p']: _x_p,
                                                          feeds['y']: _y, lr: _lr})[1]

            m.train = _train

        m.polyak_swap = lambda: sess.run(polyak_swap_op)

        # === Saving and restoring (moving average)
        saver_ema = tf.train.Saver(ema.variables_to_restore())
        m.save_ema = lambda path: saver_ema.save(
            sess, path, write_meta_graph=False)


    # === Testing
    loss_test, stats_test = f_loss(test_iterator, False, reuse=True)
    if hps.direct_iterator:
        m.test = lambda: sess.run(stats_test)
    else:
        def _test():
            _x_m, _x_p, _y = test_iterator()
            return sess.run(stats_test, {feeds['x_m']: _x_m,
                                         feeds['x_p']: _x_p,
                                         feeds['y']: _y})
        m.test = _test

    # === Saving and restoring
    saver = tf.train.Saver()
    m.save = lambda path: saver.save(sess, path, write_meta_graph=False)
    m.restore = lambda path: saver.restore(sess, path)

    # === Initialize the parameters
    if hps.restore_path != '':
        m.restore(hps.restore_path+'/model_best_loss.ckpt')
    elif hps.inference:
        m.restore(hps.logdir + '/model_best_loss.ckpt')
    else:
        with Z.arg_scope([Z.get_variable_ddi, Z.actnorm], init=True):
            results_init = f_loss(None, True, reuse=True)
        sess.run(tf.global_variables_initializer())
        sess.run(results_init, {feeds['x_m']: data_init['x_m'],
                                feeds['x_p']: data_init['x_p'],
                                feeds['y']: data_init['y']})
    sess.run(hvd.broadcast_global_variables(0))

    return m


def codec(hps):

    def encoder(name, z, objective, y, z_prior=None):
        with tf.variable_scope(name):
            eps = []
            z_list = []
            for i in range(hps.n_levels):
                z, objective = revnet2d(str(i), z, objective, hps)
                if i < hps.n_levels - 1:
                    if z_prior is not None:
                        z, z2, objective, _eps = split2d("pool" + str(i), hps.n_l, z, y, z_prior[i], objective=objective)
                    else:
                        z, z2, objective, _eps = split2d("pool" + str(i), hps.n_l, z, y, objective=objective)
                    eps.append(_eps)
                    z_list.append(z2)
            z_list.append(z) # append z finally
        return z_list, objective, eps

    def decoder(name, y, z, z_provided=None, eps=[None]*hps.n_levels, eps_std=None, z_prior=None):
        with tf.variable_scope(name):
            for i in reversed(range(hps.n_levels)):
                if i < hps.n_levels - 1:
                    if eps is not None: eps_ = eps[i]
                    else: eps_ = None
                    if z_prior is not None:
                        if z_provided is not None:
                            z = split2d_reverse("pool" + str(i), hps.n_l, z, y, z_provided[i],  eps=eps_, eps_std=eps_std,
                                                z_prior=z_prior[i])
                        else:
                            z = split2d_reverse("pool" + str(i), hps.n_l, z, y, z_provided=None, eps=eps_, eps_std=eps_std, z_prior=z_prior[i])

                    else:
                        if z_provided is not None:
                            z = split2d_reverse("pool" + str(i), hps.n_l, z, y,  z_provided[i],  eps=eps_, eps_std=eps_std)
                        else:
                            z = split2d_reverse("pool" + str(i), hps.n_l, z,  y, z_provided=None, eps=eps_, eps_std=eps_std)

                z, _ = revnet2d(str(i), z, 0, hps, reverse=True)

        return z

    return encoder, decoder


def prior(name, top_shape, hps, y, z_prior=None):
    # p_cond: using z_prior

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        n_z = top_shape[-1]
        h = tf.zeros([top_shape[0]]+top_shape[1:3]+[2*n_z])
        if hps.learntop:
            h = Z.conv2d_zeros('p', h, 2*n_z)

        if y is not None:
            temp_v = Z.linear_zeros("y_emb", y, n_z*2)
            h += tf.reshape(temp_v, [-1, 1, 1,  n_z * 2])

        mean = h[:, :, :, :n_z]
        logsd = h[:, :, :, n_z:]

        ######### embedding the z_prior ##############
        if z_prior is not None:
            # w = tf.get_variable("W_prior", [1, 1, n_z, n_z * 2], tf.float32,
            #                      initializer=tf.zeros_initializer())
            # h -= tf.nn.conv2d(z_prior, w, strides=[1, 1, 1, 1], padding='SAME')
            #h += Z.myMLP(3, z_prior, n_z, n_z * 2)
            mean, logsd = Z.condFun(mean, logsd, z_prior, hps.n_l)
        #############################################

        pz = Z.gaussian_diag(mean, logsd)

    def logp(z1):
        objective = pz.logp(z1)
        return objective

    def sample(eps=None, eps_std=None):
        if eps is not None:
            # Already sampled eps. Don't use eps_std
            z = pz.sample2(eps)
        elif eps_std is not None:
            # Sample with given eps_std
            z = pz.sample2(pz.eps * tf.reshape(eps_std, [-1, 1, 1, 1]))
        else:
            # Sample normally
            z = pz.sample

        return z

    def eps(z1):
        return pz.get_eps(z1)

    return logp, sample, eps, mean, logsd





def model(sess, hps, train_iterator, test_iterator, data_init):

    # Only for decoding/init, rest use iterators directly
    with tf.name_scope('input'):
        X_m = tf.placeholder(tf.float32, [None] + hps.mri_size, name='image_input')
        X_p = tf.placeholder(tf.float32, [None] + hps.pet_size, name='image_ouput')
        Y = tf.placeholder(tf.float32, [None], name='label')
        lr = tf.placeholder(tf.float32, None, name='learning_rate')

    encoder, decoder = codec(hps)
    hps.n_bins = 2. ** hps.n_bits_x

    def preprocess(x_o, x_u):
        # rescale the data from -0.5 to 0.5
        x_u = x_u / hps.n_bins - .5
        x_o = x_o / hps.n_bins - .5
        if not hps.inference:
            x_u += tf.random_uniform(tf.shape(x_u), 0, 1. / 256)
            x_o += tf.random_uniform(tf.shape(x_o), 0, 1. / 256)

        return x_u, x_o

    # postprocessing ...............................
    def postprocess(x):
        return tf.clip_by_value(tf.floor((x + .5) * hps.n_bins * (255. / hps.n_bins)), 0, 255)
        #return tf.clip_by_value(tf.floor((x + .5) * hps.n_bins * (255. / hps.n_bins)), 0, 255)
        # return tf.floor((x + .5) * hps.n_bins * (255. / hps.n_bins))

    # cut-off l1_loss
    def losses(pred, label, type='l1'):
        if type=='l1':
            l = tf.losses.absolute_difference(pred, label)
        elif type=='crossEntropy':
            l = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label,logits=pred)
        # if cut_off:
        #     return tf.clip_by_value(l1, 0, hps.cut_off)
        # else:
        #     return l1
        return l

    def _f_loss(x_m, x_p, y, is_training, reuse=False):

        with tf.variable_scope('model', reuse=reuse):
            if hps.ycond:
                y_onehot = tf.expand_dims(y, 1)#tf.cast(tf.one_hot(y, hps.n_y, 1, 0), 'float32')
                # y_onehot = tf.cast(tf.one_hot(y, hps.n_y, 1, 0), 'float32')
                if hps.att in ['age', 'group']:
                    y_onehot -= 0.5
            else:
                y_onehot = None

            # Discrete -> Continuous
            x_u, x_o= preprocess(x_m, x_p)
            # dequantize data
            # z += tf.random_uniform(tf.shape(z), 0, 1. / hps.n_bins)

            objective_u = tf.zeros_like(x_u, dtype='float32')[:, 0, 0, 0]
            objective_u += - np.log(256.) * np.prod(Z.int_shape(x_u)[1:])

            objective_o = tf.zeros_like(x_o, dtype='float32')[:, 0, 0, 0]
            objective_o += - np.log(256.) * np.prod(Z.int_shape(x_o)[1:])


            ############# Encode #################
            # observed
            z_o = Z.squeeze2d(x_o, 2)  # > 16x16x12
            zs_o, objective_o, eps_o = encoder('m_o', z_o, objective_o, y=None)
            z_o = zs_o[-1]
            z_2_o = zs_o[:-1]

            # unobserved
            z_u = Z.squeeze2d(x_u, 2)  # > 16x16x12
            zs_u, objective_u, _ = encoder('m_u', z_u, objective_u, y=None, z_prior=z_2_o)
            z_u = zs_u[-1]

            ############# Prior #################
            # unobserved
            hps.top_shape1 = Z.int_shape(z_u)[1:]
            top_shape1 = [tf.shape(z_u)[0]] + hps.top_shape1
            logp_u, _, _ ,_,_= prior("prior_u", top_shape1, hps, y=y_onehot, z_prior=None)  ## input for prior_u : z_o, y
            objective_u += logp_u(z_u)
            # observed
            hps.top_shape2 = Z.int_shape(z_o)[1:]
            top_shape2 = [tf.shape(z_o)[0]] + hps.top_shape2
            logp_o, _, _eps_o,_ ,_= prior("prior_o", top_shape2, hps, y=None, z_prior=None)  ## input for prior_o : z_u, y    without prior   !!!!!!!!!!
            objective_o += logp_o(z_o)
            eps_o.append(_eps_o(z_o))

            ######## Generative loss ############
            # for unobserved
            nobj_u = - objective_u
            bits_x_u = nobj_u / (np.log(2.) * int(x_u.get_shape()[1]) * int(
                x_u.get_shape()[2]) * int(x_u.get_shape()[3]))  # bits per subpixel.
            # for observed
            nobj_o = - objective_o
            bits_x_o = nobj_o / (np.log(2.) * int(x_o.get_shape()[1]) * int(
                x_o.get_shape()[2]) * int(x_o.get_shape()[3]))  # bits per subpixel
            #######################################

            # Predictive loss
            if hps.weight_y > 0 and hps.ycond:
                # z_u_f = Z.list_unsqueeze3d(zs_u)  # assemble
                y_u_logits = Z.linear_MLP('discriminator_u', z_u, out_final=hps.n_y)

                regression_loss_o_list = []
                for i in range(len(zs_o)):
                    if i == len(zs_o)-1 :
                        use_grl = False
                    else:
                        use_grl = True
                    y_o_logits = Z.linear_MLP('discriminator_O_' + str(i), zs_o[i], out_final=hps.n_y, use_grl=use_grl)
                    regression_loss_o_list.append(losses(y_onehot, y_o_logits, type='l1') / np.log(2.))

                # Regression loss
                # bits_y = tf.zeros_like(bits_x_u)
                regression_loss_u = losses(y_onehot, y_u_logits, type='l1') / np.log(2.)
                regression_loss_o = sum(regression_loss_o_list) / len(regression_loss_o_list)


                # y_predicted = tf.argmax(y_logits, 1, output_type=tf.int32)
                # classification_error = 1 - \
                #     tf.cast(tf.equal(y_predicted, y), tf.float32)
            else:
                # bits_y = tf.zeros_like(bits_x_u)
                regression_loss_u = 0
                regression_loss_o = 0

        return bits_x_u, regression_loss_u, regression_loss_o, bits_x_o, \
               regression_loss_o_list

    def f_loss(iterator, is_training, reuse=False):
        if hps.direct_iterator and iterator is not None:
            x_m, x_p, y = iterator.get_next()
        else:
            x_m, x_p, y = X_m, X_p, Y

        bits_x_u, reg_loss_u, reg_loss_o, bits_x_o, reg_o_list= _f_loss(x_m, x_p, y, is_training, reuse)
        local_loss = bits_x_u + hps.weight_lambda * bits_x_o +  hps.weight_y * (reg_loss_u + reg_loss_o)

        stats = [local_loss, bits_x_u, bits_x_o, reg_loss_u, reg_loss_o] + reg_o_list
        global_stats = Z.allreduce_mean(
            tf.stack([tf.reduce_mean(i) for i in stats]))

        return tf.reduce_mean(local_loss), global_stats

    feeds = {'x_m': X_m, 'x_p': X_p, 'y': Y}
    m = abstract_model_xy(sess, hps, feeds, train_iterator,
                          test_iterator, data_init, lr, f_loss)

    # === Sampling function
    def f_sample(y, z_prior, z_o_m, eps_std):
        with tf.variable_scope('model', reuse=True):
            if hps.ycond:
                # y_onehot = tf.cast(tf.one_hot(y, hps.n_y, 1, 0), 'float32')
                y_onehot = tf.expand_dims(y, 1)
            else:
                y_onehot = None
            top_shape = [tf.shape(z_prior)[0]] + hps.top_shape1
            _, sample, _ ,_ ,_= prior("prior_u", top_shape, hps, y_onehot, z_prior=None)
            z = sample(eps_std=eps_std)
            z = decoder("m_u", y=None, z=z, z_prior=z_o_m, eps_std=eps_std)
            z = Z.unsqueeze2d(z, 2)  # 8x8x12 -> 16x16x3
            x = postprocess(z)

        return x

    ###### Get the prior from the observed #################
    with tf.variable_scope('model', reuse=True):
        _, z_o= preprocess(X_m, X_p)
        z_o = Z.squeeze2d(z_o, 2)  # > 16x16x12
        objective_o = tf.zeros_like(z_o, dtype='float32')[:, 0, 0, 0]
        #objective += - np.log(hps.n_bins) * np.prod(Z.int_shape(z_o)[1:])
        zs_o, _, _ = encoder('m_o', z_o, objective_o, y=None)
        z_o = zs_o[-1]
        z_o_m = zs_o[:-1]
    z_prior = z_o
    #####################################

    m.eps_std = tf.placeholder(tf.float32, [None], name='eps_std')
    x_u_sampled = f_sample(Y, z_prior, z_o_m, m.eps_std)
    x_sampled = x_u_sampled


    def sample(_x_m, _x_p, _y, _eps_std):
        return m.sess.run(x_sampled, {X_m:_x_m, X_p:_x_p, Y: _y, m.eps_std: _eps_std})
    m.sample = sample

    return m
