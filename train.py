#!/usr/bin/env python

# Modified Horovod MNIST example

import os,sys,time

import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
import graphics
from utils import ResultLogger
import nibabel as nib
from PIL import Image

learn = tf.contrib.learn

# Surpress verbose warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def _print(*args, **kwargs):
    if hvd.rank() == 0:
        print(*args, **kwargs)


def init_play(hps, model, logdir):

    def sample_batch(x_m, x_p,y, eps):
        n_batch = hps.local_batch_train * 2
        xs = []
        for i in range(int(np.ceil(len(eps) / n_batch))):
            xs.append(model.sample(
                x_m[i * n_batch:i * n_batch + n_batch],
                x_p[i * n_batch:i * n_batch + n_batch],
                y[i*n_batch:i*n_batch + n_batch],
                eps[i*n_batch:i*n_batch + n_batch]))
        return np.concatenate(xs)

    def draw_samples(epoch, y_i):
        if hvd.rank() != 0:
            return

        rows = hps.n_visual_row
        cols = rows
        n_batch = 80#rows*cols
        y = np.asarray([_y % hps.n_y for _y in (
            list(range(cols)) * rows)], dtype='int32')

        val_x_m = np.load(hps.sample_dir + 'm.npy')
        val_x_p = np.load(hps.sample_dir + 'p.npy')
        val_y = np.load(hps.sample_dir + 'label_' + hps.att + '.npy')

        # if hps.ycond:
        #     y = np.load(hps.sample_dir + 'label_' + str(hps.att_id) +'.npy')
        #     y = y[:n_batch]

        x_m = val_x_m[:n_batch]
        x_p = val_x_p[:n_batch]
        y = val_y[:n_batch]

        y = np.ones(n_batch) * y_i
        # temperatures = [0., .25, .5, .626, .75, .875, 1.] #previously
        temperatures = [0., .25, .5, .6, .7, .8, .9, 1.]

        x_samples = []
        x_samples.append(sample_batch(x_m, x_p, y, [.0]*n_batch))
        x_samples.append(sample_batch(x_m, x_p, y, [.25]*n_batch))
        # x_samples.append(sample_batch(x_m, x_p, y, [.5]*n_batch))
        # x_samples.append(sample_batch(x_m, x_p, y, [.6]*n_batch))
        # x_samples.append(sample_batch(x_m, x_p, y, [.7]*n_batch))
        # x_samples.append(sample_batch(x_m, x_p, y, [.8]*n_batch))
        # x_samples.append(sample_batch(x_m, x_p, y, [.9] * n_batch))
        # x_samples.append(sample_batch(x_m, x_p, y, [1.]*n_batch))
        # previously: 0, .25, .5, .625, .75, .875, 1.

        for i in range(len(x_samples)):
            x_sample = np.reshape(
                x_samples[i], [n_batch] + hps.pet_size)
            ############## save nii file #############
            for j in range(x_sample.shape[0]):
                nii = nib.Nifti1Image(x_sample[j,:,:,:], np.eye(4))
                nib.save(nii, logdir + 'epoch_{}_y_{}_sub_{}_sample_{}.nii'.format(epoch, y_i, j, i))
                x_slice = x_sample[j, 30, :, :]
                x = np.expand_dims(x_slice, axis=2)
                im = np.concatenate((x, x, x), axis=2)
                img = Image.fromarray(im.astype(np.uint8), 'RGB')
                img.save(logdir + 'epoch_{}_y_{}_sub_{}_sample_{}.png'.format(epoch, y_i, j, i))

            ##########################################
            # x_sample = x_sample[:,30,:,:]
            # graphics.save_raster(x_sample, logdir +
            #                      'epoch_{}_y_{}_sample_{}.png'.format(epoch, y_i, i))

    return draw_samples


def init_visualizations(hps, model, logdir, iterator):

    def sample_batch(x_m, x_p,y, eps):
        n_batch = hps.local_batch_train * 2
        xs = []
        for i in range(int(np.ceil(len(eps) / n_batch))):
            xs.append(model.sample(
                x_m[i * n_batch:i * n_batch + n_batch],
                x_p[i * n_batch:i * n_batch + n_batch],
                y[i*n_batch:i*n_batch + n_batch],
                eps[i*n_batch:i*n_batch + n_batch]))
        return np.concatenate(xs)

    def draw_samples(epoch):
        if hvd.rank() != 0:
            return

        rows = hps.n_visual_row
        cols = rows
        n_batch = rows*cols
        y = np.asarray([_y % hps.n_y for _y in (
            list(range(cols)) * rows)], dtype='int32')

        val_x_m = np.load(hps.sample_dir + 'm.npy')
        val_x_p = np.load(hps.sample_dir + 'p.npy')
        val_y = np.load(hps.sample_dir + 'label_' + hps.att + '.npy')

        # if hps.ycond:
        #     y = np.load(hps.sample_dir + 'label_' + str(hps.att_id) +'.npy')
        #     y = y[:n_batch]

        x_m = val_x_m[:n_batch]
        x_p = val_x_p[:n_batch]
        y = val_y[:n_batch]
        # temperatures = [0., .25, .5, .626, .75, .875, 1.] #previously
        temperatures = [0., .25, .5, .6, .7, .8, .9, 1.]

        x_samples = []
        x_samples.append(sample_batch(x_m, x_p, y, [.0]*n_batch))
        x_samples.append(sample_batch(x_m, x_p, y, [.25]*n_batch))
        x_samples.append(sample_batch(x_m, x_p, y, [.5]*n_batch))
        x_samples.append(sample_batch(x_m, x_p, y, [.6]*n_batch))
        x_samples.append(sample_batch(x_m, x_p, y, [.7]*n_batch))
        x_samples.append(sample_batch(x_m, x_p, y, [.8]*n_batch))
        x_samples.append(sample_batch(x_m, x_p, y, [.9] * n_batch))
        x_samples.append(sample_batch(x_m, x_p, y, [1.]*n_batch))
        # previously: 0, .25, .5, .625, .75, .875, 1.

        for i in range(len(x_samples)):
            x_sample = np.reshape(
                x_samples[i], [n_batch] + hps.pet_size)
            ############## save nii file #############
            for j in range(x_sample.shape[0]):
                nii = nib.Nifti1Image(x_sample[j,:,:,:], np.eye(4))
                nib.save(nii, logdir + 'epoch_{}_sub_{}_sample_{}.nii'.format(epoch, j, i))

            ##########################################
            x_sample = x_sample[:,30,:,:]
            graphics.save_raster(x_sample, logdir +
                                 'epoch_{}_sample_{}.png'.format(epoch, i))

    return draw_samples

# ===
# Code for getting data
# ===
def get_data(hps, sess):
    if hps.pet_size == -1:
        hps.pet_size = {'brain3D': [48, 64, 48]}[hps.problem]
    if hps.mri_size == -1:
        hps.mri_size = {'brain3D': [48, 64, 48]}[hps.problem]
    if hps.n_test == -1:
        hps.n_test = {'brain3D': 80}[hps.problem]
    hps.n_y = {'brain3D': 1}[hps.problem]

    if hps.data_dir == "":
        hps.data_dir = {'brain3D': './data_loaders/Brain_img/3D/'}[hps.problem]

    if hps.sample_dir == "":
        hps.sample_dir = {'brain3D': './data_loaders/brain3D_sample_'}[hps.problem]

    if hps.problem == 'lsun_realnvp':
        hps.rnd_crop = True
    else:
        hps.rnd_crop = False


    # Use anchor_size to rescale batch size based on image_size
    # s = hps.anchor_size
    hps.local_batch_train = hps.n_batch_train
    hps.local_batch_test = hps.n_batch_test
    hps.local_batch_init = hps.n_batch_init

    assert hps.n_visual_row % hps.local_batch_test == 0

    print("Rank {} Batch sizes Train {} Test {} Init {}".format(
        hvd.rank(), hps.local_batch_train, hps.local_batch_test, hps.local_batch_init))

    if hps.problem in ['brain3D']:
        hps.direct_iterator = True
        import data_loaders.get_data_brain_3D as v
        train_iterator, test_iterator, data_init = \
            v.get_data(sess, hps.data_dir, hvd.size(), hvd.rank(), hps.pmap, hps.fmap,
                       hps.local_batch_train, hps.local_batch_test,
                       hps.local_batch_init, hps.att)
    else:
        raise Exception()

    return train_iterator, test_iterator, data_init


def process_results(results):
    stats = ['loss', 'bits_x_u', 'bits_x_o', 'reg_loss_u', 'reg_loss_o']
    assert len(stats) == results.shape[0]
    res_dict = {}
    for i in range(len(stats)):
        res_dict[stats[i]] = "{:.4f}".format(results[i])
    return res_dict

def list2string(list):
    str1 = ', '.join("{:.2f}".format(e) for e in list)
    return str1


# Get number of training and validation iterations
def get_its(hps):
    # These run for a fixed amount of time. As anchored batch is smaller, we've actually seen fewer examples
    train_its = int(np.ceil(hps.n_train / (hps.n_batch_train * hvd.size())))
    test_its = int(np.ceil(hps.n_test / (hps.n_batch_train * hvd.size())))
    train_epoch = train_its * hps.n_batch_train * hvd.size()

    # Do a full validation run
    if hvd.rank() == 0:
        print(hps.n_test, hps.local_batch_test, hvd.size())

    # assert hps.n_test % (hps.local_batch_test * hvd.size()) == 0
    full_test_its = hps.n_test // (hps.local_batch_test * hvd.size())

    if hvd.rank() == 0:
        print("Train epoch size: " + str(train_epoch))
    return train_its, test_its, full_test_its


'''
Create tensorflow session with horovod
'''
def tensorflow_session():
    # Init session and params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Pin GPU to local rank (one GPU per process)
    config.gpu_options.visible_device_list = '0,1' #'str(hvd.local_rank())
    sess = tf.Session(config=config)
    return sess

def parse_file(f_name):
    f = open(f_name, "r")
    s = f.read()
    f.close()
    s_list = s.split("\n")
    s_list = s_list[:-1]
    s_list = [int(i) for i in s_list]
    return s_list

def write2file(f_name, d_list):
    f = open(f_name, "w")
    for i in d_list: 
        f.write(str(i) + '\n')
    f.close
    return 

def train(sess, model, hps, logdir, visualise):
    _print(hps)
    _print('Starting training. Logging to', logdir)
    _print('\nepoch     train   [t_loss, bits_x_u, bits_x_o, reg_u, reg_o]   '
           'test    n_processed    n_images    (ips, dtrain, dtest, dsample, dtot), msg \n')

    # Train
    sess.graph.finalize()
    # restore the state of last checkpoint
    if hps.restore_path != '':
        pars = parse_file(hps.restore_path+"/last_checkpoint.txt")
        _print('Loaded the lastest checkpoint (epoch ' + str(pars[0]) + ', n_p ' + str(pars[1]) + ') from '+ hps.restore_path+"/last_checkpoint.txt")
        e_h = pars[0] + 1
        n_processed = pars[1]
    else:
        n_processed = 0
        e_h = 1
    n_images = 0
    train_time = 0.0
    test_loss_best = 999999

    if hvd.rank() == 0:
        train_logger = ResultLogger(logdir + "train.txt", **hps.__dict__)
        test_logger = ResultLogger(logdir + "test.txt", **hps.__dict__)

    tcurr = time.time()
    for epoch in range(e_h, hps.epochs):

        t = time.time()

        train_results = []
        for it in range(hps.train_its):

            # Set learning rate, linearly annealed from 0 in the first hps.epochs_warmup epochs.
            lr = hps.lr * min(1., n_processed /
                              (hps.n_train * hps.epochs_warmup))

            # Run a training step synchronously.
            _t = time.time()
            # checkpoint =model.train(lr)
            # print(checkpoint)
            train_results += [model.train(lr)]
            print(train_results[-1][-4:])

            if hps.verbose and hvd.rank() == 0:
                _print(n_processed, time.time()-_t, train_results[-1])
                sys.stdout.flush()

            # Images seen wrt anchor resolution
            n_processed += hvd.size() * hps.n_batch_train
            # Actual images seen at current resolution
            n_images += hvd.size() * hps.local_batch_train

        train_results = np.mean(np.asarray(train_results), axis=0)
        train_results = train_results[:-4]

        dtrain = time.time() - t
        ips = (hps.train_its * hvd.size() * hps.local_batch_train) / dtrain
        train_time += dtrain

        if hvd.rank() == 0:
            train_logger.log(epoch=epoch, n_processed=n_processed, n_images=n_images, train_time=int(
                train_time), **process_results(train_results))

        if epoch % hps.epochs_full_valid == 0:
            test_results = []
            msg = ''

            t = time.time()
            # model.polyak_swap()

            if epoch % hps.epochs_full_valid == 0:
                # Full validation run
                for it in range(hps.full_test_its):
                    test_results += [model.test()]
                test_results = np.mean(np.asarray(test_results), axis=0)

                test_results = test_results[:-4]

                if hvd.rank() == 0:
                    test_logger.log(epoch=epoch, n_processed=n_processed,
                                    n_images=n_images, **process_results(test_results))

                    # Save checkpoint
                    if True:#test_results[0] < test_loss_best:
                        test_loss_best = test_results[0]
                        model.save(logdir  +"/model_best_loss.ckpt")
                        write2file(logdir+"/last_checkpoint.txt", [epoch, n_processed])
                        msg += ' *'

            dtest = time.time() - t

            # Sample
            t = time.time()
            if epoch == 1 or epoch == 10 or epoch % hps.epochs_full_sample == 0:
                visualise(epoch)
            dsample = time.time() - t

            if hvd.rank() == 0:
                dcurr = time.time() - tcurr
                tcurr = time.time()
                train_results = list2string(train_results)
                test_results = list2string(test_results)
                _print("{:<10} [{:<20}] [{:<20}] {:>10} {:>10} ({:.1f}  {:.1f}  {:.1f}  {:.1f}  {:.1f})".format(
                    epoch, train_results,test_results, n_processed,n_images, ips,  dtrain, dtest,  dsample, dcurr), msg)

            # model.polyak_swap()

    if hvd.rank() == 0:
        _print("Finished!")


def infer(sess, model, hps, iterator):
    # Example of using model in inference mode. Load saved model using hps.restore_path
    # Can provide x, y from files instead of dataset iterator
    # If model is uncondtional, always pass y = np.zeros([bs], dtype=np.int32)

    if hps.direct_iterator:
        iterator = iterator.get_next()

    xs = []
    recxs = []
    zs = []
    gt_xs = []
    xs_m = []
    ys = []

    for it in range(hps.full_test_its):
        if hps.direct_iterator:
            # replace with x, y, attr if you're getting CelebA attributes, also modify get_data
            x_m, x_p, y = sess.run(iterator)
        else:
            x_m, x_p, y = iterator()

        ys.append(y)
        # y=[1]
        # z = model.encode(x, y)

        if hps.eps_std < 100.0:
            eps_std = [hps.eps_std] * hps.n_batch_test
            x = model.sample(x_m, x_p, y, eps_std)
        else:
            eps_std = model.encode(x, y)
            x = model.decode(x_m, x_p, y, eps_std)

        xs.append(x)
        gt_xs.append(x_p * 255)
        xs_m.append(x_m * 255)
        #recxs.append(rec_x)
        # zs.append(z)
        # io.imshow(x[0].astype(np.uint8))
        # io.show()

    x_ = np.concatenate(xs, axis=0)
    x_gt = np.concatenate(gt_xs, axis=0)
    x_ms = np.concatenate(xs_m, axis=0)
    # recx_ = np.concatenate(recxs, axis=0)
    # z = np.concatenate(zs, axis=0)
    np.save(hps.logdir + '/x.npy', x_)
    np.save(hps.logdir + '/x_gt.npy', x_gt)
    np.save(hps.logdir + '/x_m.npy', x_ms)

    # np.save(hps.logdir + '/mri_id.npy', x_ms)
    with open(hps.logdir + '/pet_label_list.txt', 'w') as f:
        for i, item in enumerate(ys):
            f.write(str(i)+" %s\n" % item)
    # np.save(hps.logdir + '/recx.npy', recx_)
    # np.save('logs/z.npy', z)
    return zs

def main(hps):

    # Initialize Horovod.
    hvd.init()

    # Create tensorflow session
    sess = tensorflow_session()

    # Download and load dataset.
    tf.set_random_seed(hvd.rank() + hvd.size() * hps.seed)
    np.random.seed(hvd.rank() + hvd.size() * hps.seed)
    if hps.n_train is None:
        hps.n_train = { 'brain3D':726}[hps.problem]

    # Get data and set train_its and valid_its
    train_iterator, test_iterator, data_init = get_data(hps, sess)
    hps.train_its, hps.test_its, hps.full_test_its = get_its(hps)

    # Create log dir
    logdir = os.path.abspath(hps.logdir) + "/"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    # Create model
    import model

    model = model.model(sess, hps, train_iterator, test_iterator, data_init)

    # Initialize visualization functions
    visualise = init_visualizations(hps, model, logdir, test_iterator)

    # visualise(0)

    # play = init_play(hps, model, logdir)
    # play(0, 1.0)
    # play(0, 0.9)
    # play(0, 0.8)
    # play(0, 0.7)
    # play(0, 0.6)
    # play(0, 0.5)
    # play(0, 0.4)
    # play(0, 0.3)
    # play(0, 0.4)
    # play(0, 0.2)
    # play(0, 0.0)


    if not hps.inference:
        # Perform training
        train(sess, model, hps, logdir, visualise)
    else:
        infer(sess, model, hps, test_iterator)


if __name__ == "__main__":

    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--restore_path", type=str, default='./logs',
                        help="Location of checkpoint to restore")
    parser.add_argument("--inference", default=False, #action="store_true",
                        help="Use in inference mode")
    parser.add_argument("--logdir", type=str,
                        default='./logs', help="Location to save logs")


    # Dataset hyperparams:
    parser.add_argument("--problem", type=str, default='brain3D',
                        help="Problem (brain3D")
    parser.add_argument("--att",  type=str, default='age',
                        help="Problem (group/adas/age/apoe/cdr/dxbl/gender/mmse/ravlt")
    parser.add_argument("--data_dir", type=str, default='',
                        help="Location of data")
    parser.add_argument("--sample_dir", type=str, default='',
                        help="Location of val data")
    parser.add_argument("--dal", type=int, default=1,
                        help="Data augmentation level: 0=None, 1=Standard, 2=Extra")

    # Dataset processing params:

    # New dataloader params
    parser.add_argument("--fmap", type=int, default=2,
                        help="# Threads for parallel file reading")
    parser.add_argument("--pmap", type=int, default=16,
                        help="# Threads for parallel map")

    # Optimization hyperparams:
    # Optimization hyperparams:
    parser.add_argument("--n_train", type=int, help="Train epoch size")
    parser.add_argument("--n_test", type=int, default=-1, help="Valid epoch size")
    parser.add_argument("--n_batch_train", type=int,
                        default=1, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=1, help="Minibatch size")
    parser.add_argument("--n_visual_row", type=int,
                        default=6, help="Minibatch size")
    parser.add_argument("--n_batch_init", type=int, default=1,
                        help="Minibatch size for data-dependent init")
    parser.add_argument("--optimizer", type=str,
                        default="adamax", help="adam or adamax")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--beta1", type=float, default=.9, help="Adam beta1")
    parser.add_argument("--polyak_epochs", type=float, default=1,
                        help="Nr of averaging epochs for Polyak and beta2")
    parser.add_argument("--weight_decay", type=float, default=1.,
                        help="Weight decay. Switched off by default.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total number of training epochs")
    parser.add_argument("--epochs_warmup", type=int,
                        default=10, help="Warmup epochs")
    parser.add_argument("--epochs_full_valid", type=int,
                        default=1, help="Epochs between valid")
    parser.add_argument("--epochs_full_sample", type=int,
                        default=3, help="Epochs between full scale sample")
    parser.add_argument("--gradient_checkpointing", type=int,
                        default=1, help="Use memory saving gradients")

    # Model hyperparams:
    parser.add_argument("--pet_size", type=int,
                        default=-1, help="PET size")
    parser.add_argument("--mri_size", type=int,
                        default=-1, help="MRI size")
    parser.add_argument("--anchor_size", type=int, default=32,
                        help="Anchor size for deciding batch size")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=8,
                        help="Depth of network")
    parser.add_argument("--weight_y", type=float, default=0.01,
                        help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--cut_off", type=float, default=1.0,
                        help="Cut_off value in dicrminators")
    parser.add_argument("--weight_lambda", type=float, default=0.001,
                        help="Weight of log p(x_o|x_u) in weighted loss")
    parser.add_argument("--n_bits_x", type=int, default=0,
                        help="Number of bits of x")
    parser.add_argument("--n_levels", type=int, default=4,
                        help="Number of levels")

    # Synthesis/Sampling hyperparameters:
    parser.add_argument("--n_sample", type=int, default=1,
                        help="minibatch size for sample")

    # Ablation
    parser.add_argument("--learntop", action="store_true",
                        help="Learn spatial prior")
    parser.add_argument("--ycond", default=True, #action="store_true",
                        help="Use y conditioning")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--flow_permutation", type=int, default=2,
                        help="Type of flow. 0=reverse (realnvp), 1=shuffle, 2=invconv (ours)")
    parser.add_argument("--flow_coupling", type=int, default=1,
                        help="Coupling type: 0=additive, 1=affine")
    parser.add_argument("--central_crop", action="store_true",
                        help="Use other conditioning")
    parser.add_argument("--eps_std", type=float, default=0.4,
                        help="control the standard deviation")

    parser.add_argument("--n_l", type=int, default=4,
                        help="mlp basic layers")

    # parser.add_argument("--attr_value", type=int, default=1,
    #                     help="attributes value")
    # parser.add_argument('--bbox', nargs='+', type=int, default=[50, 45, 128, 128],
    #                     help="[begin_v, begin_h, width, height] ex. 90 49 128 128")
    hps = parser.parse_args()  # So error if typo
    main(hps)
