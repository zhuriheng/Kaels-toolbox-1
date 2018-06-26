import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
import time

from mxnet.gluon.model_zoo.vision import mobilenet_v2_1_0 

def _get_lr_scheduler(lr,num_samples):
    # if '--lr-factor' not in args or float(args['--lr-factor']) >= 1:
    #     return (args['--lr'], None)
    batch_size = 512 
    epoch_size = int(num_samples/batch_size)
    begin_epoch = 0
    # step_epochs = [5,10,20,30,40,55]
    step_epochs = [3,6,9]
    lr_factor = 0.316228
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= lr_factor
    # if lr != float(args['--lr']):
    #     print('Adjust learning rate to %e for epoch %d' %
    #                  (lr, begin_epoch))
    steps = [epoch_size * (x - begin_epoch)
             for x in step_epochs if x - begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_factor)

def _get_iterators(data_train, data_dev, batch_size, data_shape=(3, 224, 224), resize=[-1, -1], dtype=None, data_index=None):
    '''
    define the function which returns the data iterators
    '''
    mean_r, mean_g, mean_b = 123.68,116.779,103.939
    std_r, std_g, std_b = 58.395,57.12,57.375
    # [max_random_scale, min_random_scale] = [float(x) for x in args['--resize-scale'].split(',')]
    # logger.info('Input normalization params: mean_rgb {}, std_rgb {}'.format([mean_r, mean_g, mean_b],[std_r, std_g, std_b]))
    # [resize_train, resize_dev] = resize

    label_name='softmax_label'
    # if not args['--use-svm']:
    #     if data_index:
    #         label_name = 'softmax-{}_label'.format(data_index)
    #     else:
    #         label_name = 'softmax_label'
    # else :
    #     label_name = 'svm_label'

    train = mx.io.ImageRecordIter(
        dtype=dtype,
        path_imgrec=data_train,
        # preprocess_threads=32,
        data_name='data',
        label_name=label_name,
        label_width=1,
        batch_size=batch_size,
        data_shape=data_shape,
        # resize=resize_train,
        # max_random_scale=max_random_scale,
        # min_random_scale=min_random_scale,
        shuffle=True,
        rand_crop=True,
        rand_mirror=True,
        mean_r=mean_r,
        mean_g=mean_g,
        mean_b=mean_b,
        std_r=std_r,
        std_g=std_g,
        std_b=std_b
        )
    val = mx.io.ImageRecordIter(
        dtype=dtype,
        path_imgrec=data_dev,
        # preprocess_threads=32,
        data_name='data',
        label_name=label_name,
        label_width=1,
        batch_size=batch_size,
        data_shape=data_shape,
        # resize=resize_dev,
        shuffle=False,
        rand_crop=False,
        rand_mirror=False,
        mean_r=mean_r,
        mean_g=mean_g,
        mean_b=mean_b,
        std_r=std_r,
        std_g=std_g,
        std_b=std_b
        )
    # logger.info("Data iters created successfully")
    return train, val

def _get_eval_metrics(lst_metrics):
    '''
    return multiple evaluation metrics
    '''
    all_metrics = ['accuracy', 'ce', 'f1',
                   'mae', 'mse', 'rmse', 'top_k_accuracy', 'hinge_loss']
    lst_child_metrics = []
    eval_metrics = mx.metric.CompositeEvalMetric()
    for metric in lst_metrics:
        assert metric in all_metrics, logger.error(
            'Invalid evaluation metric!')
        if metric == 'accuracy':
            lst_child_metrics.append(mx.metric.Accuracy())
        elif metric == 'ce':
            lst_child_metrics.append(mx.metric.CrossEntropy())
        elif metric == 'f1':
            lst_child_metrics.append(mx.metric.F1())
        elif metric == 'mae':
            lst_child_metrics.append(mx.metric.MAE())
        elif metric == 'mse':
            lst_child_metrics.append(mx.metric.MSE())
        elif metric == 'rmse':
            lst_child_metrics.append(mx.metric.RMSE())
        elif metric == 'top_k_accuracy':
            lst_child_metrics.append(
                mx.metric.TopKAccuracy(top_k=int(args['--top-k'])))
        elif metric == 'hinge_loss':
            lst_child_metrics.append(svm_metric.HingeLoss())
    for child_metric in lst_child_metrics:
        eval_metrics.add(child_metric)
    return eval_metrics

def load_model(ctx=None):
    sym, arg_params, aux_params = mx.model.load_checkpoint('/workspace/blademaster/model/xblade_v6_mobilenet_v2x/mobilenetv2-1_0', 0)

    sym = sym.get_internals()['flatten_output']
    net = gluon.nn.SymbolBlock(outputs=sym, inputs=mx.sym.var('data'))

    # Set the params
    net_params = net.collect_params()
    for param in arg_params:
        if param in net_params:
                net_params[param]._load_init(arg_params[param], ctx=ctx)
    for param in aux_params:
        if param in net_params:
                net_params[param]._load_init(aux_params[param], ctx=ctx)

    return net

def load_model_tmp(model_path,load_epoch=0,ctx=None):
    def _load_model_gluon(model_path, load_epoch=0):
        sym = mx.sym.load(model_path + '-symbol.json')
        save_dict = nd.load('%s-%04d.params' % (model_path, load_epoch))
        arg_params = dict()
        for k, v in save_dict.items():
            arg_params[k] = v
        return sym, arg_params
     
    # sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, load_epoch)
    # sym = sym.get_internals()['flatten_output']

    sym, arg_params = _load_model_gluon(model_path,load_epoch) 
    # print(sym.list_arguments())
    # print(arg_params.keys())
    aux_params = list()

    net = gluon.nn.SymbolBlock(outputs=sym, inputs=mx.sym.var('data'))

    # Set the params
    net_params = net.collect_params()
    for param in arg_params:
        if param in net_params:
                net_params[param]._load_init(arg_params[param], ctx=ctx)
    for param in aux_params:
        if param in net_params:
                net_params[param]._load_init(aux_params[param], ctx=ctx)
    return net

def evaluate(net, val_data, ctx):
    # ctx = [mx.gpu(0)]
    metric = mx.metric.Accuracy()
    metric.reset()
    val_data.reset()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            # outputs.append(net(x))
            outputs.append(nd.softmax(net(x)))
        metric.update(label, outputs)
    return metric.get()


def alter_train(nets, train_data, val_data, epochs, batch_size, ctx):
    # nets: [master,slave_1,slave_2]
    # train_data: [train_iter_1, train_iter_2]
    # val_data: [val_iter_1, val_iter_2]
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    # net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
    # kv = mx.kv.create(opt.kvstore)
    # train_data, val_data = get_data_iters(dataset, batch_size, kv.num_workers, kv.rank)
    kv = "device"
    lr = 0.01
    wd = 0.0005
    momentum = 0.9
    log_interval = 80
    num_samples = [931534,87667]
    # num_samples = [10000,10000]
    lr_scheduler_0 = _get_lr_scheduler(lr,num_samples[0]+num_samples[1])
    lr_scheduler_1 = _get_lr_scheduler(lr,num_samples[0])
    lr_scheduler_2 = _get_lr_scheduler(lr,num_samples[1])
    trainer_master = gluon.Trainer(nets[0].collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': wd, 'momentum': momentum, 'lr_scheduler': lr_scheduler_0}, kvstore = kv)
    trainer_slave_1 = gluon.Trainer(nets[1].collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': wd, 'momentum': momentum, 'lr_scheduler': lr_scheduler_1}, kvstore = kv)
    trainer_slave_2 = gluon.Trainer(nets[2].collect_params(), 'sgd',
                            {'learning_rate': lr, 'wd': wd, 'momentum': momentum, 'lr_scheduler': lr_scheduler_2}, kvstore = kv)
    # metric = mx.metric.Accuracy()
    metric = _get_eval_metrics(["accuracy","ce"])
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(epochs):
        # Train Branch with slave_1 on dataset 1
        tic = time.time()
        train_data[0].reset()
        metric.reset()
        btic = time.time()
        for i, batch in enumerate(train_data[0]):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = list() 
            Ls = list() 
            with autograd.record():
                for x, y in zip(data, label):
                    z = nets[1](nets[0](x))
                    L = loss(z, y)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(z)
                for L in Ls:
                    L.backward()
            trainer_master.step(batch.data[0].shape[0])    # batch_size
            trainer_slave_1.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            if i!=0 and log_interval and not (i)%log_interval:
                name, acc = metric.get()
                print('Epoch[{}] Batch [{}]\tSpeed: {:.2f} samples/sec\t{}={:.6f}\t{}={:.6f}'.format(epoch, i, batch_size/(time.time()-btic), name[0], acc[0], name[1], acc[1]))
                print('Epoch[{}] Batch [{}]\ttrainer_master: {},trainer_slave_1: {}'.format(epoch, i, trainer_master.learning_rate,trainer_slave_1.learning_rate))
            btic = time.time()

        name, acc = metric.get()
        print('[Epoch %d] slave-1 training: %s=%f'%(epoch, name[0], acc[0]))
        print('[Epoch %d] slave-1 training: %s=%f'%(epoch, name[1], acc[1]))
        print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        # val_acc = evaluate(ctx, val_data[0])
        name, val_acc = evaluate(lambda x: nets[1](nets[0](x)), val_data[0], ctx)
        print('[Epoch %d] slave-1 validation: %s=%f'%(epoch, name, val_acc))
        # nets[0].collect_params().save('/workspace/tmp/{}-master-{:0>4}.params'.format("tmp", (epoch+1)))
        # nets[1].collect_params().save('/workspace/tmp/{}-slave-1-{:0>4}.params'.format("tmp", (epoch+1)))
        # assert False, "debugging"

        # Train Branch with slave_2 on dataset 2
        tic = time.time()
        train_data[1].reset()
        metric.reset()
        btic = time.time()
        for i, batch in enumerate(train_data[1]):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = list() 
            Ls = list() 
            with autograd.record():
                for x, y in zip(data, label):
                    z = nets[2](nets[0](x))
                    L = loss(z, y)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(z)
                for L in Ls:
                    L.backward()
            trainer_master.step(batch.data[0].shape[0])    # batch_size
            trainer_slave_2.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            if i!=0 and log_interval and not (i)%log_interval:
                name, acc = metric.get()
                print('Epoch[{}] Batch [{}]\tSpeed: {:.2f} samples/sec\t{}={:.6f}\t{}={:.6f}'.format(epoch, i, batch_size/(time.time()-btic), name[0], acc[0], name[1], acc[1]))
                print('Epoch[{}] Batch [{}]\ttrainer_master: {},trainer_slave_2: {}'.format(epoch, i, trainer_master.learning_rate,trainer_slave_2.learning_rate))
            btic = time.time()
            btic = time.time()

        name, acc = metric.get()
        print('[Epoch %d] slave-2 training: %s=%f'%(epoch, name[0], acc[0]))
        print('[Epoch %d] slave-2 training: %s=%f'%(epoch, name[1], acc[1]))
        print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        # val_acc = evaluate(ctx, val_data[0])
        name, val_acc = evaluate(lambda x: nets[2](nets[0](x)), val_data[1], ctx)
        print('[Epoch %d] slave-2 validation: %s=%f'%(epoch, name, val_acc))
        # nets[0].collect_params().save('/workspace/blademaster/model/xblade_v6_mobilenet_v2x/{}-master-{:0>4}.params'.format("xb-v6", (epoch+1)))
        # nets[1].collect_params().save('/workspace/blademaster/model/xblade_v6_mobilenet_v2x/{}-slave-1-{:0>4}.params'.format("xb-v6", (epoch+1)))
        # nets[2].collect_params().save('/workspace/blademaster/model/xblade_v6_mobilenet_v2x/{}-slave-2-{:0>4}.params'.format("xb-v6", (epoch+1)))
        nets[0].collect_params().save('/workspace/tmp/{}-master-{:0>4}.params'.format("xb-v6", (epoch+1)))
        # nets[0].save_params('/workspace/tmp/{}-master-{:0>4}.params'.format("xb-v6", (epoch+1)))
        nets[0](mx.sym.Variable('data')).save('/workspace/tmp/xb-v6-master-symbol.json')
        # nets[0].export('/workspace/tmp/xb-v6-master', (epoch+1))
        nets[1].collect_params().save('/workspace/tmp/{}-slave-1-{:0>4}.params'.format("xb-v6", (epoch+1)))
        # nets[1](nets[0](mx.sym.Variable('data'))).save('/workspace/tmp/xb-v6-slave-1-symbol.json')
        nets[1](mx.sym.Variable('data')).save('/workspace/tmp/xb-v6-slave-1-symbol.json')
        nets[2].collect_params().save('/workspace/tmp/{}-slave-2-{:0>4}.params'.format("xb-v6", (epoch+1)))
        # nets[2](nets[0](mx.sym.Variable('data'))).save('/workspace/tmp/xb-v6-slave-2-symbol.json')
        nets[2](mx.sym.Variable('data')).save('/workspace/tmp/xb-v6-slave-2-symbol.json')



def alter_train_with_frozen_master(nets, train_data, val_data, epochs, batch_size, ctx):
    # nets: [master,slave_1,slave_2]
    # train_data: [train_iter_1, train_iter_2]
    # val_data: [val_iter_1, val_iter_2]
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    # net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
    # kv = mx.kv.create(opt.kvstore)
    # train_data, val_data = get_data_iters(dataset, batch_size, kv.num_workers, kv.rank)
    kv = "device"
    lr = 0.0001
    wd = 0.0005
    momentum = 0.9
    log_interval = 80
    num_samples = [931534,87667]
    # num_samples = [10000,10000]
    lr_scheduler_0 = _get_lr_scheduler(lr,num_samples[0]+num_samples[1])
    lr_scheduler_1 = _get_lr_scheduler(lr,num_samples[0])
    lr_scheduler_2 = _get_lr_scheduler(lr,num_samples[1])
    # trainer_master = gluon.Trainer(nets[0].collect_params(), 'sgd', {'learning_rate': 0}, kvstore = kv)
    trainer_slave_1 = gluon.Trainer(nets[1].collect_params(), 'sgd', {'learning_rate': lr, 'wd': wd, 'momentum': momentum, 'lr_scheduler': lr_scheduler_1}, kvstore = kv)
    trainer_slave_2 = gluon.Trainer(nets[2].collect_params(), 'sgd', {'learning_rate': lr, 'wd': wd, 'momentum': momentum, 'lr_scheduler': lr_scheduler_2}, kvstore = kv)
    # metric = mx.metric.Accuracy()
    metric = _get_eval_metrics(["accuracy","ce"])
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # freeze master
    # for p in nets[0].collect_params():
    #     nets[0].collect_params()[p].grad_req = 'null'
    # for p in nets[0].collect_params():
    #     nets[0].collect_params()[p].lr_mult = 0
    #     nets[0].collect_params()[p].wd_mult = 0

    name, val_acc = evaluate(lambda x: nets[1](nets[0](x)), val_data[0], ctx)
    print('[Epoch %d] slave-1 validation: %s=%f'%(-1, name, val_acc))

    for epoch in range(epochs):
        # Train Branch with slave_1 on dataset 1
        tic = time.time()
        train_data[0].reset()
        metric.reset()
        btic = time.time()
        # curr_loss = np.array(()) 
        # curr_loss = list() 

        name, val_acc = evaluate(lambda x: nets[1](nets[0](x)), val_data[0], ctx)
        print('[Epoch %d-] slave-1 validation: %s=%f'%(-1, name, val_acc))

        for i, batch in enumerate(train_data[0]):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = list() 
            Ls = list() 
            # for idx,x in enumerate(data):
            #     data[idx] = nets[0](x)
            with autograd.record():
                for x, y in zip(data, label):
                    # z = nets[1](nets[0](x))
                    # z = nets[1](x)
                    with autograd.predict_mode():
                        _ = nets[0](x)
                        # print('autograd mode:',autograd.is_training())
                    z = nets[1](_)
                    L = loss(z, y)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(nd.softmax(z))   # or ce=nan
                    # outputs.append(z)
                for L in Ls:
                    L.backward()
            # print(len(Ls))
            # curr_loss.append(np.array([nd.mean(x).asscalar() for x in Ls]).mean())
            # print("==> before:",nets[0].collect_params('first-3x3-conv-conv2d_weight').items()[0][1]._reduce())
            # nets[0].collect_params().zero_grad()    # freeze master
            # trainer_master.step(batch.data[0].shape[0])    # batch_size
            # print("==> after:",nets[0].collect_params('first-3x3-conv-conv2d_weight').items()[0][1]._reduce())
            trainer_slave_1.step(batch.data[0].shape[0])
            # print("outputs:",outputs)
            # print("label:",label)
            metric.update(label, outputs)
            if i!=0 and log_interval and not (i)%log_interval:
                name, acc = metric.get()
                print('Epoch[{}] Batch [{}]\tSpeed: {:.2f} samples/sec\t{}={:.6f}\t{}={:.6f}'.format(epoch, i, batch_size/(time.time()-btic), name[0], acc[0], name[1], acc[1]))
                # print('Epoch[{}] Batch [{}]\tSpeed: {:.2f} samples/sec\t{}={:.6f}\t{}={:.6f}'.format(epoch, i, batch_size/(time.time()-btic), name[0], acc[0], name[1], curr_loss[-1]))
                print('Epoch[{}] Batch [{}]\ttrainer_master: {},trainer_slave_1: {}'.format(epoch, i, 0, trainer_slave_1.learning_rate))
            btic = time.time()

        name, acc = metric.get()
        print('[Epoch %d] slave-1 training: %s=%f'%(epoch, name[0], acc[0]))
        print('[Epoch %d] slave-1 training: %s=%f'%(epoch, name[1], acc[1]))
        # print('[Epoch %d] slave-1 training: %s=%f'%(epoch, name[1], np.array(curr_loss).mean()))
        print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        # val_acc = evaluate(ctx, val_data[0])
        name, val_acc = evaluate(lambda x: nets[1](nets[0](x)), val_data[0], ctx)
        print('[Epoch %d] slave-1 validation: %s=%f'%(epoch, name, val_acc))
        name, val_acc = evaluate(lambda x: nets[2](nets[0](x)), val_data[1], ctx)
        print('[Epoch %d+] slave-2 validation: %s=%f'%(epoch, name, val_acc))
        # nets[0].collect_params().save('/workspace/tmp/{}-master-{:0>4}.params'.format("tmp", (epoch+1)))
        # nets[1].collect_params().save('/workspace/tmp/{}-slave-1-{:0>4}.params'.format("tmp", (epoch+1)))

        before = nets[0].collect_params("last-1x1-conv-conv2d_weight").items()[0][1]._reduce()
        _before = nets[0].collect_params('last-1x1-conv-batchnorm_moving_mean').items()[0][1]._reduce()
        __before = nets[0].collect_params('last-1x1-conv-batchnorm_gamma').items()[0][1]._reduce()
        before_ = nets[1].collect_params("sequential1_dense0_weight").items()[0][1]._reduce()
        before__ = nets[2].collect_params("sequential2_dense0_weight").items()[0][1]._reduce()

        # freeze master
        # for p in nets[0].collect_params():
        #         nets[0].collect_params()[p].grad_req = 'null'
        nets[0].collect_params()['last-1x1-conv-batchnorm_moving_mean'].grad_req = 'null'
        
        # Train Branch with slave_2 on dataset 2
        tic = time.time()
        train_data[1].reset()
        metric.reset()
        btic = time.time()
        for i, batch in enumerate(train_data[1]):
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = list() 
            Ls = list()

            # print('data[0]:',data[0])
            # for idx,x in enumerate(data):
            #     data[idx] = nets[0](x)
            # print('data[0]:',data[0])

            with autograd.record():
                for x, y in zip(data, label):
                    # z = nets[2](nets[0](x))
                    # z = nets[2](x)
                    with autograd.predict_mode():
                        _ = nets[0](x)
                        # print('autograd mode:',autograd.is_training())
                    z = nets[2](_)
                    L = loss(z, y)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(nd.softmax(z))   # or ce=nan
                    # outputs.append(z)
                for L in Ls:
                    L.backward()
            # nets[0].collect_params().zero_grad()    # freeze master
            # trainer_master.step(batch.data[0].shape[0])    # batch_size
            trainer_slave_2.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            if i!=0 and log_interval and not (i)%log_interval:
                name, acc = metric.get()
                print('Epoch[{}] Batch [{}]\tSpeed: {:.2f} samples/sec\t{}={:.6f}\t{}={:.6f}'.format(epoch, i, batch_size/(time.time()-btic), name[0], acc[0], name[1], acc[1]))
                print('Epoch[{}] Batch [{}]\ttrainer_master: {},trainer_slave_2: {}'.format(epoch, i, 0, trainer_slave_2.learning_rate))
            btic = time.time()

        name, acc = metric.get()
        print('[Epoch %d] slave-2 training: %s=%f'%(epoch, name[0], acc[0]))
        print('[Epoch %d] slave-2 training: %s=%f'%(epoch, name[1], acc[1]))
        print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        # val_acc = evaluate(ctx, val_data[0])
        name, val_acc = evaluate(lambda x: nets[2](nets[0](x)), val_data[1], ctx)
        print('[Epoch %d] slave-2 validation: %s=%f'%(epoch, name, val_acc))
        name, val_acc = evaluate(lambda x: nets[1](nets[0](x)), val_data[0], ctx)
        print('[Epoch %d+] slave-1 validation: %s=%f'%(epoch, name, val_acc))
        # nets[0].collect_params().save('/workspace/blademaster/model//xblade_v6_mobilenet_v2x/{}-master-{:0>4}.params'.format("xb-v6", (epoch+1)))
        # nets[1].collect_params().save('/workspace/blademaster/model//xblade_v6_mobilenet_v2x/{}-slave-1-{:0>4}.params'.format("xb-v6", (epoch+1)))
        # nets[2].collect_params().save('/workspace/blademaster/model//xblade_v6_mobilenet_v2x/{}-slave-2-{:0>4}.params'.format("xb-v6", (epoch+1)))


        after = nets[0].collect_params('last-1x1-conv-conv2d_weight').items()[0][1]._reduce()
        _after = nets[0].collect_params('last-1x1-conv-batchnorm_moving_mean').items()[0][1]._reduce()
        __after = nets[0].collect_params('last-1x1-conv-batchnorm_gamma').items()[0][1]._reduce()
        after_ = nets[1].collect_params("sequential1_dense0_weight").items()[0][1]._reduce()
        after__ = nets[2].collect_params("sequential2_dense0_weight").items()[0][1]._reduce()
        print('master before == after:', np.array_equal(before.asnumpy(),after.asnumpy()))
        print('master bn mean before == after:', np.array_equal(_before.asnumpy(),_after.asnumpy()))
        print('master bn gamma before == after:', np.array_equal(__before.asnumpy(),__after.asnumpy()))
        print('slave-1 before == after:', np.array_equal(before_.asnumpy(),after_.asnumpy()))
        print('slave-2 before == after:', np.array_equal(before__.asnumpy(),after__.asnumpy()))


        # nets[0].collect_params().save('/workspace/tmp/{}-master-phase-2-{:0>4}.params'.format("xb-v6", (epoch+1)))
        nets[0].collect_params().save('/workspace/blademaster/model/xblade_v6_mobilenet_v2x/{}-master-phase-2-{:0>4}.params'.format("xb-v6", (epoch+1)))
        # nets[0].save_params('/workspace/tmp/{}-master-{:0>4}.params'.format("xb-v6", (epoch+1)))
        nets[0](mx.sym.Variable('data')).save('/workspace/blademaster/model/xblade_v6_mobilenet_v2x/xb-v6-master-phase-2-symbol.json')
        # nets[0].export('/workspace/tmp/xb-v6-master', (epoch+1))
        nets[1].collect_params().save('/workspace/blademaster/model/xblade_v6_mobilenet_v2x/{}-slave-1-phase-2-{:0>4}.params'.format("xb-v6", (epoch+1)))
        # nets[1](nets[0](mx.sym.Variable('data'))).save('/workspace/tmp/xb-v6-slave-1-symbol.json')
        nets[1](mx.sym.Variable('data')).save('/workspace/blademaster/model/xblade_v6_mobilenet_v2x/xb-v6-slave-1-phase-2-symbol.json')
        nets[2].collect_params().save('/workspace/blademaster/model/xblade_v6_mobilenet_v2x/{}-slave-2-phase-2-{:0>4}.params'.format("xb-v6", (epoch+1)))
        # nets[2](nets[0](mx.sym.Variable('data'))).save('/workspace/tmp/xb-v6-slave-2-symbol.json')
        nets[2](mx.sym.Variable('data')).save('/workspace/blademaster/model/xblade_v6_mobilenet_v2x/xb-v6-slave-2-phase-2-symbol.json')


def main():
    ctx = [mx.gpu(i) for i in xrange(8)]

    batch_size = 64*8 

    # train_data1 = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), batch_size, shuffle=True)
    # test_data1 = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), batch_size, shuffle=False)
    # train_data2 = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform), batch_size, shuffle=True)
    # test_data2 = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), batch_size, shuffle=False)
    train, val = ["dummy" for x in range(2)], ["dummy" for x in range(2)]
    train[0], val[0] = _get_iterators('/workspace/datacache/blademaster/v26_se_resnext_50/recordio/train-256.rec', '/workspace/datacache/blademaster/v26_se_resnext_50/recordio/dev-224.rec', batch_size, (3, 224, 224))
    train[1], val[1] = _get_iterators('/workspace/datacache/blademaster/xfast_blade_v6/recordio/train-2-256.rec', '/workspace/datacache/blademaster/xfast_blade_v6/recordio/dev-2-224.rec', batch_size, (3, 224, 224))
    # train[0], val[0] = _get_iterators('/workspace/tmp/train-1.rec', '/workspace/tmp/train-1.rec', batch_size, (3, 224, 224))
    # train[1], val[1] = _get_iterators('/workspace/tmp/train-2.rec', '/workspace/tmp/train-2.rec', batch_size, (3, 224, 224))

    # load from model zoo
    # net_master = mobilenet_v2_1_0()
    # net_master.load_params('/workspace/tmp/tmp-0000.params',ctx)

    # load from symbol
    # net_pretrained = load_model(ctx=ctx)
    # net_pretrained = load_model_tmp('/workspace/blademaster/model/xblade_v6_mobilenet_v2x/mobilenetv2-1_0',load_epoch=0,ctx=ctx)
    net_pretrained = load_model_tmp('/workspace/blademaster/model/xblade_v6_mobilenet_v2x/xb-v6-master',load_epoch=33,ctx=ctx)
    # net_pretrained = load_model_tmp('/workspace/tmp/xb-v6-master-phase-2',load_epoch=1,ctx=ctx)
    # net_pretrained = load_model_tmp('/workspace/blademaster/model/xblade_v6_mobilenet_v2x/xb-v6-master-phase-2',load_epoch=10,ctx=ctx)
    # print(net_pretrained.params)
    net_master = gluon.nn.Sequential()
    # net_master = gluon.nn.HybridSequential()
    with net_master.name_scope():
        net_master.add(net_pretrained)
        # net_master.add(gluon.nn.Dense(64, activation='relu'))
    # net_master.hybridize()
    print(net_master.collect_params())

    # scratch
    # net_slave_1 = gluon.nn.Sequential()
    # with net_slave_1.name_scope():
    #     net_slave_1.add(gluon.nn.Dense(3))
    # # print(net_slave_1.collect_params())

    # net_slave_2 = gluon.nn.Sequential()
    # with net_slave_2.name_scope():
    #     net_slave_2.add(gluon.nn.Dense(48))
    # # print(net_slave_2.collect_params())

    # # net_master.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)
    # net_slave_1.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)
    # net_slave_2.collect_params().initialize(mx.init.Uniform(scale=0.1), ctx=ctx)

    # resume
    net_pretrained = load_model_tmp('/workspace/blademaster/model/xblade_v6_mobilenet_v2x/xb-v6-slave-1',load_epoch=33,ctx=ctx)
    # net_pretrained = load_model_tmp('/workspace/tmp/xb-v6-slave-1-phase-2',load_epoch=1,ctx=ctx)
    # net_pretrained = load_model_tmp('/workspace/blademaster/model/xblade_v6_mobilenet_v2x/xb-v6-slave-1-phase-2',load_epoch=10,ctx=ctx)
    net_slave_1 = gluon.nn.Sequential()
    with net_slave_1.name_scope():
        net_slave_1.add(net_pretrained)
    print(net_slave_1.collect_params())

    net_pretrained = load_model_tmp('/workspace/blademaster/model/xblade_v6_mobilenet_v2x/xb-v6-slave-2',load_epoch=33,ctx=ctx)
    # net_pretrained = load_model_tmp('/workspace/tmp/xb-v6-slave-2-phase-2',load_epoch=1,ctx=ctx)
    # net_pretrained = load_model_tmp('/workspace/blademaster/model/xblade_v6_mobilenet_v2x/xb-v6-slave-2-phase-2',load_epoch=10,ctx=ctx)
    net_slave_2 = gluon.nn.Sequential()
    with net_slave_2.name_scope():
        net_slave_2.add(net_pretrained)
    print(net_slave_2.collect_params())


    # softmax_cross_entropy_1 = gluon.loss.SoftmaxCrossEntropyLoss()
    # softmax_cross_entropy_2 = gluon.loss.SoftmaxCrossEntropyLoss()

    # trainer_master = gluon.Trainer(net_master.collect_params(), 'sgd', {'learning_rate': 0.05})
    # trainer_slave_1 = gluon.Trainer(net_slave_1.collect_params(), 'sgd', {'learning_rate': 0.05})
    # trainer_slave_2 = gluon.Trainer(net_slave_2.collect_params(), 'sgd', {'learning_rate': 0.05})
    
    print("#### Before Training ####")
    train_accuracy, test_accuracy = 0, 0
    test_accuracy = evaluate(lambda x: net_slave_1(net_master(x)), val[0], ctx)[1]
    # train_accuracy = evaluate(lambda x: net_slave_1(net_master(x)), train[0], ctx)[1]
    print("Mod1: Train_acc %s, Test_acc %s" % (train_accuracy, test_accuracy))
    test_accuracy = evaluate(lambda x: net_slave_2(net_master(x)), val[1], ctx)[1]
    # train_accuracy = evaluate(lambda x: net_slave_2(net_master(x)), train[1], ctx)[1]
    print("Mod2: Train_acc %s, Test_acc %s" % (train_accuracy, test_accuracy))


    nets = [net_master,net_slave_1,net_slave_2] 
    # alter_train(nets, train, val, 60, batch_size, ctx)
    alter_train_with_frozen_master(nets, train, val, 10, batch_size, ctx)


    # print("\n#### Shared+Module1 Training ####")
    # for e in range(epochs):
        # metric.reset()
        # Train Branch with slave_1 on dataset 1 
        # for i, (data, label) in enumerate(train_data1):
        #     data = data.as_in_context(ctx).reshape((-1, 784))
        #     label = label.as_in_context(ctx)
        #     with autograd.record():
        #         output = net_slave_1(net_master(data))
        #         loss = softmax_cross_entropy_1(output, label)
        #         loss.backward()
        #     trainer_master.step(batch_size)
        #     trainer_slave_1.step(batch_size)

        #     metric.update([label], [output])

        #     curr_loss = nd.mean(loss).asscalar()
        #     moving_loss = (curr_loss if ((i == 0) and (e == 0))
        #                 else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

        #     if i % 100 == 0 and i > 0:
        #         name, acc = metric.get()
        #         print('[Epoch %d Batch %d] Loss: %s Training: %s=%f'%(e, i, moving_loss, name, acc))

        # _, train_accuracy = metric.get()
        # _, test_accuracy = evaluate_accuracy(test_data1, lambda x: net_slave_1(net_master(x)))
        # print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s\n" % (e, moving_loss, train_accuracy, test_accuracy))


    # We expect the shared module to start where the first module finished
    # There will be a small accuracy decrease since one layer was not trained
    # _, test_accuracy = evaluate_accuracy(test_data2, lambda x: net_slave_2(net_master(x)))
    # _, train_accuracy = evaluate_accuracy(train_data2, lambda x: net_slave_2(net_master(x)))
    # print("\n#### Shared+Module2 Result after Mod1 Training ####")
    # print("Mod2: Train_acc %s, Test_acc %s" % (train_accuracy, test_accuracy))
    # print("\n#### Shared+Module2 Training ####")
    # for e in range(epochs):
    #     metric.reset()
    #     # Train Branch with slave_2 on dataset 2 
    #     for i, (data, label) in enumerate(train_data2):
    #         data = data.as_in_context(ctx).reshape((-1,784))
    #         label = label.as_in_context(ctx)
    #         with autograd.record():
    #             output = net_slave_2(net_master(data))
    #             loss = softmax_cross_entropy_2(output, label)
    #             loss.backward()
    #         trainer_master.step(batch_size)
    #         trainer_slave_2.step(batch_size)

    #         metric.update([label], [output])

    #         curr_loss = nd.mean(loss).asscalar()
    #         moving_loss = (curr_loss if ((i == 0) and (e == 0))
    #                     else (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss)

    #         if i % 100 == 0 and i > 0:
    #             name, acc = metric.get()
    #             print('[Epoch %d Batch %d] Loss: %s Training: %s=%f'%(e, i, moving_loss, name, acc))

    #     _, train_accuracy = metric.get()
    #     _, test_accuracy = evaluate_accuracy(test_data1, lambda x: net_slave_2(net_master(x)))
    #     print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s\n" % (e, moving_loss, train_accuracy, test_accuracy))

    # print("\n#### After Training ####")
    # _, test_accuracy = evaluate_accuracy(test_data1, lambda x: net_slave_1(net_master(x)))
    # _, train_accuracy = evaluate_accuracy(train_data1, lambda x: net_slave_1(net_master(x)))
    # print("Mod1: Train_acc %s, Test_acc %s" % (train_accuracy, test_accuracy))
    # _, test_accuracy = evaluate_accuracy(test_data2, lambda x: net_slave_2(net_master(x)))
    # _, train_accuracy = evaluate_accuracy(train_data2, lambda x: net_slave_2(net_master(x)))
    # print("Mod2: Train_acc %s, Test_acc %s" % (train_accuracy, test_accuracy))

if __name__ == '__main__':
    main()
