import torch
import numpy as np
import argparse
import os
import time
import datetime

import tasks
from tranBot import TranBot
from utils import init_dl_program, pkl_save, name_with_datetime, cal_patch_num
import datautils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--d_model', type=int, default=64, help='The representation dimension (defaults to 320)')
    parser.add_argument('--n_heads', type=int, default=4, help='The number of heads in the multihead attention (defaults to 4)')
    parser.add_argument('--n_layers', type=int, default=2, help='The number of layers in the transformer encoder (defaults to 2)')
    parser.add_argument('--n_hierarchy', type=int, default=3, help='The number of hierarchies in the transformer encoder (defaults to 3)')
    parser.add_argument('--d_ff', type=int, default=64, help='The difference dimension (defaults to 64)')
    parser.add_argument('--patch_size', type=int, default=24, help='The patch size (defaults to 24)')
    parser.add_argument('--stride', type=int, default=24, help='The stride (defaults to 24)')
    parser.add_argument('--d_k', type=int, default=64, help='The key dimension (defaults to 64)')
    parser.add_argument('--d_v', type=int, default=64, help='The value dimension (defaults to 64)')
    parser.add_argument('--mask_ratio', type=float, default=0.2, help='The mask ratio (defaults to 0.2)')


    parser.add_argument('--epochs', type=int, help='The number of epochs')
    parser.add_argument('--iters', type=int, help='The number of iterations')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    args = parser.parse_args()
    
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    # Setting the random seed, some parameters for cuda, and the number of threads
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    print(device)
    
    print('Loading data...', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
        
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
        
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
        print(data.shape)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
        
    elif args.loader == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
        train_data = data[:, train_slice]
        
    elif args.loader == 'anomaly':
        task_type = 'anomaly_detection'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data = datautils.gen_ano_train_data(all_train_data)
        
    elif args.loader == 'anomaly_coldstart':
        task_type = 'anomaly_detection_coldstart'
        all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay = datautils.load_anomaly(args.dataset)
        train_data, _, _, _ = datautils.load_UCR('FordA')
        
    else:
        raise ValueError(f'Unknown loader: {args.loader}')
           
    print('done')
    
    config = dict(
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_hierarchy=args.n_hierarchy,
        d_ff=args.d_ff,
        patch_size=args.patch_size,
        stride=args.stride,
        d_k=args.d_k,
        d_v=args.d_v
    )
    
    def save_checkpoint_callback(
        save_every=1,
        unit='epoch'
    ):
        assert unit in ('epoch', 'iter')
        def callback(model, loss):
            n = model.n_epochs if unit == 'epoch' else model.n_iters
            if n % save_every == 0:
                model.save(f'{run_dir}/model_{n}.pkl')
        return callback
    
    # decide whether to save the model every epoch/iteration
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)
    
    # create a directory to save the model and output
    run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    n_patch, padding = cal_patch_num(train_data.shape[1], args.patch_size, args.stride)
    # print(n_patch)
    t = time.time()
    batch_size = min(args.batch_size, train_data.shape[0])
    model = TranBot(
        ts_dim=train_data.shape[-1],
        n_patch=n_patch,
        padding=padding,
        batch_size=batch_size,
        device=device,
        **config
    )
    loss = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    
    model.save(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
    
    # evaluation part
    if args.eval:
        if task_type == 'classification':
            out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
        elif task_type == 'forecasting':
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, univar=False)
        elif task_type == 'anomaly_detection':
            out, eval_res = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        elif task_type == 'anomaly_detection_coldstart':
            out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
        else:
            assert False
        pkl_save(f'{run_dir}/out.pkl', out)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        print('Evaluation result:', eval_res)

    print("Finished.")

# Example command:
# python -u main.py Chinatown UCR --loader UCR --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
# Dataset: Chinatown
# Arguments: Namespace(batch_size=8, dataset='Chinatown', epochs=None, eval=True, gpu=0, iters=None, loader='UCR', lr=0.001, max_threads=8, repr_dims=320, run_name='UCR', s_temp=2, save_every=None, seed=42, t_temp=5)
# cuda:0
# Loading data... done
# Epoch 1 loss: 0.2930034399032593
# Epoch 2 loss: -0.25440114736557007
# Epoch 3 loss: 4.086831569671631
# Epoch 4 loss: 1.028947353363037
# Epoch 5 loss: 3.284266471862793
# Epoch 6 loss: -2.250728130340576
# Epoch 7 loss: -2.2256038188934326
# Epoch 8 loss: -4.159049034118652
# Epoch 9 loss: 3.347928524017334
# Epoch 10 loss: 0.5131765604019165
# Epoch 11 loss: -2.178114652633667
# Epoch 12 loss: 0.5817365646362305
# Epoch 13 loss: 1.9178519248962402
# Epoch 14 loss: 0.7439870834350586
# Epoch 15 loss: -0.0094408318400383
# Epoch 16 loss: 6.794332504272461
# Epoch 17 loss: -0.68076491355896
# Epoch 18 loss: 2.53469181060791
# Epoch 19 loss: -1.638542652130127
# Epoch 20 loss: 1.385166049003601
# Epoch 21 loss: 0.028024137020111084
# Epoch 22 loss: 1.3539611101150513
# Epoch 23 loss: -2.062892436981201
# Epoch 24 loss: 1.0025149583816528
# Epoch 25 loss: 2.072673797607422
# Epoch 26 loss: -2.570943832397461
# Epoch 27 loss: 2.7207698822021484
# Epoch 28 loss: 6.780943393707275
# Epoch 29 loss: 0.5430023670196533
# Epoch 30 loss: -1.7411929368972778
# Epoch 31 loss: 2.5804855823516846
# Epoch 32 loss: -1.7874424457550049
# Epoch 33 loss: 1.829345941543579
# Epoch 34 loss: 2.987247943878174
# Epoch 35 loss: -0.3072805404663086
# Epoch 36 loss: -0.1635923683643341
# Epoch 37 loss: -1.4080185890197754
# Epoch 38 loss: 0.7717499732971191
# Epoch 39 loss: 1.0758179426193237
# Epoch 40 loss: 5.509942054748535

# Training time: 0:00:13.372400

# Evaluation result: {'acc': 0.6676384839650146, 'auprc': 0.9240271167453953, 'auroc': 0.8294454413398273, 'f1': 0.657504729871768}
# Finished.