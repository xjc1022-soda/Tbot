import torch
import numpy as np
import argparse
import os
import time
import datetime

import tasks
from tbot import TBot
from utils import init_dl_program, pkl_save, name_with_datetime
import datautils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
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
    
    print('Loading data... ', end='')
    if args.loader == 'UCR':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UCR(args.dataset)
        
    elif args.loader == 'UEA':
        task_type = 'classification'
        train_data, train_labels, test_data, test_labels = datautils.load_UEA(args.dataset)
        
    elif args.loader == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
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
        raise ValueError(f"Unknown loader {args.loader}.")
        
    print('done')
    
    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
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
    
    t = time.time()
    
    model = TBot(
        input_dims=train_data.shape[-1],
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
            out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
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

# add a running example
# python -u main.py Chinatown UCR --loader UCR --batch-size 8 --repr-dims 320 --max-threads 8 --seed 42 --eval
# Dataset: Chinatown
# Arguments: Namespace(batch_size=8, dataset='Chinatown', epochs=None, eval=True, gpu=0, iters=None, loader='UCR', lr=0.001, max_threads=8, repr_dims=320, run_name='UCR', save_every=None, seed=42)
# cuda:0
# Loading data... done
# Epoch 1 loss: -2.8164639472961426
# Epoch 2 loss: -13.905488967895508
# Epoch 3 loss: -3.057619094848633
# Epoch 4 loss: -0.79825758934021
# Epoch 5 loss: -2.1611990928649902
# Epoch 6 loss: -2.8378591537475586
# Epoch 7 loss: -1.0815879106521606
# Epoch 8 loss: 5.286574363708496
# Epoch 9 loss: -10.371329307556152
# Epoch 10 loss: -2.6371264457702637
# Epoch 11 loss: 0.42556095123291016
# Epoch 12 loss: 6.107945919036865
# Epoch 13 loss: 2.7525644302368164
# Epoch 14 loss: 0.37131616473197937
# Epoch 15 loss: 7.795841217041016
# Epoch 16 loss: 7.1385650634765625
# Epoch 17 loss: 3.1102495193481445
# Epoch 18 loss: -2.874086856842041
# Epoch 19 loss: 4.188469886779785
# Epoch 20 loss: -0.6280419230461121
# Epoch 21 loss: 2.021824836730957
# Epoch 22 loss: -4.434417724609375
# Epoch 23 loss: -7.869085311889648
# Epoch 24 loss: 0.6798977851867676
# Epoch 25 loss: 2.5356507301330566
# Epoch 26 loss: -4.297179222106934
# Epoch 27 loss: 7.297275543212891
# Epoch 28 loss: -4.855055809020996
# Epoch 29 loss: -3.974665641784668
# Epoch 30 loss: -8.916358947753906
# Epoch 31 loss: 5.701718330383301
# Epoch 32 loss: -1.6803921461105347
# Epoch 33 loss: -5.390422344207764
# Epoch 34 loss: -2.591742992401123
# Epoch 35 loss: 2.2067902088165283
# Epoch 36 loss: 2.5787227153778076
# Epoch 37 loss: -0.18691407144069672
# Epoch 38 loss: -0.5377658009529114
# Epoch 39 loss: -0.7613030672073364
# Epoch 40 loss: -4.717661380767822

# Training time: 0:00:12.087501

# Evaluation result: {'acc': 0.7405247813411079, 'auprc': 0.9009963745487906}
# Finished.