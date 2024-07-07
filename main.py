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
    parser.add_argument('--epochs', type=int, default=40, help='The number of epochs')
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
    
    # TODO: 等待模型的实现再进行save checkpoint的实现
    # # TODO: 统一训练的单位
    # def save_checkpoint_callback(
    #     save_every=1,
    #     unit='epoch'
    # ):
    #     assert unit in ('epoch', 'iter')
    #     def callback(model, loss):
    #         n = model.n_epochs if unit == 'epoch' else model.n_iters
    #         if n % save_every == 0:
    #             model.save(f'{run_dir}/model_{n}.pkl')
    #     return callback
    
    # # decide whether to save the model every epoch/iteration
    # if args.save_every is not None:
    #     unit = 'epoch' if args.epochs is not None else 'iter'
    #     config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)
    
    # # create a directory to save the model and output
    # run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
    # os.makedirs(run_dir, exist_ok=True)
    
    # t = time.time()
    
    model = TBot(
        input_dims=train_data.shape[-1],
        device=device,
        **config
    )
    loss = model.fit(
        train_data,
        n_epochs=args.epochs,
        # TODO: n_iters=args.iters, 考虑是否要删除这个参数
        verbose=True
    )
    
    # model.save(f'{run_dir}/model.pkl')

    # t = time.time() - t
    # print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
    
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
        # pkl_save(f'{run_dir}/out.pkl', out)
        # pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        # print('Evaluation result:', eval_res)

    print("Finished.")
