# multivar
python -u train.py electricity forecast_multivar --loader forecast_csv --repr-dims 320 --max-threads 8 --seed 42 --eval

python -u main.py ETTh1 test --loader forecast_csv --d_model 64 --max-threads 8 --seed 42 --eval

# multivar testing result in 1/7/2024
"""
Evaluation result: {3548974990844727}}
    'ours': {
        24: {
            'norm': {'MSE': 0.28560859960356133, 'MAE': 0.3743418346937081},
            'raw': {'MSE': 10580976.309338834, 'MAE': 353.3317975309586}
        },
        48: {
            'norm': {'MSE': 0.3117815542288609, 'MAE': 0.3918274853776107},
            'raw': {'MSE': 11768697.575496037, 'MAE': 370.93935807728997}
        },
        168: {
            'norm': {'MSE': 0.3386149457921282, 'MAE': 0.4110006876939884},
            'raw': {'MSE': 14502399.946747126, 'MAE': 398.148201574026}
        },
        336: {
            'norm': {'MSE': 0.35524216947736553, 'MAE': 0.42377517305178386},
            'raw': {'MSE': 16274082.279494977, 'MAE': 416.29448698322483}
        },
        720: {
            'norm': {'MSE': 0.3771496556438049, 'MAE': 0.4393651760261609},
            'raw': {'MSE': 18935242.14616922, 'MAE': 442.3744529490107}
        }
    },
    'ts2vec_infer_time': 410.83311128616333,
    'lr_train_time': {
        24: 19.735604524612427,
        48: 16.169066905975342,
        168: 41.41940188407898,
        336: 31.106695890426636,
        720: 81.28691720962524
    },
    'lr_infer_time': {
        24: 1.6816163063049316,
        48: 1.0271875858306885,
        168: 2.0906612873077393,
        336: 2.0546231269836426,
        720: 3.3548974990844727
    }
}
Finished.
"""

# univar
python -u train.py electricity forecast_univar --loader forecast_csv_univar --repr-dims 320 --max-threads 8 --seed 42 --eval
