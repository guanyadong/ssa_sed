# -*- coding: utf-8 -*-
# adaptive alpha 0.8
import argparse
import datetime
import inspect
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import random
from pprint import pprint
import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from torch import nn
from utilities.utilities import Mixup
from utilities.utils import do_mixup
from data_utils.Desed import DESED
from data_utils.DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from TestModel import _load_cnn_trans
from evaluation_measures import get_predictions, psds_score, compute_psds_from_operating_points, compute_metrics, get_predictions1
from models.CNN_TRANS import CNN_TRANS
import config as cfg
from utilities import ramps
from utilities.Logger import create_logger
from utilities.Scaler import ScalerPerAudio, Scaler
from utilities.utils import SaveBest, to_cuda_if_available, weights_init, AverageMeterSet, EarlyStopping, \
    get_durations_df
from utilities.ManyHotEncoder import ManyHotEncoder
from utilities.Transforms import get_transforms

def adjust_learning_rate(optimizer, rampup_value, rampdown_value=1):
    """ adjust the learning rate
    Args:
        optimizer: torch.Module, the optimizer to be updated
        rampup_value: float, the float value between 0 and 1 that should increases linearly
        rampdown_value: float, the float between 1 and 0 that should decrease linearly
    Returns:

    """
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    # We commented parts on betas and weight decay to match 2nd system of last year from Orange

    lr = rampup_value * (cfg.max_learning_rate ** rampdown_value)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    i = 0
    for ema_params, params in zip(ema_model.parameters(), model.parameters()):
        # i += 1
        # print(ema_params.data.shape)
        # print(params.data.shape)
        # if i==72:
        #     print(i)
        ema_params.data.mul_(alpha).add_(1 - alpha, params.data)


def train(train_loader, model, optimizer, c_epoch, ema_model=None, mask_weak=None, mask_strong=None,
          mask_weak_mixup =None, mask_strong_mixup = None, batch_size_mixup = None, adjust_lr=False, mixup=True):
    """ One epoch of a Mean Teacher model
    Args:
        train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
            Should return a tuple: ((teacher input, student input), labels)
        model: torch.Module, model to be trained, should return a weak and strong prediction
        optimizer: torch.Module, optimizer used to train the model
        c_epoch: int, the current epoch of training
        ema_model: torch.Module, student model, should return a weak and strong prediction
        mask_weak: slice or list, mask the batch to get only the weak labeled data (used to calculate the loss)
        mask_strong: slice or list, mask the batch to get only the strong labeled data (used to calcultate the loss)
        adjust_lr: bool, Whether or not to adjust the learning rate during training (params in config)
    """
    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    class_criterion = nn.BCELoss()
    consistency_criterion = nn.MSELoss()
    class_criterion, consistency_criterion = to_cuda_if_available(class_criterion, consistency_criterion)

    meters = AverageMeterSet()
    log.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()

    for i, ((batch_input, ema_batch_input), target) in enumerate(train_loader):

        global_step = c_epoch * len(train_loader) + i
        rampup_value = ramps.exp_rampup(global_step, cfg.n_epoch_rampup*len(train_loader))
        rampdown_value = ramps.exp_rampdown(global_step, cfg.n_epoch_rampdown*len(train_loader))

        if adjust_lr:
            adjust_learning_rate(optimizer, rampup_value, rampdown_value)
        meters.update('lr', optimizer.param_groups[0]['lr'])

        batch_input, ema_batch_input, target = \
            to_cuda_if_available(batch_input, ema_batch_input, target)

        # mixup
        if mixup == True:
            # mixup feature
            mixup_lambda = mixup_augmenter.get_lambda(batch_size=batch_size_mixup)

            mixup_lambda = torch.from_numpy(mixup_lambda).float()

            mixup_lambda = to_cuda_if_available(mixup_lambda)

            batch_input_mask_weak = do_mixup(batch_input[mask_weak_mixup], mixup_lambda[mask_weak_mixup])
            batch_input_mask_strong = do_mixup(batch_input[mask_strong_mixup], mixup_lambda[mask_strong_mixup])
            batch_input = torch.cat([batch_input_mask_weak, batch_input[12:24], batch_input_mask_strong])

            # mixup_lambda = mixup_augmenter.get_lambda(batch_size=batch_size_mixup)
            ema_batch_input_mask_weak = do_mixup(ema_batch_input[mask_weak_mixup], mixup_lambda[mask_weak_mixup])
            ema_batch_input_mask_strong = do_mixup(ema_batch_input[mask_strong_mixup], mixup_lambda[mask_strong_mixup])
            ema_batch_input = torch.cat([ema_batch_input_mask_weak, ema_batch_input[12:24],
                                     ema_batch_input_mask_strong])

            # mixup tag
            target_weak = target.max(-2)[0]
            target_mask_weak = do_mixup(target_weak[mask_weak_mixup], mixup_lambda[mask_weak_mixup])
            target_mask_strong = do_mixup(target[mask_strong_mixup], mixup_lambda[mask_strong_mixup])

        # batch_input, ema_batch_input, target, target_mask_weak, target_mask_strong = \
        #     to_cuda_if_available(batch_input, ema_batch_input, target, target_mask_weak, target_mask_strong)

        ema_batch_input = ema_batch_input.float()
        batch_input = batch_input.float()
        target_mask_weak = target_mask_weak.float()
        target_mask_strong = target_mask_strong.float()

        # Outputs
        strong_pred_ema, weak_pred_ema = ema_model(ema_batch_input, c_epoch)
        strong_pred_ema = strong_pred_ema.detach()
        weak_pred_ema = weak_pred_ema.detach()


        strong_pred, weak_pred = model(batch_input, c_epoch)

        loss = None

        # Weak BCE Loss
        if mask_weak is not None:
            weak_class_loss = class_criterion(weak_pred[mask_weak], target_mask_weak)
            ema_class_loss = class_criterion(weak_pred_ema[mask_weak], target_mask_weak)
            loss = weak_class_loss

            if i == 0:
                log.debug(f"target: {target.mean(-2)} \n Target_weak: {target_weak} \n "
                          f"Target weak mask: {target_mask_weak} \n "
                          f"Target strong mask: {target[mask_strong].sum(-2)}\n"
                          f"weak loss: {weak_class_loss} \t rampup_value: {rampup_value}"
                          f"tensor mean: {batch_input.mean()}")

            meters.update('weak_class_loss', weak_class_loss.item())
            meters.update('Weak EMA loss', ema_class_loss.item())

        # Strong BCE loss
        if mask_strong is not None:
            strong_class_loss = class_criterion(strong_pred[mask_strong], target_mask_strong)
            meters.update('Strong loss', strong_class_loss.item())

            strong_ema_class_loss = class_criterion(strong_pred_ema[mask_strong], target_mask_strong)
            meters.update('Strong EMA loss', strong_ema_class_loss.item())

            if loss is not None:
                loss += strong_class_loss
            else:
                loss = strong_class_loss

        # Teacher-student consistency cost
        if ema_model is not None:
            consistency_cost = cfg.max_consistency_cost * rampup_value
            meters.update('Consistency weight', consistency_cost)
            # Take consistency about strong predictions (all data)
            consistency_loss_strong = consistency_cost * consistency_criterion(strong_pred, strong_pred_ema)
            meters.update('Consistency strong', consistency_loss_strong.item())
            if loss is not None:
                loss += consistency_loss_strong
            else:
                loss = consistency_loss_strong
            meters.update('Consistency weight', consistency_cost)
            # Take consistency about weak predictions (all data)
            consistency_loss_weak = consistency_cost * consistency_criterion(weak_pred, weak_pred_ema)
            meters.update('Consistency weak', consistency_loss_weak.item())
            if loss is not None:
                loss += consistency_loss_weak
            else:
                loss = consistency_loss_weak

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        assert not loss.item() < 0, 'Loss problem, cannot be negative'
        meters.update('Loss', loss.item())

        # compute gradient and do optimizer step
        # loss += attention_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if ema_model is not None:
            update_ema_variables(model, ema_model, 0.999, global_step)

    epoch_time = time.time() - start
    log.info(f"Epoch: {c_epoch}\t Time {epoch_time:.2f}\t {meters}")
    return loss


def get_dfs(desed_dataset, nb_files=None, separated_sources=False):
    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
    audio_weak_ss = None
    audio_unlabel_ss = None
    audio_validation_ss = None
    audio_synthetic_ss = None
    if separated_sources:
        audio_weak_ss = cfg.weak_ss
        audio_unlabel_ss = cfg.unlabel_ss
        audio_validation_ss = cfg.validation_ss
        audio_synthetic_ss = cfg.synthetic_ss

    weak_df = desed_dataset.initialize_and_get_df(cfg.weak, audio_dir_ss=audio_weak_ss, nb_files=nb_files)
    unlabel_df = desed_dataset.initialize_and_get_df(cfg.unlabel, audio_dir_ss=audio_unlabel_ss, nb_files=nb_files)
    # Event if synthetic not used for training, used on validation purpose
    synthetic_df = desed_dataset.initialize_and_get_df(cfg.synthetic, audio_dir_ss=audio_synthetic_ss,
                                                       nb_files=nb_files, download=False)
    # evaluation feature
    evaluate_df = desed_dataset.initialize_and_get_df(cfg.eval_desed, audio_dir_ss=None, nb_files=nb_files, download=False)
    log.debug(f"synthetic: {synthetic_df.head()}")
    validation_df = desed_dataset.initialize_and_get_df(cfg.validation, audio_dir=cfg.audio_validation_dir,
                                                        audio_dir_ss=audio_validation_ss, nb_files=nb_files)
    # Divide synthetic in train and valid
    filenames_train = synthetic_df.filename.drop_duplicates().sample(frac=1, random_state=26)
    train_synth_df = synthetic_df[synthetic_df.filename.isin(filenames_train)]
    valid_synth_df = synthetic_df.drop(train_synth_df.index).reset_index(drop=True)
    # Put train_synth in frames so many_hot_encoder can work.
    #  Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    train_synth_df.onset = train_synth_df.onset * cfg.sample_rate // cfg.hop_size // pooling_time_ratio
    train_synth_df.offset = train_synth_df.offset * cfg.sample_rate // cfg.hop_size // pooling_time_ratio

    log.debug(valid_synth_df.event_label.value_counts())

    data_dfs = {"weak": weak_df,
                "unlabel": unlabel_df,
                "synthetic": synthetic_df,
                "train_synthetic": train_synth_df,
                "valid_synthetic": valid_synth_df,
                "validation": validation_df,
                "evaluation": evaluate_df,
                }

    return data_dfs


if __name__ == '__main__':

    for num in range(1):

        random.seed(520)
        os.environ['PYTHONHASHSEED'] = str(520)
        np.random.seed(520)
        torch.manual_seed(520)
        torch.cuda.manual_seed(520)
        torch.cuda.manual_seed_all(520)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        #
        logger = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=cfg.terminal_level)
        logger.info("Baseline 2020")
        logger.info(f"Starting time: {datetime.datetime.now()}")
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("-s", '--subpart_data', type=int, default=None, dest="subpart_data",
                            help="Number of files to be used. Useful when testing on small number of files.")

        parser.add_argument("-n", '--no_synthetic', dest='no_synthetic', action='store_true', default=False,
                            help="Not using synthetic labels during training")
        f_args = parser.parse_args()
        pprint(vars(f_args))

        reduced_number_of_data = f_args.subpart_data
        no_synthetic = f_args.no_synthetic

        if no_synthetic:
            add_dir_model_name = "_no_synthetic"
        else:
            add_dir_model_name = "_with_synthetic"

        store_dir = os.path.join("stored_data", "MeanTeacher" + add_dir_model_name)
        saved_model_dir = os.path.join(store_dir, "model")
        saved_pred_dir = os.path.join(store_dir, "predictions")
        os.makedirs(store_dir, exist_ok=True)
        os.makedirs(saved_model_dir, exist_ok=True)
        os.makedirs(saved_pred_dir, exist_ok=True)

        n_channel = 1
        add_axis_conv = 0

        n_layers = 7

        cnn_transformer_kwargs = {"n_in_channel": n_channel, "nclass": len(cfg.classes), "attention": True,
                       "activation": "cg",
                       "dropout": 0.5,
                       "kernel_size": n_layers * [3], "padding": n_layers * [1], "stride": n_layers * [1],
                       "nb_filters": [16,  32,  64,  128,  128, 128, 128],
                       "pooling": [[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
                       "normal_func": "sparsemax",   # or softmax
                       "sparsity": 1.3       # sparsity can be adjusted
                        }

        pooling_time_ratio = 4  # 2 * 2

        out_nb_frames_1s = cfg.sample_rate / cfg.hop_size / pooling_time_ratio

        median_window = max(int(cfg.median_window_s * out_nb_frames_1s), 1)

        # different median window
        median_window_short = int(cfg.median_window_short * out_nb_frames_1s)
        median_window_media = int(cfg.median_window_media * out_nb_frames_1s)
        median_window_long = int(cfg.median_window_long * out_nb_frames_1s)

        logger.debug(f"median_window: {median_window}")
        logger.debug(f"median_window_short: {median_window_short}")
        logger.debug(f"median_window_media: {median_window_media}")
        logger.debug(f"median_window_long: {median_window_long}")
        # ##############
        # DATA
        # ##############
        dataset = DESED(base_feature_dir=os.path.join(cfg.workspace, "dataset", "features"),
                        compute_log=False)

        # dfs = get_dfs(dataset, reduced_number_of_data)
        many_hot_encoder = ManyHotEncoder(cfg.classes, n_frames=cfg.max_frames // pooling_time_ratio)
        prepare = 0
        if prepare == 1:
            dfs = get_dfs(dataset, reduced_number_of_data)

            # Meta path for psds
            durations_synth = get_durations_df(cfg.synthetic)

            encod_func = many_hot_encoder.encode_strong_df

            # Normalisation per audio or on the full dataset
            if cfg.scaler_type == "dataset":
                transforms = get_transforms(cfg.max_frames, add_axis=add_axis_conv)
                weak_data = DataLoadDf(dfs["weak"], encod_func, transforms)
                unlabel_data = DataLoadDf(dfs["unlabel"], encod_func, transforms)
                train_synth_data = DataLoadDf(dfs["train_synthetic"], encod_func, transforms)
                scaler_args = []
                scaler = Scaler()
                # # Only on real data since that's our final goal and test data are real
                scaler.calculate_scaler(ConcatDataset([weak_data, unlabel_data, train_synth_data]))
                logger.debug(f"scaler mean: {scaler.mean_}")
            else:
                scaler_args = ["global", "min-max"]
                scaler = ScalerPerAudio(*scaler_args)

            transforms2 = get_transforms(cfg.max_frames, scaler, add_axis_conv,
                                        noise_dict_params={"mean": 0., "snr": cfg.noise_snr})

            transforms = get_transforms(cfg.max_frames, scaler, add_axis_conv,
                                        noise_dict_params={"mean": 0., "snr": cfg.noise_snr},
                                        time_move={"mean": 0, "time_std": 90, "apply_to_label": False, "max":270},
                                        freq_move={"mean": 0, "std": 8/3, "max": 8})
            transforms1 = get_transforms(cfg.max_frames, scaler, add_axis_conv,
                                        noise_dict_params={"mean": 0., "snr": cfg.noise_snr},
                                        time_move = {"mean":0, "time_std": 90, "apply_to_label":True, "max":270},
                                        freq_move={"mean": 0, "std": 8/3, "max": 8})
            transforms_valid = get_transforms(cfg.max_frames, scaler, add_axis_conv)

            weak_data = DataLoadDf(dfs["weak"], encod_func, transforms, in_memory=cfg.in_memory)
            unlabel_data = DataLoadDf(dfs["unlabel"], encod_func, transforms, in_memory=cfg.in_memory_unlab)
            train_synth_data = DataLoadDf(dfs["train_synthetic"], encod_func, transforms1, in_memory=cfg.in_memory)
            valid_synth_data = DataLoadDf(dfs["validation"], encod_func, transforms_valid,
                                          return_indexes=True, in_memory=cfg.in_memory)
            evaluation_data = DataLoadDf(dfs["evaluation"], encod_func, transforms_valid,
                                         return_indexes=True, in_memory=cfg.in_memory )

            logger.debug(f"len synth: {len(train_synth_data)}, len_unlab: {len(unlabel_data)}, len weak: {len(weak_data)}")
            mixup = True

            if not no_synthetic:
                list_dataset = [weak_data, unlabel_data, train_synth_data]
                batch_sizes = [cfg.batch_size // 4, cfg.batch_size // 2, cfg.batch_size // 4]
                if mixup ==False:
                    batch_sizes_mixup = [cfg.batch_size_mixup//3, cfg.batch_size//3, cfg.batch_size//3]
                else:
                    batch_sizes_mixup = [cfg.batch_size_mixup // 3, cfg.batch_size_mixup // 3, cfg.batch_size_mixup // 3]
                strong_mask = slice((3*cfg.batch_size)//4, cfg.batch_size)
                strong_mask_mixup = slice((2*cfg.batch_size_mixup)//3, cfg.batch_size_mixup)
            else:
                list_dataset = [weak_data, unlabel_data]
                batch_sizes = [cfg.batch_size // 4, 3 * cfg.batch_size // 4]
                strong_mask = None
            weak_mask = slice(batch_sizes[0])  # Assume weak data is always the first one
            weak_mask_mixup = slice(batch_sizes_mixup[0])

            concat_dataset = ConcatDataset(list_dataset)

            # Sampler
            sampler = MultiStreamBatchSampler(concat_dataset, batch_sizes=batch_sizes_mixup)

            # Data loader
            training_loader = DataLoader(concat_dataset, batch_sampler=sampler, num_workers=cfg.num_workers)
            valid_synth_loader = DataLoader(valid_synth_data, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
            evaluation_loader = DataLoader(evaluation_data, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)

            dict1 = {}
            dict1['dfs'] = dfs
            dict1['training_loader'] = training_loader
            dict1['valid_synth_loader'] = valid_synth_loader
            dict1['evaluation_loader'] = evaluation_loader
            dict1['scaler'] = scaler
            dict1['weak_mask'] = weak_mask
            dict1['strong_mask'] = strong_mask
            dict1['weak_mask_mixup'] = weak_mask_mixup
            dict1['mask_strong_mixup'] = strong_mask_mixup

            f1 = open('./stored_data/training_data.txt', 'wb')
            pickle.dump(dict1, f1)
            f1.close()

        else:
            f1 = open('./stored_data/training_data.txt', 'rb')
            dict1 = pickle.load(f1)
            f1.close()
            dfs = dict1['dfs']
            training_loader = dict1['training_loader']
            valid_synth_loader = dict1['valid_synth_loader']
            evaluation_loader = dict1['evaluation_loader']
            scaler = dict1['scaler']
            weak_mask = dict1['weak_mask']
            strong_mask = dict1['strong_mask']
            weak_mask_mixup = dict1['weak_mask_mixup']
            strong_mask_mixup = dict1['mask_strong_mixup']


        if cfg.scaler_type == "dataset":
            scaler_args = []
        else:
            scaler_args = ["global", "min-max"]

        mixup = True
        if mixup == True:
            mixup_augmenter = Mixup(mixup_alpha=1.)
        # ##############
        # Model
        # ##############

        resume = False

        if resume is False:
            crnn = CNN_TRANS(**cnn_transformer_kwargs)
            crnn = crnn.cuda()

            pytorch_total_params = sum(p.numel() for p in crnn.parameters() if p.requires_grad)
            logger.info(crnn)
            logger.info("number of parameters in the model: {}".format(pytorch_total_params))
            crnn.apply(weights_init)

            crnn_ema = CNN_TRANS(**cnn_transformer_kwargs)
            crnn_ema.apply(weights_init)
            for param in crnn_ema.parameters():
                param.detach_()

            optim_kwargs = {"lr": cfg.default_learning_rate, "betas": (0.9, 0.999)}
            optim = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
            bce_loss = nn.BCELoss()

            state = {
                'model': {"name": crnn.__class__.__name__,
                          'args': '',
                          "kwargs": cnn_transformer_kwargs,
                          'state_dict': crnn.state_dict()},
                'model_ema': {"name": crnn_ema.__class__.__name__,
                              'args': '',
                              "kwargs": cnn_transformer_kwargs,
                              'state_dict': crnn_ema.state_dict()},
                'optimizer': {"name": optim.__class__.__name__,
                              'args': '',
                              "kwargs": optim_kwargs,
                              'state_dict': optim.state_dict()},
                "pooling_time_ratio": pooling_time_ratio,
                "scaler": {
                    "type": type(scaler).__name__,
                    "args": scaler_args,
                    "state_dict": scaler.state_dict()},
                "many_hot_encoder": many_hot_encoder.state_dict(),
                "median_window": median_window,
                "median_window_short": median_window_short,
                "median_window_media": median_window_media,
                "median_window_long": median_window_long,
                "desed": dataset.state_dict()
            }
        else:
            model_fname = os.path.join(saved_model_dir, "baseline_epoch_424")
            state = torch.load(model_fname)
            crnn, crnn_ema = _load_cnn_trans(state)

        optim_kwargs = {"lr": cfg.default_learning_rate, "betas": (0.9, 0.999)}
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs)
        bce_loss = nn.BCELoss()

        save_best_cb = SaveBest("sup")
        if cfg.early_stopping is not None:
            early_stopping_call = EarlyStopping(patience=cfg.early_stopping, val_comp="sup", init_patience=cfg.es_init_wait)

        # ##############
        # Train
        # ##############
        results = pd.DataFrame(columns=["loss", "valid_synth_f1", "weak_metric", "global_valid"])
        for epoch in range(500):
            crnn.train()
            crnn_ema.train()

            crnn, crnn_ema = to_cuda_if_available(crnn, crnn_ema)

            loss_value = train(training_loader, crnn, optim, epoch,ema_model=crnn_ema,
                               mask_weak=weak_mask, mask_strong=strong_mask,
                               mask_weak_mixup=weak_mask_mixup, mask_strong_mixup=strong_mask_mixup,
                               batch_size_mixup = cfg.batch_size_mixup,
                               adjust_lr=True, mixup=mixup)

            # Validation
            crnn = crnn.eval()
            if epoch >= 400:
                logger.info("\n ### Valid synthetic metric ### \n")

                predictions = get_predictions(crnn, valid_synth_loader, many_hot_encoder.decode_strong, pooling_time_ratio,
                                              median_window=median_window, median_window_short = median_window_short,
                                              median_window_media = median_window_media, median_window_long = median_window_long,
                                              save_predictions=None)
                # Validation with synthetic data (dropping feature_filename for psds)
                valid = dfs["validation"].drop("feature_filename", axis=1)
                valid_synth_f1, _ = compute_metrics(predictions, valid)
                state['valid_metric'] = valid_synth_f1
            # Update state
            state['model']['state_dict'] = crnn.state_dict()
            state['model_ema']['state_dict'] = crnn_ema.state_dict()
            state['optimizer']['state_dict'] = optim.state_dict()
            state['epoch'] = epoch


            # Callbacks
            if cfg.checkpoint_epochs is not None and (epoch + 1) % cfg.checkpoint_epochs == 0:
                model_fname = os.path.join(saved_model_dir, "baseline_epoch_" + str(epoch))
                torch.save(state, model_fname)

            if cfg.save_best and epoch >= 400:
                if save_best_cb.apply(valid_synth_f1):
                    model_fname = os.path.join(saved_model_dir, "baseline_best_{}".format(num))
                    torch.save(state, model_fname)
                results.loc[epoch, "global_valid"] = valid_synth_f1
            results.loc[epoch, "loss"] = loss_value.item()

            if epoch >= 400:
                results.loc[epoch, "valid_synth_f1"] = valid_synth_f1

            if cfg.early_stopping:
                if early_stopping_call.apply(valid_synth_f1):
                    logger.warn("EARLY STOPPING")
                    break

        if cfg.save_best:
            model_fname = os.path.join(saved_model_dir, "baseline_best_{}".format(num))
            state = torch.load(model_fname)
            crnn, _ = _load_cnn_trans(state)
            logger.info(f"testing model: {model_fname}, epoch: {state['epoch']}")
        else:
            logger.info("testing model of last epoch: {}".format(cfg.n_epoch))
        results_df = pd.DataFrame(results).to_csv(os.path.join(saved_pred_dir, "results.tsv"),
                                                  sep="\t", index=False, float_format="%.4f")
        # ##############
        # Evaluation
        # ##############
        crnn.eval()

        predicitons_fname = os.path.join(saved_pred_dir, "baseline_evaluation.tsv")

        evaluation_labels_df = dfs["evaluation"].drop("feature_filename", axis=1)

        evaluation_predictions = get_predictions(crnn, evaluation_loader, many_hot_encoder.decode_strong,
                                            pooling_time_ratio, median_window=median_window,median_window_short = median_window_short,
                                            median_window_media = median_window_media, median_window_long = median_window_long,
                                            save_predictions=predicitons_fname)

        _, a = compute_metrics(evaluation_predictions, evaluation_labels_df)
        best_epoch = state['epoch']

        file_path = "./stored_data/output_sparse_attention.txt"
        with open(file_path, "a+") as f:
            f.write(str(best_epoch))
            f.write("\n")
            f.write(str(a))