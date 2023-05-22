import pdb
import os
import csv
import pandas as pd
import numpy as np
import torch
import argparse
from sklearn.svm import SVC, SVR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from counterfactual_gan import CounterfactualGAN, CounterfactualGANMultimodal, SelfTrainingCounterfactualGAN
from utils import load_airbnb_data, load_clothing_data, get_metrics
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-split', default='08')
    parser.add_argument('--label-type', default='synthetic')
    parser.add_argument('--dataset', default='airbnb')
    parser.add_argument('--modalities', default='all')
    parser.add_argument('--representation', default='embeds')
    parser.add_argument('--embeds-type', default='finetuned')
    parser.add_argument('--num-clusters', type=int, default=5)
    parser.add_argument('--model', default='SVM')
    parser.add_argument('--noise', default='strong')
    parser.add_argument('--text-cols', default='text')
    parser.add_argument('--labelgen-model', default='_lr')
    parser.add_argument('--modality-prop', default='060202')
    parser.add_argument('--task', default='classification')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-gen-iters', type=int, default=500)
    parser.add_argument('--num-discr-iters', type=int, default=8)
    parser.add_argument('--num-selftrain-iters', type=int, default=1)
    parser.add_argument('--out-file', type=str)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--irm', action='store_true')
    parser.add_argument('--irm-generator', action='store_true')
    parser.add_argument('--counterfactual-invariance', action='store_true')
    parser.add_argument('--counterfactual-invariance-generator', action='store_true')
    parser.add_argument('--separate-discriminators', action='store_true')
    parser.add_argument('--discriminator-t', action='store_true')
    parser.add_argument('--no-counterfactual', action='store_true')
    parser.add_argument('--lm-gen', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--simple-ipw', action='store_true')
    parser.add_argument('--self-train', action='store_true')
    parser.add_argument('--share-weights', action='store_true')
    parser.add_argument('--pre-scaling', action='store_true')
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--strong-bias', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot-oracle', action='store_true')
    parser.add_argument('--model-dir', type=str)
    parser.add_arguemtn('--results-dir', type=str)
    args = parser.parse_args()

    return args

def split_text_image(data, df_cols):
    df = pd.DataFrame(data, columns=df_cols)
    df_text = df['listing_text'].values
    df_image = df['picture_path'].values
    df.drop(['listing_text', 'picture_path'], axis=1, inplace=True)
    df = torch.tensor(df.to_numpy(dtype=float)).to(device)

    return (df, df_text, df_image)

start_time = time.time()

args = get_args()

label_split = args.label_split
modalities = args.modalities
representation = args.representation
noise = args.noise
labelgen_model = args.labelgen_model
if labelgen_model == None:
    labelgen_model = ''
modality_prop = args.modality_prop
save = args.save
load = args.load
no_counterfactual = args.no_counterfactual
if args.counterfactual_invariance == None:
    args.counterfactual_invariance = False
if args.counterfactual_invariance_generator == None:
    args.counterfactual_invariance_generator = False
if args.lm_gen == None:
    args.lm_gen = False
if args.share_weights == None:
    args.share_weights = False
    
print('#########################\n')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

if modality_prop == '060202':
    modality_stem = ''
else:
    modality_stem = '_{}'.format(modality_prop)

if args.dataset == 'airbnb':
    load_data = load_airbnb_data
elif args.dataset == 'clothing_review':
    load_data = load_clothing_data

if representation == 'embeds' and modalities == 'all':
    X_train, X_train_text, X_train_image, X_train_heldout, X_train_heldout_text, X_train_heldout_image, t_train, t_train_heldout, y_train, y_train_heldout, X_cols = load_data(
        split='train', data='biased', label_split=label_split, modalities=modalities, representation=representation, noise=noise, labelgen_model=labelgen_model,
        modality_prop=modality_prop, output_for='gan', embeds_type=args.embeds_type, data_dir=args.data_dir, label_type=args.label_type, task=args.task,
        text_cols=args.text_cols, num_clusters=args.num_clusters, model_dir=args.model_dir)
    X_test_biased, X_test_biased_text, X_test_biased_image, _, _, _, t_test_biased, _, y_test_biased, _, X_cols = load_data(
        split='test', data='biased', label_split=label_split, modalities=modalities, representation=representation, noise=noise, labelgen_model=labelgen_model,
        modality_prop=modality_prop, output_for='gan', embeds_type=args.embeds_type, data_dir=args.data_dir, label_type=args.label_type, task=args.task,
        text_cols=args.text_cols, num_clusters=args.num_clusters, model_dir=args.model_dir)
    X_test_unbiased, X_test_unbiased_text, X_test_unbiased_image, _, _, _, t_test_unbiased, _, y_test_unbiased, _, _ = load_data(
        split='test', data='unbiased', label_split=label_split, modalities=modalities, representation=representation, noise=noise, labelgen_model=labelgen_model,
        modality_prop=modality_prop, output_for='gan', embeds_type=args.embeds_type, data_dir=args.data_dir, label_type=args.label_type, task=args.task,
        text_cols=args.text_cols, num_clusters=args.num_clusters, model_dir=args.model_dir)

elif representation == 'embeds' and modalities == 'text':
    X_train, X_train_text, X_train_heldout, X_train_heldout_text, t_train, t_train_heldout, y_train, y_train_heldout, X_cols = load_data(
        split='train', data='biased', label_split=label_split, modalities=modalities, representation=representation, noise=noise, labelgen_model=labelgen_model,
        modality_prop=modality_prop, output_for='ganite', embeds_type=args.embeds_type, data_dir=args.data_dir, label_type=args.label_type, task=args.task,
        text_cols=args.text_cols, num_clusters=args.num_clusters)
    X_test_biased, X_test_biased_text, _, _, t_test_biased, _, y_test_biased, _, X_cols = load_data(
        split='test', data='biased', label_split=label_split, modalities=modalities, representation=representation, noise=noise, labelgen_model=labelgen_model,
        modality_prop=modality_prop, output_for='ganite', embeds_type=args.embeds_type, data_dir=args.data_dir, label_type=args.label_type, task=args.task,
        text_cols=args.text_cols, num_clusters=args.num_clusters)
    X_test_unbiased, X_test_unbiased_text, _, _, t_test_unbiased, _, y_test_unbiased, _, _ = load_data(
        split='test', data='unbiased', label_split=label_split, modalities=modalities, representation=representation, noise=noise, labelgen_model=labelgen_model,
        modality_prop=modality_prop, output_for='ganite', embeds_type=args.embeds_type, data_dir=args.data_dir, label_type=args.label_type, task=args.task,
        text_cols=args.text_cols, num_clusters=args.num_clusters)

else:
    X_train, X_train_heldout, t_train, t_train_heldout, y_train, y_train_heldout, X_cols = load_data(
        split='train', data='biased', label_split=label_split, modalities=modalities, representation=representation, noise=noise, labelgen_model=labelgen_model,
        modality_prop=modality_prop, output_for='gan', embeds_type=args.embeds_type, data_dir=args.data_dir, label_type=args.label_type, task=args.task,
        text_cols=args.text_cols, num_clusters=args.num_clusters, model_dir=args.model_dir)
    X_test_biased, _, t_test_biased, _, y_test_biased, _, X_cols = load_data(
        split='test', data='biased', label_split=label_split, modalities=modalities, representation=representation, noise=noise, labelgen_model=labelgen_model,
        modality_prop=modality_prop, output_for='gan', embeds_type=args.embeds_type, data_dir=args.data_dir, label_type=args.label_type, task=args.task,
        text_cols=args.text_cols, num_clusters=args.num_clusters, model_dir=args.model_dir)
    X_test_unbiased, _, t_test_unbiased, _, y_test_unbiased, _, _ = load_data(
        split='test', data='unbiased', label_split=label_split, modalities=modalities, representation=representation, noise=noise, labelgen_model=labelgen_model,
        modality_prop=modality_prop, output_for='gan', embeds_type=args.embeds_type, data_dir=args.data_dir, label_type=args.label_type, task=args.task,
        text_cols=args.text_cols, num_clusters=args.num_clusters, model_dir=args.model_dir)

    if representation == 'raw' and args.model != 'AutoGluon':
        X_train, X_train_text, X_train_image = split_text_image(X_train, X_cols)
        X_train_heldout, X_train_heldout_text, X_train_heldout_image = split_text_image(X_train_heldout, X_cols)
        X_test_biased, X_test_biased_text, X_test_biased_image = split_text_image(X_test_biased, X_cols)
        X_test_unbiased, X_test_unbiased_text, X_test_unbiased_image = split_text_image(X_test_unbiased, X_cols)

if args.text_cols == 'all':
    clusters = np.load(os.path.join(args.data_dir, args.dataset, 'splitB_clusters{}_{}_alltext.npy'.format(
        args.num_clusters, args.label_type)))
else:
    clusters = np.load(os.path.join(args.data_dir, args.dataset, 'splitB_clusters{}_{}.npy'.format(
        args.num_clusters, args.label_type)))

regtype = ''
if args.irm:
    if args.irm_generator:
        regtype = '_irm_both'
    else:
        regtype = '_irm'
elif args.counterfactual_invariance:
    if args.counterfactual_invariance_generator:
        regtype = '_ci_both'
    else:
        regtype = '_ci'

if args.pre_scaling: 
    scaler = MinMaxScaler()
    scaler.fit(X_train.cpu())
    X_train = torch.tensor(scaler.transform(X_train.cpu())).to(device)
    X_train_heldout = torch.tensor(scaler.transform(X_train_heldout.cpu())).to(device)
    X_test_biased = torch.tensor(scaler.transform(X_test_biased.cpu())).to(device)
    X_test_unbiased = torch.tensor(scaler.transform(X_test_unbiased.cpu())).to(device)

if not no_counterfactual:
    if load:
        print('Loading labels')
        if noise == None:
            y_train_full = np.load(os.path.join(args.data_dir, 'airbnb/splitB_counterfactual_label_{}{}_{}{}_embeds.npy'.format(
                label_split, modality_stem, modalities, labelgen_model)))
        else:
            y_train_full = np.load(os.path.join(args.data_dir, 'airbnb/splitB_counterfactual_label_{}{}_{}_tabnoise_{}{}_embeds.npy'.format(
                label_split, modality_stem, modalities, noise, labelgen_model)))

    else:
        print('Generating labels\n')
        if not args.finetune:
            print('CounterfactualGAN with static embeddings')
            if args.self_train:
                model = SelfTrainingCounterfactualGAN(X=X_train, Treatments=t_train, Y=y_train, clusters=clusters, num_iterations=int(args.num_gen_iters),
                    num_discr_iterations=int(args.num_discr_iters), data_dir=args.data_dir, task=args.task,
                    IRM=args.irm, IRM_generator=args.irm_generator,
                    counterfactual_invariance=args.counterfactual_invariance,
                    counterfactual_invariance_generator=args.counterfactual_invariance_generator,
                    num_selftrain_iterations=args.num_selftrain_iters,
                    share_weights=args.share_weights,
                    minibatch_size=args.batch_size,
                    separate_discriminators=args.separate_discriminators,
                    discriminator_t=args.discriminator_t,
                    save=False)
            else:
                model = CounterfactualGAN(X=X_train, Treatments=t_train, Y=y_train, clusters=clusters, num_iterations=int(args.num_gen_iters),
                    num_discr_iterations=int(args.num_discr_iters), data_dir=args.data_dir, task=args.task,
                    IRM=args.irm, IRM_generator=args.irm_generator,
                    counterfactual_invariance=args.counterfactual_invariance,
                    counterfactual_invariance_generator=args.counterfactual_invariance_generator,
                    minibatch_size=args.batch_size,
                    separate_discriminators=args.separate_discriminators,
                    discriminator_t=args.discriminator_t,
                    save=False)

                
            print('Evaluating CounterfactualGAN')
            pred = model(X_train_heldout, None, None, 256).cpu().numpy()
            test_pred_biased = model(X_test_biased, None, None, 256).cpu().numpy()
            test_pred_unbiased = model(X_test_unbiased, None, None, 256).cpu().numpy()
        elif not args.lm_gen:
            print('CounterfactualGAN with static generator and tunable inference')
            save_dir='{}counterfactual_gan_staticgen_difflr_tabcluster5_iter{}discr{}_{}{}_all_tabnoise_{}{}{}/'.format(args.model_dir, args.num_gen_iters,
                args.num_discr_iters, label_split, modality_stem, noise, labelgen_model, regtype)
            model = CounterfactualGANMultimodal(X=X_train, Treatments=t_train, Y=y_train, clusters=clusters, X_text=X_train_text, X_image=X_train_image,
                language_model='distilbert-base-uncased',
                image_model='google/vit-base-patch16-224-in21k',
                data_dir=args.data_dir,
                task=args.task,
                num_iterations=int(args.num_gen_iters),
                num_discr_iterations=int(args.num_discr_iters), 
                IRM=args.irm, IRM_generator=args.irm_generator,
                counterfactual_invariance=args.counterfactual_invariance,
                counterfactual_invariance_generator=args.counterfactual_invariance_generator,
                lm_gen=args.lm_gen,
                minibatch_size=args.batch_size, # SET AT MAXIMUM 16
                separate_discriminators=args.separate_discriminators,
                discriminator_t=args.discriminator_t,
                save=True,
                save_dir=save_dir)
            
            print('Evaluating CounterfactualGAN')
            pred = model(X_train_heldout[:, 0:120], X_train_heldout_text, X_train_heldout_image, 32).cpu().numpy()
            test_pred_biased = model(X_test_biased[:, 0:120], X_test_biased_text, X_test_biased_image, 32).cpu().numpy()
            test_pred_unbiased = model(X_test_unbiased[:, 0:120], X_test_unbiased_text, X_test_unbiased_image, 32).cpu().numpy()
        else:
            print('CounterfactualGAN with tunable generator and inference')
            save_dir='{}counterfactual_gan_lmgen_difflr_tabcluster5_iter{}discr{}_{}{}_all_tabnoise_{}{}{}/'.format(args.model_dir, args.num_gen_iters,
                args.num_discr_iters, label_split, modality_stem, noise, labelgen_model, regtype)
            model = CounterfactualGANMultimodal(X=X_train, Treatments=t_train, Y=y_train, clusters=clusters, X_text=X_train_text, X_image=X_train_image,
                language_model='distilbert-base-uncased',
                image_model='google/vit-base-patch16-224-in21k',
                data_dir=args.data_dir,
                task=args.task,
                num_iterations=int(args.num_gen_iters),
                num_discr_iterations=int(args.num_discr_iters), 
                IRM=args.irm, IRM_generator=args.irm_generator,
                counterfactual_invariance=args.counterfactual_invariance,
                counterfactual_invariance_generator=args.counterfactual_invariance_generator,
                lm_gen=args.lm_gen,
                minibatch_size=args.batch_size, # SET AT MAXIMUM 16
                separate_discriminators=args.separate_discriminators,
                discriminator_t=args.discriminator_t,
                save=True,
                save_dir=save_dir)
        

            print('Evaluating CounterfactualGAN')
            pred = model(X_train_heldout, X_train_heldout_text, X_train_heldout_image, 32).cpu().numpy()
            test_pred_biased = model(X_test_biased, X_test_biased_text, X_test_biased_image, 32).cpu().numpy()
            test_pred_unbiased = model(X_test_unbiased, X_test_unbiased_text, X_test_unbiased_image, 32).cpu().numpy()

        if not os.path.exists('{}{}/{}'.format(args.results_dir, args.dataset, args.out_file)):
            with open(r'{}'.format('{}{}/{}'.format(args.results_dir, args.dataset, args.out_file)), 'a') as f:
                writer = csv.writer(f, delimiter = ",")
                headers = ['eval_set', 'label_type', 'text_cols', 'gen_iters', 'discr_iters', 'embeds', 'gen_reg', 'inference_reg', 
                           'self_train', 'share_weights', 'separate_discriminators', 'discriminator_t', 'prescaling', 'ipw', 'oracle', 'model']
                if args.task == 'classification':
                    headers += ['accuracy', 'f1_weighted', 'f1_macro', 'f1_negative']
                elif args.task == 'multiclass':
                    headers += ['accuracy', 'f1_weighted', 'f1_macro']
                elif args.task == 'regression':
                    headers += ['r2', 'nrmse', 'nrmse_macro', 'nrmse_negative']
                writer.writerow(headers)
                
        
        common_fields = [args.num_gen_iters, args.num_discr_iters, args.embeds_type]
        if args.irm_generator:
            common_fields += ['irm']
        elif args.counterfactual_invariance_generator:
            common_fields += ['counterfactual_invariance']
        else:
            common_fields += ['none']
        
        if args.irm:
            common_fields += ['irm']
        elif args.counterfactual_invariance:
            common_fields += ['counterfactual_invariance']
        else:
            common_fields += ['none']
        
        if args.self_train:
            common_fields += ['yes']
        else:
            common_fields += ['no']
        
        if args.share_weights:
            common_fields += ['yes']
        else:
            common_fields += ['no']
        
        if args.separate_discriminators:
            common_fields += ['yes']
        else:
            common_fields += ['no']
        
        if args.discriminator_t:
            common_fields += ['yes']
        else:
            common_fields += ['no']

        if args.pre_scaling:
            common_fields += ['yes']
        else:
            common_fields += ['no']

        if args.simple_ipw:
            common_fields += ['yes']
        else:
            common_fields += ['no']

        if args.oracle:
            common_fields += ['yes']
        else:
            common_fields += ['no']
            
        if args.task == 'classification':
            counterfactual_labels = (pred > 0.5).astype(int)
            test_pred_biased = (test_pred_biased > 0.5).astype(int)
            test_pred_unbiased = (test_pred_unbiased > 0.5).astype(int)
        elif args.task == 'regression':
            counterfactual_labels = pred.astype(int)
            test_pred_biased = test_pred_biased.astype(int)
            test_pred_unbiased = test_pred_unbiased.astype(int)
        elif args.task == 'multiclass':
            counterfactual_0 = pred[:, 0:5].argmax(axis=1)
            counterfactual_1 = pred[:, 5:].argmax(axis=1)
            counterfactual_labels = np.hstack([counterfactual_0.reshape(-1, 1), counterfactual_1.reshape(-1, 1)])

            test_pred_biased_0 = test_pred_biased[:, 0:5].argmax(axis=1)
            test_pred_biased_1 = test_pred_biased[:, 5:].argmax(axis=1)
            test_pred_biased = np.hstack([test_pred_biased_0.reshape(-1, 1), test_pred_biased_1.reshape(-1, 1)])

            test_pred_unbiased_0 = test_pred_unbiased[:, 0:5].argmax(axis=1)
            test_pred_unbiased_1 = test_pred_unbiased[:, 5:].argmax(axis=1)
            test_pred_unbiased = np.hstack([test_pred_unbiased_0.reshape(-1, 1), test_pred_unbiased_1.reshape(-1, 1)])

        pred_labels = counterfactual_labels[np.arange(len(counterfactual_labels)), t_train_heldout]
        test_pred_biased = test_pred_biased[np.arange(len(test_pred_biased)), t_test_biased]
        
        print('Biased set')
        
        if args.task == 'regression':

            mean_nrmse = np.sqrt(mean_squared_error(y_test_biased, [np.mean(y_train)]*len(y_test_biased), squared=True))/np.std(y_test_biased)
            print('Mean prediction NRMSE: {}'.format(mean_nrmse))

        metrics = get_metrics(y_true=y_test_biased, y_pred=test_pred_biased, output=True, task=args.task, norm_constant=np.std(y_test_biased))
        fields = ['biased', args.label_type, args.text_cols] + common_fields + ['none'] + list(metrics)
        
        with open(r'{}'.format('{}{}/{}'.format(args.results_dir, args.dataset, args.out_file)), 'a') as f:
            writer = csv.writer(f, delimiter = ",")
            writer.writerow(fields)

        test_pred_unbiased = test_pred_unbiased[np.arange(len(test_pred_unbiased)), t_test_unbiased]

        print('\nUnbiased set')
        if args.task == 'regression':
        
            mean_nrmse = np.sqrt(mean_squared_error(y_test_unbiased, [np.mean(y_train)]*len(y_test_unbiased), squared=True))/np.std(y_test_unbiased)
            print('Mean prediction NRMSE: {}'.format(mean_nrmse))
    
        metrics = get_metrics(y_true=y_test_unbiased, y_pred=test_pred_unbiased, output=True, task=args.task, norm_constant=np.std(y_test_unbiased))
        fields = ['unbiased', args.label_type, args.text_cols] + common_fields + ['none'] + list(metrics)
        
        with open(r'{}'.format('{}{}/{}'.format(args.results_dir, args.dataset, args.out_file)), 'a') as f:
            writer = csv.writer(f, delimiter = ",")
            writer.writerow(fields)

        end_time = time.time()

        print("Runtime in seconds: {}".format(end_time - start_time))

        if args.strong_bias:
            y_train_full = np.concatenate([y_train, y_train_heldout[t_train_heldout==1], pred_labels[t_train_heldout==0]])
        else:
            y_train_full = np.concatenate([y_train, pred_labels])
        
        if args.plot:
            import seaborn as sns
            import matplotlib.pyplot as plt
            sns.set_style('white')
            plt.rcParams.update({'font.size': 16})
            # plt.rc('legend', fontsize=14)

            if args.task == 'regression':
                if args.dataset == 'airbnb':
                    sns.kdeplot(y_train, fill=True, label='uncorrected')
                    sns.kdeplot(y_train_full, fill=True, label='bias-corrected with CA')
                    if args.plot_oracle:
                        y_train_oracle = np.hstack([y_train, y_train_heldout])
                        sns.kdeplot(y_train_oracle, fill=True, label='unbiased')
                    plt.xlim(0, 2000)
                    
                    if not args.plot_oracle:
                        plt.legend(labels=['uncorrected', 'bias-corrected with CA'], loc='upper right')
                    else:
                        plt.legend(labels=['uncorrected', 'bias-corrected with CA', 'unbiased'], loc='upper right')
                elif args.dataset == 'clothing_review':
                    plt1 = sns.histplot(y_train, stat='density', bins=[1, 1.8, 2.6, 3.4, 4.2, 5.0], 
                        alpha=0.4, label='uncorrected')
                    plt.xlim(np.min(y_train), np.max(y_train))
                    plt2 = sns.kdeplot(y_train_full, fill=True, label='bias-corrected with CA')
                    h, l = plt1.get_legend_handles_labels()
                    plt.legend([h[0], h[2]], [l[0], l[2]], loc='upper left')

            elif args.task == 'multiclass':
                plt1 = sns.histplot(y_train+1, stat='probability', bins=[1, 1.8, 2.6, 3.4, 4.2, 5.0], alpha=0.4, label='uncorrected')
                plt.xlim(np.min(y_train_full+1), np.max(y_train_full+1))
                plt2 = sns.histplot(y_train_full+1, stat='probability', bins=[1, 1.8, 2.6, 3.4, 4.2, 5.0], alpha=0.5, label='bias-corrected with CA')
                h, l = plt1.get_legend_handles_labels()
                plt.legend([h[0], h[2]], [l[0], l[2]], loc='upper left')

            else:
                plt1 = sns.histplot(y_train, stat='probability', bins=2, alpha=0.4, label='uncorrected')
                plt2 = sns.histplot(y_train_full, stat='probability', bins=2, alpha=0.5, label='bias-corrected with CA')
                if args.plot_oracle:
                    y_train_oracle = np.hstack([y_train, y_train_heldout])
                    plt3 = sns.histplot(y_train_oracle, stat='probability', bins=2, alpha=0.4, label='unbiased')
                h, l = plt1.get_legend_handles_labels()
                if not args.plot_oracle:
                    plt.legend([h[0], h[2]], [l[0], l[2]], loc='upper left')
                else:
                    plt.legend([h[0], h[2], h[4]], [l[0], l[2], l[4]], loc='upper left')

            if args.label_type == 'real':
                plt.title('{} dataset, {} task'.format(args.dataset, args.task))
            else:
                plt.title('synthetic dataset, {} task'.format(args.task))
            if not args.plot_oracle:
                plt.savefig('label_balance_{}_{}_{}.png'.format(args.dataset, args.task, args.label_type),
                    bbox_inches='tight')
            else:
                plt.savefig('label_balance_{}_{}_{}_with_oracle.png'.format(args.dataset, args.task, args.label_type),
                    bbox_inches='tight')

            plt.clf()

            if args.task == 'regression':
                if args.dataset == 'airbnb':
                    sns.kdeplot(y_train_heldout, fill=True, label='true counterfactuals')
                    sns.kdeplot(pred_labels, fill=True, label='generated counterfactuals')
                    plt.xlim(0, 2000)
                    plt.legend(labels=['true counterfactuals', 'generated counterfactuals'], loc='upper right')
                elif args.dataset == 'clothing_review':
                    plt1 = sns.histplot(y_train_heldout, stat='density', bins=[1, 1.8, 2.6, 3.4, 4.2, 5.0], 
                        alpha=0.4, label='true counterfactuals')
                    plt.xlim(np.min(y_train_heldout), np.max(y_train_heldout))
                    plt2 = sns.kdeplot(pred_labels, fill=True, label='generated counterfactuals')
                    h, l = plt1.get_legend_handles_labels()
                    plt.legend([h[0], h[2]], [l[0], l[2]], loc='upper left')

            elif args.task == 'multiclass':
                plt1 = sns.histplot(y_train_heldout+1, stat='probability', bins=[1, 1.8, 2.6, 3.4, 4.2, 5.0], alpha=0.4, label='true counterfactuals')
                plt.xlim(np.min(y_train_heldout+1), np.max(y_train_heldout+1))
                plt2 = sns.histplot(pred_labels+1, stat='probability', bins=[1, 1.8, 2.6, 3.4, 4.2, 5.0], alpha=0.5, label='generated counterfactuals')
                h, l = plt1.get_legend_handles_labels()
                plt.legend([h[0], h[2]], [l[0], l[2]], loc='upper left')

            else:
                plt1 = sns.histplot(y_train_heldout, stat='probability', bins=2, alpha=0.4, label='true counterfactuals')
                plt2 = sns.histplot(pred_labels, stat='probability', bins=2, alpha=0.5, label='generated counterfactuals')
                h, l = plt1.get_legend_handles_labels()
                plt.legend([h[0], h[2]], [l[0], l[2]], loc='upper left')

            if args.label_type == 'real':
                plt.title('{} dataset, {} task'.format(args.dataset, args.task))
            else:
                plt.title('synthetic dataset, {} task'.format(args.task))
            plt.savefig('label_dist_{}_{}_{}.png'.format(args.dataset, args.task, args.label_type),
                bbox_inches='tight')
            os._exit()

        if save:
            if not args.finetune:
                np.save(os.path.join(args.data_dir, 'airbnb/splitB_counterfactual_label_{}{}_{}_tabnoise_{}{}_embeds.npy'.format(
                    label_split, modality_stem, modalities, noise, labelgen_model), y_train_full))
            
            else:
                save_path = os.path.join(save_dir, 
                    'splitB_counterfactual_label_{}{}_{}_tabnoise_{}{}_embeds.npy'.format(
                    label_split, modality_stem, modalities, noise, labelgen_model))
                np.save(save_path, y_train_full)

if args.model == 'SVM' or args.model == 'GBC' or args.model == 'LR':
    if representation != 'embeds':
        if modalities == 'all':
            X_train, X_train_text, X_train_image, X_train_heldout, X_train_heldout_text, X_train_heldout_image, t_train, t_train_heldout, y_train, y_train_heldout, X_cols = load_data(
                split='train', data='biased', label_split=label_split, modalities=modalities, representation='embeds', noise=noise, labelgen_model=labelgen_model,
                modality_prop=modality_prop, output_for='gan', embeds_type=args.embeds_type, data_dir=args.data_dir, label_type=args.label_type, task=args.task,
                text_cols=args.text_cols)
            X_test_biased, X_test_biased_text, X_test_biased_image, _, _, _, t_test_biased, _, y_test_biased, _, X_cols = load_data(
                split='test', data='biased', label_split=label_split, modalities=modalities, representation='embeds', noise=noise, labelgen_model=labelgen_model,
                modality_prop=modality_prop, output_for='gan', embeds_type=args.embeds_type, data_dir=args.data_dir, label_type=args.label_type, task=args.task,
                text_cols=args.text_cols)
            X_test_unbiased, X_test_unbiased_text, X_test_unbiased_image, _, _, _, t_test_unbiased, _, y_test_unbiased, _, _ = load_data(
                split='test', data='unbiased', label_split=label_split, modalities=modalities, representation='embeds', noise=noise, labelgen_model=labelgen_model,
                modality_prop=modality_prop, output_for='gan', embeds_type=args.embeds_type, data_dir=args.data_dir, label_type=args.label_type, task=args.task,
                text_cols=args.text_cols)
        elif modalities == 'tab':
            X_train, X_train_heldout, t_train, t_train_heldout, y_train, y_train_heldout, X_cols = load_data(
                split='train', data='biased', label_split=label_split, modalities=modalities, representation='embeds', noise=noise, labelgen_model=labelgen_model,
                modality_prop=modality_prop, output_for='gan', embeds_type=args.embeds_type, data_dir=args.data_dir, label_type=args.label_type, task=args.task,
                text_cols=args.text_cols)
            X_test_biased, _, t_test_biased, _, y_test_biased, _, X_cols = load_data(
                split='test', data='biased', label_split=label_split, modalities=modalities, representation='embeds', noise=noise, labelgen_model=labelgen_model,
                modality_prop=modality_prop, output_for='gan', embeds_type=args.embeds_type, data_dir=args.data_dir, label_type=args.label_type, task=args.task,
                text_cols=args.text_cols)
            X_test_unbiased, _, t_test_unbiased, _, y_test_unbiased, _, _ = load_data(
                split='test', data='unbiased', label_split=label_split, modalities=modalities, representation='embeds', noise=noise, labelgen_model=labelgen_model,
                modality_prop=modality_prop, output_for='gan', embeds_type=args.embeds_type, data_dir=args.data_dir, label_type=args.label_type, task=args.task,
                text_cols=args.text_cols)

    scaler = MinMaxScaler()
    if no_counterfactual:
        if args.oracle:
            X_train_full = np.vstack([X_train.cpu(), X_train_heldout.cpu()])
            y_train_full = np.hstack([y_train, y_train_heldout])
            t_train_full = np.hstack([t_train, t_train_heldout])
        elif args.strong_bias:
            X_train_full = np.vstack([X_train.cpu(), X_train_heldout[t_train_heldout==1,].cpu()])
            y_train_full = np.hstack([y_train, y_train_heldout[t_train_heldout==1]])
            t_train_full = np.hstack([t_train, t_train_heldout[t_train_heldout==1]])
        else:
            X_train_full = X_train.cpu()
            y_train_full = y_train
            t_train_full = t_train
        if not os.path.exists('{}{}/{}'.format(args.results_dir, args.dataset, args.out_file)):
            with open(r'{}'.format('{}{}/{}'.format(args.results_dir, args.dataset, args.out_file)), 'a') as f:
                writer = csv.writer(f, delimiter = ",")
                headers = ['eval_set', 'label_type', 'text_cols', 'gen_iters', 'discr_iters', 'embeds', 'gen_reg', 'inference_reg', 
                           'self_train', 'share_weights', 'separate_discriminators', 'discriminator_t', 'prescaling', 'ipw', 'oracle', 'model']
                if args.task == 'classification':
                    headers += ['accuracy', 'f1_weighted', 'f1_macro', 'f1_negative']
                elif args.task == 'multiclass':
                    headers += ['accuracy', 'f1_weighted', 'f1_macro']
                elif args.task == 'regression':
                    headers += ['r2', 'nrmse', 'nrmse_macro', 'nrmse_negative']
                writer.writerow(headers)
        common_fields = ['NA', 'NA', args.embeds_type, 'NA', 'NA', 'NA', 'NA', 'NA', 'NA']
        if args.pre_scaling:
            common_fields += ['yes']
        else:
            common_fields += ['no']
        if args.simple_ipw:
            common_fields += ['yes']
        else:
            common_fields += ['no']
        if args.oracle:
            common_fields += ['yes']
        else:
            common_fields += ['no']
    else:

        if args.strong_bias:
            X_train_full = np.vstack([X_train.cpu(), X_train_heldout[t_train_heldout==1,].cpu(), X_train_heldout[t_train_heldout==0,].cpu()])
            t_train_full = np.hstack([t_train, t_train_heldout[t_train_heldout==1], t_train_heldout[t_train_heldout==0]])
        else:
            X_train_full = np.vstack([X_train.cpu(), X_train_heldout.cpu()])
            t_train_full = np.hstack([t_train, t_train_heldout])

    X_test_biased = X_test_biased.cpu()
    X_test_unbiased = X_test_unbiased.cpu()
    print(X_train_full.shape, y_train_full.shape)
    scaler.fit(X_train_full)

    if args.no_counterfactual:
        print('\nEvaluating {}'.format(args.model))
    else:
        print('\nEvaluating {} + CounterfactualGAN'.format(args.model))
    print('Evaluation on biased set')
    if args.model == 'SVM':
        if args.task == 'classification' or args.task == 'multiclass':
            clf = SVC()
        elif args.task == 'regression':
            clf = SVR()
    elif args.model == 'GBC':
        if args.task == 'classification' or args.task == 'multiclass':
            clf = GBC()
        elif args.task == 'regression':
            clf = GBR()
    elif args.model == 'LR':
        if args.task == 'classification' or args.task == 'multiclass':
            clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)
        elif args.task == 'regression':
            clf = ElasticNet()

    if args.simple_ipw:
        print('Fitting propensity model')
        if args.model == 'SVM':
            t_clf = SVC(probability=True)
        elif args.model == 'GBC':
            t_clf = GBC(probability=True)
        elif args.model == 'LR':
            t_clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)
        t_clf.fit(scaler.transform(X_train_full), t_train_full)
        t_train_probs = t_clf.predict_proba(scaler.transform(X_train_full))[:,1]

        print('Evaluating propensity model')
        print('Mean propensity score: {}'.format(np.mean(t_train_probs)))
        print('Actual mean treatment: {}'.format(np.mean(t_train_full)))
        print('Mean A=0 propensity score: {}'.format(np.mean(t_train_probs[t_train_full == 0])))

        print('Fitting outcome model')
        clf.fit(scaler.transform(X_train_full), y_train_full, sample_weight=1.0/(t_train_probs+1e-5))
    else:
        print('Fitting outcome model')
        clf.fit(scaler.transform(X_train_full), y_train_full)

    preds = clf.predict(scaler.transform(X_test_biased))
    metrics = get_metrics(y_true=y_test_biased, y_pred=preds, output=True, task=args.task, norm_constant=np.std(y_test_biased))
    fields = ['biased', args.label_type, args.text_cols] + common_fields + [args.model] + list(metrics)
    with open(r'{}'.format('{}{}/{}'.format(args.results_dir, args.dataset, args.out_file)), 'a') as f:
        writer = csv.writer(f, delimiter = ",")
        writer.writerow(fields)
    
    print('\nEvaluation on unbiased set')
    preds = clf.predict(scaler.transform(X_test_unbiased))
    metrics = get_metrics(y_true=y_test_unbiased, y_pred=preds, output=True, task=args.task, norm_constant=np.std(y_test_unbiased))
    fields = ['unbiased', args.label_type, args.text_cols] + common_fields + [args.model] + list(metrics)
    with open(r'{}'.format('{}{}/{}'.format(args.results_dir, args.dataset, args.out_file)), 'a') as f:
        writer = csv.writer(f, delimiter = ",")
        writer.writerow(fields)    
    print()

end_time = time.time()

print("Runtime in seconds: {}".format(end_time - start_time))