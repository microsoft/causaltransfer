import pdb
import os
import torch
import pandas as pd
import numpy as np
import shutil
from urllib.parse import urlparse
from torch.autograd import grad
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_irm_penalty(loss, dummy):
    try:
        g = grad(loss, dummy, create_graph=True)[0]
        return torch.sum(g**2)
    except:
        pdb.set_trace()

def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

def bmse(pred, target, norm_constant):
    pred = torch.tensor(pred)
    target = torch.tensor(target)
    logits = - 0.5 * (pred - target.T).pow(2) / norm_constant
    pdb.set_trace()
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0], dtype=torch.float32))
    loss = loss * (2 * norm_constant)
    return loss

def get_metrics(y_true, y_pred, output=True, pos_label=0, task='classification', norm_constant=1):
    if task == 'classification' or task == 'multiclass':
        accuracy = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        if task == 'classification':
            f1_negative = f1_score(y_true, y_pred, average='binary', pos_label=pos_label)
        else:
            f1_negative = None
        if output:
            print('Accuracy: {}'.format(accuracy))
            print('Weighted F1 score: {}'.format(f1_weighted))
            print('Macro F1 score: {}'.format(f1_macro))
            if task == 'classification':
                print('Negative F1 score: {}'.format(f1_negative))

        return (accuracy, f1_weighted, f1_macro, f1_negative)
    elif task == 'regression':
        r2 = r2_score(y_true, y_pred)
        nrmse = np.sqrt(mean_squared_error(y_true, y_pred, squared=True))/norm_constant
        pos_nrmse = np.sqrt(mean_squared_error(y_true[y_true > np.quantile(y_true, 0.2)], y_pred[y_true > np.quantile(y_true, 0.2)], squared=True))/norm_constant
        neg_nrmse = np.sqrt(mean_squared_error(y_true[y_true < np.quantile(y_true, 0.2)], y_pred[y_true < np.quantile(y_true, 0.2)], squared=True))/norm_constant
        macro_nrmse = np.mean([pos_nrmse, neg_nrmse])
        if output:
            print('R2: {}'.format(r2))
            print('NRMSE: {}'.format(nrmse))
            print('Macro NRMSE: {}'.format(macro_nrmse))
            print('Negative NRMSE: {}'.format(neg_nrmse))

        return (r2, nrmse, macro_nrmse, neg_nrmse)

def download_image(image_url, destination_path):
    # Source: https://towardsdatascience.com/how-to-download-an-image-using-python-38a75cfa21c
    # pdb.set_trace()
    try:
        parsed_url = urlparse(image_url)
        filename = parsed_url.path.split("/")[-1]

        # Open the url image, set stream to True, this will return the stream content.
        headers = {"User-Agent": "aUserAgent"}
        r = requests.get(image_url, headers=headers, stream=True)

        # Check if the image was retrieved successfully
        if r.status_code == 200:

            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            r.raw.decode_content = True
            with open('{}{}'.format(destination_path, filename), 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        else:
            print('Image {} couldn\'t be retrieved'.format(image_url))
            return('NaN')
    except:
        print("Error with {}".format(image_url))
        return('NaN')
        

    return('{}{}'.format(destination_path, filename))

def load_clothing_data(data_dir, model_dir, split='train', data='biased', label_split='08', modalities='tab', representation='embeds', noise='weak', 
    labelgen_model='', modality_prop='060202', output_for='gan', embeds_type=None, label_type='synthetic', task='classification', text_cols='text',
    num_clusters=5):

    if not output_for == 'dragonnet':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if split == 'train':
        split_name = 'B'
    elif split == 'test':
        split_name = 'C'
    X = pd.read_csv(os.path.join(data_dir, 'clothing_review/split{}.csv'.format(split_name))).reset_index(drop=True)
    if label_type != 'synthetic':
        if task == 'regression':
            y_biased = np.load(os.path.join(data_dir, 'clothing_review/split{}_labels_pred_rating_reg_new.npy'.format(split_name)))
            y_unbiased = np.load(os.path.join(data_dir, 'clothing_review/split{}_labels_true_rating_reg.npy'.format(split_name)))
        elif task == 'multiclass':
            y_biased = np.load(os.path.join(data_dir, 'clothing_review/split{}_labels_pred_rating_multiclass_new.npy'.format(split_name)))
            y_unbiased = np.load(os.path.join(data_dir, 'clothing_review/split{}_labels_true_rating_multiclass.npy'.format(split_name)))
        elif task == 'classification':
            y_biased = np.load(os.path.join(data_dir, 'clothing_review/split{}_labels_pred_rating_classification_new.npy'.format(split_name)))
            y_unbiased = np.load(os.path.join(data_dir, 'clothing_review/split{}_labels_true_rating_classification.npy'.format(split_name)))
    
    if embeds_type == 'pretrained' or embeds_type is None:
        if text_cols == 'all':
            embeds_title_text = np.load(os.path.join(data_dir, 'clothing_review/split{}_embeds_title_text_distilbert-base-uncased.npy'.format(split_name)))
            embeds_review_text = np.load(os.path.join(data_dir, 'clothing_review/split{}_embeds_review_text_distilbert-base-uncased.npy'.format(split_name)))
            embeds_text = np.hstack([embeds_title_text, embeds_review_text])
        elif text_cols == 'text':
            embeds_text = np.load(os.path.join(data_dir, 'clothing_review/split{}_embeds_text_distilbert-base-uncased.npy'.format(split_name)))

    elif embeds_type == 'finetuned':
        if text_cols == 'all':
            embeds_title_text = np.load(os.path.join(model_dir, 'clothing_review/distilbert-base-uncased_finetune_rating_title_{}/epochs_5/best_model/split{}_embeds_text.npy'.format(
                task, split_name)))
            embeds_review_text = np.load(os.path.join(model_dir, 'clothing_review/distilbert-base-uncased_finetune_rating_review_text_{}/epochs_5/best_model/split{}_embeds_text.npy'.format(
                task, split_name)))
            embeds_text = np.hstack([embeds_title_text, embeds_review_text])
        elif text_cols == 'text':
            embeds_text = np.load(os.path.join(model_dir, 'clothing_review/distilbert-base-uncased_finetune_rating_text_{}/epochs_5/best_model/split{}_embeds_text.npy'.format(
                task, split_name)))
    elif embeds_type == 'finetuned_irm':
        if text_cols == 'all':
            embeds_title_text = np.load(os.path.join(model_dir, 'clothing_review/distilbert-base-uncased_finetune_rating_title_{}_irm_cluster{}/epochs_5/best_model/split{}_embeds_text.npy'.format(
                task, num_clusters, split_name
            )))
            embeds_review_text = np.load(os.path.join(model_dir, 'clothing_review/distilbert-base-uncased_finetune_rating_review_text_{}_irm_cluster{}/epochs_5/best_model/split{}_embeds_text.npy'.format(
                task, num_clusters, split_name
            )))
            embeds_text = np.hstack([embeds_title_text, embeds_review_text])
        elif text_cols == 'text':
            embeds_text = np.load(os.path.join(model_dir, 'clothing_review/distilbert-base-uncased_finetune_rating_text_{}_irm_cluster{}/epochs_5/best_model/split{}_embeds_text.npy'.format(
                task, num_clusters, split_name
            )))
       
    elif embeds_type == 'finetuned_ci':
        if text_cols == 'all':
            embeds_title_text = np.load(os.path.join(model_dir, 'clothing_review/distilbert-base-uncased_finetune_rating_title_{}_ci_cluster{}/epochs_5/best_model/split{}_embeds_text.npy'.format(
                task, num_clusters, split_name
            )))
            embeds_review_text = np.load(os.path.join(model_dir, 'clothing_review/distilbert-base-uncased_finetune_rating_review_text_{}_ci_cluster{}/epochs_5/best_model/split{}_embeds_text.npy'.format(
                task, num_clusters, split_name
            )))
            embeds_text = np.hstack([embeds_title_text, embeds_review_text])
        elif text_cols == 'text':
            embeds_text = np.load(os.path.join(model_dir, 'clothing_review/distilbert-base-uncased_finetune_rating_text_{}_ci_cluster{}/epochs_5/best_model/split{}_embeds_text.npy'.format(
                task, num_clusters, split_name
            )))


    X['biased_label'] = y_biased
    X['true_label'] = y_unbiased

    X_full = X.copy()

    drop_indices = []

    if data == 'biased':
        np.random.seed(220628)
        if task == 'regression':
            drop_indices = np.random.choice(
                X[X['biased_label'] <=3 ].index, int(0.9*X[X['biased_label'] <= 3].shape[0]), 
                replace=False).astype(int)
        elif task == 'multiclass':
            drop_indices = np.random.choice(
                X[X['biased_label'] <=2 ].index, int(0.9*X[X['biased_label'] <= 2].shape[0]), 
                replace=False).astype(int)
        elif task == 'classification':
            drop_indices = np.random.choice(X[X['biased_label'] == 0].index, int(0.6*X[X['biased_label'] == 0].shape[0]), 
                replace=False).astype(int)
        # pdb.set_trace()
        X = X.drop(drop_indices)
    
    if X.shape[0] > 5400 and not output_for == 'dragonnet':
        new_drop_indices = np.random.choice(X.index, X.shape[0] - 5400, replace=False)
        drop_indices = np.concatenate([drop_indices, new_drop_indices]).astype(int)
        X = X.drop(new_drop_indices)

    X_heldout = X_full.iloc[drop_indices]
    embeds_text_heldout = embeds_text[drop_indices]

    embeds_text = np.delete(embeds_text, drop_indices, axis=0)

    drop_cols = ['Clothing ID', 'recommended_ind', 'rating', 'biased_label', 'true_label']

    if modalities == 'tab' or output_for == 'dragonnet':
        drop_cols += ['title', 'review_text', 'text']

    if output_for == 'dragonnet':
        y = X['true_label'].values.astype('float32')
        if task == 'regression':
            t = (X['biased_label'].values > 3).astype('float32')
        elif task == 'multiclass':
            t = (X['biased_label'].values > 2).astype('float32')
        elif task == 'classification':
            t = X['biased_label'].values.astype('float32')
        x = X.drop(drop_cols, axis=1).to_numpy().astype('float32')

        if modalities == 'all':
            x = np.hstack([x, embeds_text])

        data = {'x':x,'t':t,'y':y,'t':t}
        data['t'] = data['t'].reshape(-1,1) #we're just padding one dimensional vectors with an additional dimension 
        data['y'] = data['y'].reshape(-1,1)

        return data

    y = X['true_label'].values.astype(int)
    if task == 'regression':
        t = (X['biased_label'].values > 3).astype(int)
    elif task == 'multiclass':
        t = (X['biased_label'].values > 2).astype(int)
    elif task == 'classification':
        t = X['biased_label'].values.astype(int)
    X = X.drop(drop_cols, axis=1)
    if representation == 'embeds' and modalities == 'all':
        X_text = X['text'].values
        X = X.drop(['title', 'review_text', 'text'], axis=1)
    colnames = X.columns
    if representation == 'embeds' and not output_for == 'dragonnet':
        x = torch.tensor(X.to_numpy(dtype=float)).to(device)
    else:
        x = X.to_numpy()
    
    if task == 'regression':
        t_heldout = (X_heldout['biased_label'].values > 3).astype(int)
    elif task == 'multiclass':
        t_heldout = (X_heldout['biased_label'].values > 2).astype(int)
    elif task == 'classification':
        t_heldout = X_heldout['biased_label'].values.astype(int)
    y_heldout = X_heldout['true_label'].values.astype(int)
    X_heldout = X_heldout.drop(drop_cols, axis=1)
    if representation == 'embeds' and modalities == 'all':
        X_heldout_text = X_heldout['text'].values
        X_heldout = X_heldout.drop(['title', 'review_text', 'text'], axis=1)

    if representation == 'embeds' and not output_for == 'dragonnet':
        x_heldout = torch.tensor(X_heldout.to_numpy(dtype=float)).to(device)
    else:
        x_heldout = X_heldout.to_numpy()

    if modalities == 'all':
        if representation == 'embeds':
            if output_for == 'dragonnet':
                x = np.hstack([x, embeds_text])
                x_heldout = np.hstack([x_heldout, embeds_text_heldout])
            else:
                x = torch.cat([x, torch.tensor(embeds_text).to(device)], axis=1)
                x_heldout = torch.cat([x_heldout, torch.tensor(embeds_text_heldout).to(device)], axis=1)
            return (x, X_text, None, x_heldout, X_heldout_text, None, t, t_heldout, y, y_heldout, colnames)

    return (x, x_heldout, t, t_heldout, y, y_heldout, colnames)

def load_airbnb_data(data_dir, model_dir, split='train', data='biased', label_split='08', modalities='tab', representation='embeds', noise='weak', 
    labelgen_model='', modality_prop='060202', output_for='gan', embeds_type=None, label_type='synthetic', task='classification', text_cols=None,
    num_clusters=5):

    if not output_for == 'dragonnet':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if split == 'train':
        split_name = 'B'
    elif split == 'test':
        split_name = 'C'
    X = pd.read_csv(os.path.join(data_dir, 'airbnb/split{}_w_img_prob.csv'.format(split_name))).reset_index(drop=True)
    if label_type != 'synthetic':
        if task == 'classification':
            y_biased = np.load(os.path.join(data_dir, 'airbnb/split{}_labels_pred_price_pos{}.npy'.format(split_name, label_split)))
            y_unbiased = np.load(os.path.join(data_dir, 'airbnb/split{}_labels_true_price_pos{}.npy'.format(split_name, label_split)))
        elif task == 'regression':
            y_biased = np.load(os.path.join(data_dir, 'airbnb/split{}_labels_pred_price_reg.npy'.format(split_name)))
            y_unbiased = np.load(os.path.join(data_dir, 'airbnb/split{}_labels_true_price_reg.npy'.format(split_name)))
    else:
        if noise == None:
            y_biased = np.load(os.path.join(data_dir, 'airbnb/split{}_pred_labels_imgpredprob_pos{}_{}{}.npy'.format(split_name, label_split, modality_prop, labelgen_model)))
            y_unbiased = np.load(os.path.join(data_dir, 'airbnb/split{}_labels_true_pricepred_img_pos{}_{}.npy'.format(split_name, label_split, modality_prop)))
        else:
            y_biased = np.load(os.path.join(data_dir, 'airbnb/split{}_pred_labels_imgpredprob_pos{}_{}_tabnoise_{}{}.npy'.format(split_name, label_split,
                modality_prop, noise, labelgen_model)))
            y_unbiased = np.load(os.path.join(data_dir, 'airbnb/split{}_labels_true_pricepred_img_pos{}_{}_tabnoise_{}.npy'.format(split_name, label_split, 
                modality_prop, noise)))
    
    if embeds_type == 'pretrained' or embeds_type is None:
        embeds_list_text = np.load(os.path.join(data_dir, 'airbnb/split{}_embeds_listing_text_distilbert-base-uncased.npy'.format(split_name)))
        embeds_image = np.load(os.path.join(data_dir, 'airbnb/split{}_embeds_image_vit-base-patch16-224-in21k.npy'.format(split_name)))
    
    if label_type == 'real':
        if embeds_type == 'finetuned':
            embeds_list_text = np.load(os.path.join(model_dir, 'airbnb/distilbert-base-uncased_finetune_price_{}/epochs_5/best_model/split{}_embeds_listing_text.npy'.format(
                task, split_name)))
            embeds_image = np.load(os.path.join(model_dir, 'airbnb/google/vit-base-patch16-224-in21k_finetune_price_{}/epochs_5/best_model/split{}_embeds_image.npy'.format(
                task, split_name)))
        elif embeds_type == 'finetuned_irm':
            embeds_list_text = np.load(os.path.join(model_dir, 'airbnb/distilbert-base-uncased_finetune_price_{}_irm_cluster{}/epochs_5/best_model/split{}_embeds_listing_text.npy'.format(
                task, num_clusters, split_name)))
            embeds_image = np.load(os.path.join(model_dir, 'airbnb/google/vit-base-patch16-224-in21k_finetune_price_{}_irm_cluster{}/epochs_5/best_model/split{}_embeds_image.npy'.format(
                task, num_clusters, split_name)))
        elif embeds_type == 'finetuned_ci':
            embeds_list_text = np.load(os.path.join(model_dir, 'airbnb/distilbert-base-uncased_finetune_price_{}_ci_cluster{}/epochs_5/best_model/split{}_embeds_listing_text.npy'.format(
                task, num_clusters, split_name)))
            embeds_image = np.load(os.path.join(model_dir, 'airbnb/google/vit-base-patch16-224-in21k_finetune_price_{}_ci_cluster{}/epochs_5/best_model/split{}_embeds_image.npy'.format(
                task, num_clusters, split_name)))
    elif label_type == 'synthetic':
        if embeds_type == 'finetuned':
            embeds_list_text = np.load(os.path.join(model_dir, 'airbnb/distilbert-base-uncased_finetune_08_060202_tabnoise_strong_lr/epochs_5/best_model/split{}_embeds_listing_text.npy'.format(
                split_name)))
            embeds_image = np.load(os.path.join(model_dir, 'airbnb/google/vit-base-patch16-224-in21k_finetune_08_060202_tabnoise_strong_lr/epochs_5/best_model/split{}_embeds_image.npy'.format(
                split_name)))
        elif embeds_type == 'finetuned_irm':
            embeds_list_text = np.load(os.path.join(model_dir, 'airbnb/distilbert-base-uncased_finetune_08_060202_tabnoise_strong_lr_irm_cluster{}/epochs_5/best_model/split{}_embeds_listing_text.npy'.format(
                num_clusters, split_name)))
            embeds_image = np.load(os.path.join(model_dir, 'airbnb/google/vit-base-patch16-224-in21k_finetune_08_060202_tabnoise_strong_lr_irm_cluster{}/epochs_5/best_model/split{}_embeds_image.npy'.format(
                num_clusters, split_name)))
        elif embeds_type == 'finetuned_ci':
            embeds_list_text = np.load(os.path.join(model_dir, 'airbnb/distilbert-base-uncased_finetune_08_060202_tabnoise_strong_lr_ci_cluster{}/epochs_5/best_model/split{}_embeds_listing_text.npy'.format(
                num_clusters, split_name)))
            embeds_image = np.load(os.path.join(model_dir, 'airbnb/google/vit-base-patch16-224-in21k_finetune_08_060202_tabnoise_strong_lr_ci_cluster{}/epochs_5/best_model/split{}_embeds_image.npy'.format(
                num_clusters, split_name)))

    X['biased_label'] = y_biased
    X['true_label'] = y_unbiased

    X_full = X.copy()

    drop_indices = []
    if task == 'regression':
        cutoff = np.quantile(X['biased_label'].values, 0.2)

    if data == 'biased':
        np.random.seed(220628)
        if task == 'classification':
            drop_indices = np.random.choice(X[X['biased_label'] == 0].index, int(0.9*X[X['biased_label'] == 0].shape[0]), 
                replace=False).astype(int)
        elif task == 'regression':
            drop_indices = np.random.choice(
                X[X['biased_label'] < cutoff].index, 
                int(0.9*X[X['biased_label'] < cutoff].shape[0]), 
                replace=False).astype(int)

        X = X.drop(drop_indices)
    
    if X.shape[0] > 5400 and not output_for == 'dragonnet':
        new_drop_indices = np.random.choice(X.index, X.shape[0] - 5400, replace=False)
        drop_indices = np.concatenate([drop_indices, new_drop_indices]).astype(int)
        X = X.drop(new_drop_indices)

    X_heldout = X_full.iloc[drop_indices]
    embeds_list_text_heldout = embeds_list_text[drop_indices]
    embeds_image_heldout = embeds_image[drop_indices]

    embeds_list_text = np.delete(embeds_list_text, drop_indices, axis=0)
    embeds_image = np.delete(embeds_image, drop_indices, axis=0)

    drop_cols = ['host_about', 'price', 'label', 'pred_prob', 'pred_prob_img', 'split', 'biased_label', 'true_label']

    if modalities == 'tab' or output_for == 'dragonnet':
        drop_cols += ['picture_path', 'listing_text']

    if output_for == 'dragonnet':
        y = X['true_label'].values.astype('float32')
        if task == 'regression':
            t = (X['biased_label'].values > 3).astype('float32')
        elif task == 'multiclass':
            t = (X['biased_label'].values > 2).astype('float32')
        elif task == 'classification':
            t = X['biased_label'].values.astype('float32')
        x = X.drop(drop_cols, axis=1).to_numpy().astype('float32')

        if modalities == 'all':
            x = np.hstack([x, embeds_list_text, embeds_image])
        elif modalities == 'text':
            x = np.hstack([x, embeds_list_text])

        data = {'x':x,'t':t,'y':y,'t':t}
        data['t'] = data['t'].reshape(-1,1) #we're just padding one dimensional vectors with an additional dimension 
        data['y'] = data['y'].reshape(-1,1)

        return data

    y = X['true_label'].values.astype(int)
    if task == 'classification':
        t = X['biased_label'].values.astype(int)
    elif task == 'regression':
        t = (X['biased_label'].values > cutoff).astype(int)
    X = X.drop(drop_cols, axis=1)
    if representation == 'embeds' and modalities == 'all':
        X_text = X['listing_text'].values
        if modalities == 'all':
            X_image = X['picture_path'].values
        X = X.drop(['listing_text', 'picture_path'], axis=1)
    colnames = X.columns
    if representation == 'embeds' and not output_for == 'dragonnet':
        x = torch.tensor(X.to_numpy(dtype=float)).to(device)
    else:
        x = X.to_numpy()
    
    if task == 'classification':
        t_heldout = X_heldout['biased_label'].values.astype(int)
    elif task == 'regression':
        t_heldout = (X_heldout['biased_label'].values > cutoff).astype(int)
    y_heldout = X_heldout['true_label'].values.astype(int)
    X_heldout = X_heldout.drop(drop_cols, axis=1)
    if representation == 'embeds' and modalities in ['all', 'text']:
        X_heldout_text = X_heldout['listing_text'].values
        if modalities == 'all':
            X_heldout_image = X_heldout['picture_path'].values
        X_heldout = X_heldout.drop(['listing_text', 'picture_path'], axis=1)

    if representation == 'embeds' and not output_for == 'dragonnet':
        x_heldout = torch.tensor(X_heldout.to_numpy(dtype=float)).to(device)
    else:
        x_heldout = X_heldout.to_numpy()

    if modalities == 'all':
        if representation == 'embeds':
            if output_for == 'dragonnet':
                x = np.hstack([x, embeds_list_text, embeds_image])
                x_heldout = np.hstack([x_heldout, embeds_list_text_heldout, embeds_image_heldout])
            else:
                x = torch.cat([x, torch.tensor(embeds_list_text).to(device), torch.tensor(embeds_image).to(device)], axis=1)
                x_heldout = torch.cat([x_heldout, torch.tensor(embeds_list_text_heldout).to(device), torch.tensor(embeds_image_heldout).to(device)], axis=1)
            return (x, X_text, X_image, x_heldout, X_heldout_text, X_heldout_image, t, t_heldout, y, y_heldout, colnames)

    elif modalities == 'text':
        if representation == 'embeds':
            if output_for == 'dragonnet':
                x = np.hstack([x, embeds_list_text])
                x_heldout = np.hstack([x_heldout, embeds_list_text_heldout])
            else:
                x = torch.cat([x, torch.tensor(embeds_list_text).to(device)], axis=1)
                x_heldout = torch.cat([x_heldout, torch.tensor(embeds_list_text_heldout).to(device)], axis=1)
            return (x, X_text, x_heldout, X_heldout_text, t, t_heldout, y, y_heldout, colnames)
    
    return (x, x_heldout, t, t_heldout, y, y_heldout, colnames)