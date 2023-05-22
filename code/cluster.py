import pandas as pd
import pdb
import argparse
import numpy as np
from sklearn.cluster import KMeans
from utils import load_airbnb_data, load_clothing_data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-split', default='08')
    parser.add_argument('--cluster-modalities', default='tab')
    parser.add_argument('--noise', default='strong')
    parser.add_argument('--labelgen-model', default='_lr')
    parser.add_argument('--modality-prop', default='060202')
    parser.add_argument('--num-clusters', default=5)
    parser.add_argument('--data-dir')
    parser.add_argument('--label-type', default='real')
    parser.add_argument('--task', default='classification')
    parser.add_argument('--text-cols', default='text')
    parser.add_argument('--dataset')
    args = parser.parse_args()

    return args

args = get_args()

if args.dataset == 'airbnb':
    X_train, X_train_heldout, t_train, t_train_heldout, y_train, X_cols = load_airbnb_data(
        split='train', data='biased', label_split=args.label_split, modalities=args.cluster_modalities, 
        representation='embeds', noise=args.noise, labelgen_model=args.labelgen_model,
        modality_prop=args.modality_prop, output_for='ganite', label_type=args.label_type,
        data_dir=args.data_dir, task=args.task, text_cols=args.text_cols
)
elif args.dataset == 'clothing_review':
    X_train, X_train_text, _, X_train_heldout, X_train_heldout_text, _, t_train, t_train_heldout, y_train, X_cols = load_clothing_data(
        split='train', data='biased', label_split=args.label_split, modalities=args.cluster_modalities, 
        representation='embeds', noise=args.noise, labelgen_model=args.labelgen_model,
        modality_prop=args.modality_prop, output_for='ganite', label_type=args.label_type,
        data_dir=args.data_dir, task=args.task, text_cols=args.text_cols
    )

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.detach().cpu())

model = KMeans(n_clusters=int(args.num_clusters), random_state=2837)

model.fit(X_train)
print(X_train.shape)

clusters = model.predict(X_train)

if args.text_cols == 'all':
    np.save(os.path.join(args.data_dir, args.dataset, 'splitB_clusters{}_{}_alltext.npy'.format(
        args.num_clusters, args.label_type)), clusters)
else:
    np.save(os.path.join(args.data_dir, args.dataset, 'splitB_clusters{}_{}.npy'.format(
        args.num_clusters, args.label_type)), clusters)