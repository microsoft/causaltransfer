import pandas as pd
import os
import pdb
import argparse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor, DefaultDataCollator, DataCollatorWithPadding, AutoModel, AutoModelForSequenceClassification, AutoModelForImageClassification, TrainingArguments, Trainer, AutoImageProcessor
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, ToPILImage
from datasets import Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from trainer import IRMTrainer, CITrainer
from tqdm import tqdm
from utils import load_airbnb_data, load_clothing_data, download_image
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
pd.set_option('display.max_columns', None)

def preprocess(df, text_col, tokenizer):
    return tokenizer(df[text_col], truncation=True, padding="max_length")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def transform(example):
    example['pixel_values'] = [_transforms(img.convert(mode='RGB')) for img in example['image']]
    # del example['image']
    return example

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-split', default='08')
    parser.add_argument('--label-type', default='synthetic')
    parser.add_argument('--dataset', default='airbnb')
    parser.add_argument('--text-cols', default='text')
    parser.add_argument('--text-col', default='listing_text')
    parser.add_argument('--num-clusters', default=5)
    parser.add_argument('--noise', default='strong')
    parser.add_argument('--model', default='distilbert-base-uncased')
    parser.add_argument('--lm-library', default='transformers')
    parser.add_argument('--task', default='classification')
    parser.add_argument('--data-type')
    parser.add_argument('--labelgen-model', default='_lr')
    parser.add_argument('--modality-prop', default='060202')
    parser.add_argument('--metric', default='f1')
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--data-dir')
    parser.add_argument('--irm', action='store_true')
    parser.add_argument('--counterfactual-invariance', action='store_true')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    return args

start_time = time.time()

args = get_args()

torch.cuda.empty_cache()

if args.labelgen_model == None:
    args.labelgen_model = ''
if args.irm == None:
    args.irm = False
if args.counterfactual_invariance == None:
    args.counterfactual_invariance = False

data_dir = '{}data/'.format(args.data_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

if args.dataset == 'airbnb':

    X_train, X_train_heldout, t_train, t_train_heldout, y_train, y_train_heldout, X_cols = load_airbnb_data(
        split='train', data='biased', label_split=args.label_split, modalities='all', 
        representation='raw', noise=args.noise, labelgen_model=args.labelgen_model, modality_prop=args.modality_prop, 
        output_for='ganite', label_type=args.label_type, data_dir=data_dir, task=args.task)

    target = 'price'
    
elif args.dataset == 'clothing_review':

    X_train, X_train_heldout, t_train, t_train_heldout, y_train, y_train_heldout, X_cols = load_clothing_data(
        split='train', data='biased', label_split=args.label_split, modalities='all',
        representation='raw', noise=args.noise, labelgen_model=args.labelgen_model, modality_prop=args.modality_prop, output_for='ganite',
        label_type=args.label_type, data_dir=data_dir, task=args.task, text_cols=args.text_cols)
    
    target = 'rating'

train = pd.DataFrame(X_train, columns=X_cols)
train['label'] = y_train
if args.task == 'regression':
    train['label'] = train['label'].astype(float)

if args.irm or args.counterfactual_invariance:
    if args.text_cols == 'all':
        clusters = np.load(os.path.join(data_dir, '{}/splitB_clusters{}_{}_alltext.npy'.format(
            args.dataset, args.num_clusters, args.label_type)))
    else:
        clusters = np.load(os.path.join(data_dir, '{}/splitB_clusters{}_{}.npy'.format(
            args.dataset, args.num_clusters, args.label_type)))

    train, val, train_clusters, val_clusters = train_test_split(train, clusters, test_size=0.3, random_state=230128)

else:
    train, val = train_test_split(train, test_size=0.3, random_state=230128)

if args.dataset == 'airbnb':
    df_full = pd.read_csv(os.path.join(data_dir, 'airbnb/airbnb_train_full.csv'))

if args.lm_library == 'sentence_transformers':
    model = SentenceTransformer(args.model)
    model.to(device)

    if args.data_type == 'text':
        train = [InputExample(texts=[row[[args.text_col]]], label=row['label']) for index, row in train.iterrows()]
    elif args.data_type == 'image':
        from PIL import Image

        img_list = []
        for index, row in tqdm(train.iterrows(), total=train.shape[0]):
            filepath = row['picture_path'].split('/')[-1]
            try:
                img = Image.open(os.path.join(data_dir, '{}/images/{}'.format(args.dataset, filepath)))
            except:
                img_url = df_full[df_full['picture_url'].str.contains(filepath)]['picture_url'].values[0]
                print('Downloading {}'.format(img_url))
                download_image(img_url, os.path.join(data_dir, '{}/images/'.format(args.dataset)))
                img = Image.open(os.path.join(data_dir, '{}/images/'.format(args.dataset), filepath))
            img = img.convert(mode='RGB')
            img_list.append(img)
        train = [InputExample(texts=[img_list[i]], label=y_train[i]) for i in range(len(y_train))]

    if args.noise == None:
        args.noise = ''
    else:
        args.noise = '_tabnoise_{}'.format(args.noise)
    
    print('Training model')

    dataloader = DataLoader(train, shuffle=True, batch_size=32)
    loss = losses.BatchAllTripletLoss(model=model)

    model.fit(train_objectives=[(dataloader, loss)], epochs=5, warmup_steps=1000, weight_decay=0.,
        output_path='{}models/{}_finetune_{}_{}{}{}'.format(args.data_dir, args.model, args.label_split, args.modality_prop, args.noise, args.labelgen_model),
        )
    
    if args.data_type == 'image':
        for image in img_list:
            image.close()

elif args.lm_library == 'transformers':

    # metric = load_metric(args.metric)

    # train_df, test_df = train_test_split(train, random_state=120123)
    if args.data_type == 'text':

        tokenizer = AutoTokenizer.from_pretrained(args.model, max_length=512, truncation=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        if args.task == 'classification':
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
        elif args.task == 'multiclass':
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(np.unique(y_train)))
        else:
            model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)
        model.to(device)

        train_nas = train[[args.text_col]].isna().values.flatten()
        train_df = Dataset.from_pandas(train[[args.text_col, 'label']].dropna(subset=[args.text_col]))
        val_nas = val[[args.text_col]].isna().values.flatten()
        val_df = Dataset.from_pandas(val[[args.text_col, 'label']].dropna(subset=[args.text_col]))

        print('Preprocessing data')

        train_df = train_df.map(preprocess, fn_kwargs={'text_col': args.text_col, 'tokenizer': tokenizer}, batched=True)
        train_df = train_df.remove_columns(args.text_col)
        val_df = val_df.map(preprocess, fn_kwargs={'text_col': args.text_col, 'tokenizer': tokenizer}, batched=True)
        val_df = val_df.remove_columns(args.text_col)


    elif args.data_type == 'image':

        from datasets import Image

        train_filepaths = [os.path.join(data_dir, '{}/images/{}'.format(args.dataset, filename.split('/')[-1]))
            for filename in train['picture_path']]
        val_filepaths = [os.path.join(data_dir, '{}/images/{}'.format(args.dataset, filename.split('/')[-1]))
            for filename in val['picture_path']]
        train_df = Dataset.from_dict({'image': train_filepaths,
            'label': train['label'].tolist()}).cast_column('image', Image())
        val_df = Dataset.from_dict({'image': val_filepaths,
            'label': val['label'].tolist()}).cast_column('image', Image())
        data_collator = DefaultDataCollator()

        tokenizer = AutoImageProcessor.from_pretrained(args.model)

        normalize = Normalize(mean=tokenizer.image_mean, std=tokenizer.image_std)
        _transforms = Compose([RandomResizedCrop(tokenizer.size['height']), ToTensor(), normalize])

        train_df = train_df.map(transform, remove_columns=['image'], batched=True)
        val_df = val_df.map(transform, remove_columns=['image'], batched=True)

        model = AutoModelForImageClassification.from_pretrained(args.model)

        if args.task == 'classification':
            model.config.problem_type = 'single_label_classification'
            model.config.num_labels = 2
        elif args.task == 'multiclass':
            model = AutoModelForImageClassification.from_pretrained(args.model, num_labels=len(np.unique(y_train)))
        else:
            model = AutoModelForImageClassification.from_pretrained(args.model, num_labels=1)
        model.to(device)

    if args.irm or args.counterfactual_invariance:
        if args.data_type == 'text':
            train_clusters = train_clusters[~train_nas]
            val_clusters = val_clusters[~val_nas]

        train_labels = np.transpose(np.vstack([np.array(train_df['label']), train_clusters]))
        train_df = train_df.remove_columns('label')
        train_df = train_df.add_column('label', train_labels.tolist())

        val_labels = np.transpose(np.vstack([np.array(val_df['label']), val_clusters]))
        val_df = val_df.remove_columns('label')
        val_df = val_df.add_column('label', val_labels.tolist())

    if args.noise == None:
        args.noise = ''
    else:
        args.noise = '_tabnoise_{}'.format(args.noise)

    regtype = ''
    if args.irm:
        regtype = '_irm_cluster{}'.format(args.num_clusters)
    elif args.counterfactual_invariance:
        regtype = '_ci_cluster{}'.format(args.num_clusters)

    if args.label_type == 'synthetic':
        output_dir = '{}models/{}/{}_finetune_{}_{}{}{}{}/epochs_{}/'.format(args.data_dir, args.dataset, args.model, args.label_split, args.modality_prop, args.noise, args.labelgen_model, 
            regtype, args.num_epochs)
    else:
        if args.dataset == 'clothing_review':
            output_dir = '{}models/{}/{}_finetune_{}_{}_{}{}/epochs_{}/'.format(args.data_dir, args.dataset, args.model, target, args.text_col, args.task, regtype, args.num_epochs)
        else:
            output_dir = 'models/{}/{}_finetune_{}_{}{}/epochs_{}/'.format(args.data_dir, args.dataset, args.model, target, args.task, regtype, args.num_epochs)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.lr,
        warmup_steps=1000,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True
    )
    
    if args.irm:
        trainer = IRMTrainer(
            model=model,
            args=training_args,
            train_dataset=train_df,
            eval_dataset=val_df,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
    
    elif args.counterfactual_invariance:
        trainer = CITrainer(
            model=model,
            args=training_args,
            train_dataset=train_df,
            eval_dataset=val_df,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

    else:

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_df,
            eval_dataset=val_df,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

    if not args.eval:

        trainer.train()

        trainer.save_model('{}best_model/'.format(output_dir))
    
    else:
        trainer.evaluate()

end_time = time.time()

print("Runtime in seconds: {}".format(end_time - start_time)) 