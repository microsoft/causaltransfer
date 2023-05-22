import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch
import argparse
import pdb
from tqdm import tqdm
import os
from torch.utils.data import Dataset
from transformers import pipeline, AutoTokenizer, AutoFeatureExtractor, AutoModel
from tqdm.auto import tqdm
from utils import download_image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir')
    parser.add_argument('--lm-library', default='transformers')
    parser.add_argument('--model', help='embedding model')
    parser.add_argument('--filename')
    parser.add_argument('--output-dir')
    parser.add_argument('--output-filename')
    parser.add_argument('--data-type')
    parser.add_argument('--data-col')
    args = parser.parse_args()

    return args

class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list
    
    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]

args = get_args()
df = pd.read_csv(os.path.join(args.data_dir, args.filename))
df.reset_index(inplace=True, drop=True)
stem = args.filename.split('.')[0].split('_')[0]

if args.data_type == 'image':
    df_full = pd.read_csv(os.path.join(args.data_dir, 'airbnb_train_full.csv'))
    df_full2 = pd.read_csv(os.path.join(args.data_dir, 'airbnb_test_full.csv'))
    df_full = df_full.append(df_full2, ignore_index=True)

print(df.shape)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# pdb.set_trace()

if args.data_type == 'text':
    if args.lm_library == 'sentence_transformers':
        model = SentenceTransformer(args.model)
        model.to(device)
        embeds = model.encode(df[args.data_col].astype(str).tolist(), show_progress_bar=True)
    elif args.lm_library == 'transformers':
        tokenizer = AutoTokenizer.from_pretrained('microsoft/mpnet-base', max_length=512, truncation=True)
        feature_extraction = pipeline('feature-extraction', model=args.model, tokenizer=tokenizer, device=0,
            padding=True, truncation=True)
        inputs = ListDataset(df[args.data_col].astype(str).tolist())
        output_list = []
        for out in tqdm(feature_extraction(inputs, batch_size=64), total=len(inputs)):
            output_list.append(out[0][0])
        embeds = np.vstack(output_list)
elif args.data_type == 'image':
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.to(device)

    div_num = 5000

    embeds_list = []

    for filepath in tqdm(df[args.data_col]):
        try:
            filepath_trunc = filepath.split('/')[-1]
            img = Image.open(os.path.join(args.data_dir, 'images/', filepath_trunc)).convert(mode='RGB')
        except:
            img_url = df_full[df_full['picture_url'].str.contains(filepath_trunc)]['picture_url'].values[0]
            print('Downloading {}'.format(img_url))
            download_image(img_url, os.path.join(args.data_dir, 'images/'))
            img = Image.open(os.path.join(args.data_dir, 'images/', filepath_trunc)).convert(mode='RGB')
        inputs = feature_extractor(images=img, return_tensors='pt')
        img.close()
        del(img)
        torch.cuda.empty_cache()
        inputs.to(device)
        outputs = model(**inputs)
        out = outputs.last_hidden_state.mean(axis=1)
        embeds_list.append(out.detach().cpu())
        del(inputs)
        del(outputs)
        del(out)

    embeds = np.vstack(embeds_list)

np.save('{}{}'.format(args.output_dir, args.output_filename), embeds)