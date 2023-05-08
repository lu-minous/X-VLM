# !/usr/bin/env python3
import argparse
import os
import sys
import math

import numpy as np
import ruamel.yaml as yaml

import time
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.model_retrieval import XVLM

from models.tokenization_bert import BertTokenizer
import utils

import json
from collections import OrderedDict


def get_transforms():
    # Data Preprocessing
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    return test_transform


def get_images_texts(images_root, texts_root, transform_ops, test=False):
    images = []
    texts = []

    f = open(texts_root, 'r')
    for line in f.readlines():
        text = line.strip()
        texts.append(text)

    idx_dic = {}
    cnts = 0
    images_names = os.listdir(images_root)
    for image_name in images_names:
        img = Image.open(os.path.join(images_root, image_name)).convert('RGB')
        img = transform_ops(img)
        images.append(img)
        idx_dic[cnts] = image_name
        cnts += 1

    if test:
        images = images[:20]
        texts = texts[:20]

    return images, texts, idx_dic

@torch.no_grad()
def get_similarity(images, texts, device):

    model.eval()
    print('Computing features for test...')
    start_time = time.time()  
    device = torch.device("cuda:0")

    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])   
    #tokenizer = SimpleTokenizer()

    num_image = len(images)
    num_text = len(texts)
    step = 5 #save memory
    
    text_feats = []
    text_embeds = []  
    text_atts = []
    for i in range(0, num_text, step):
        if i % 1000 == 0:
            print(f'{i}/{num_text}')
        text = texts[i: min(num_text, i + step)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    print('text_embeds.shape', text_embeds.shape)
    print('text_feats.shape', text_feats.shape)
    print('text_atts.shape', text_atts.shape)

    image_feats = []
    image_embeds = []    
    for i in range(0, num_image, step):
        if i % 1000 == 0:
            print(f'{i}/{num_image}')
        image = images[i: min(num_image, i + step)]
        image_input = torch.stack(image, dim=0).to(device)
        image_feat = model.vision_encoder(image_input)
        image_embed = F.normalize(model.vision_proj(image_feat[:, 0, :]))
        
        image_embeds.append(image_embed)
        image_feats.append(image_feat)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    print('image_feats.shape', image_feats.shape)
    print('image_embeds.shape', image_embeds.shape)

    sims_matrix = torch.mm(text_embeds, image_embeds.t())
    sims = sims_matrix.cpu().numpy()
    print('sims.shape', sims.shape)

    #calculate score
    k_test = 128
    score_matrix_t2i = torch.full((num_text, num_image), -100, dtype=torch.float).to(device)
    for i, sims in enumerate(sims):
        topk, topk_idx = sims.topk(k_test, dim=0)
        encoder_output = image_feats[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds=text_feats[i].repeat(k_test, 1, 1),
                                    attention_mask=text_atts[i].repeat(k_test, 1),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion'
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[i, topk_idx] = score
        print('score_matrix_t2i.shape', score_matrix_t2i.shape)
    end_time = time.time()
    print('Computing features cost time: ', end_time - start_time)
 
    return sims


def infer(similarity, idx_dic, texts):
    similarity_argsort = np.argsort(-similarity, axis=1)
    print('similarity_argsort.shape', similarity_argsort.shape)

    topk = 10
    result_list = []
    for i in range (len(similarity_argsort)):
        dic = {'text': texts[i], 'image_names': []}
        for j in range(topk):
            '''
            if(i < 7611):   #car 
                dic['image_names'].append(idx_dic[similarity_argsort[i,-j-1]])
            else:   #person
                dic['image_names'].append(idx_dic[similarity_argsort[i,j]])
            '''
            dic['image_names'].append(idx_dic[similarity_argsort[i,j]])
        result_list.append(dic)
    with open('infer_json.json', 'w') as f:
        f.write(json.dumps({'results': result_list}, indent=4))
        


if __name__ == '__main__':
    # pretrained = './pretrained/vitbase_clip.pdparams'

    images_root = './images/bd-images/test/'
    texts_root = './images/bd-images/test_text.txt' 

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    print('args', args)

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    #device = torch.device("cuda:0")
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    model = XVLM(config=config)
    model.load_pretrained(args.checkpoint, config, is_eval=True)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # model = build_model(embed_dim=512,
    #                     image_resolution=224,
    #                     vision_layers=12,
    #                     vision_width=768,
    #                     vision_patch_size=32,
    #                     context_length=77,
    #                     vocab_size=49408,
    #                     transformer_width=512,
    #                     transformer_heads=8,
    #                     transformer_layers=12,
    #                     qkv_bias=True,
    #                     pre_norm=True,
    #                     proj=True,
    #                     patch_bias=False)

    # model = load_pretrained(model, pretrained)

    transform_ops = get_transforms()
    images, texts, idx_dic = get_images_texts(images_root, texts_root, transform_ops, test=False)


    similarity = get_similarity(images, texts, device)

    infer(similarity, idx_dic, texts)

#python3 xvlm_infer_json.py --checkpoint ./output/bd_itr_20e/checkpoint_19.pth --config ./configs/Retrieval_bd.yaml
python3 -m torch.distributed.launch --nproc_per_node={:}  --nnodes=1 ".format(2) --use_env xvlm_score_infer_json.py --config ./configs/Retrieval_bd.yaml --checkpoint ./output/bd_itr_v3/checkpoint_s1_20e.pth
