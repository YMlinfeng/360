import copy
import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision import transforms
#from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
from einops import rearrange
from random import choice
from PIL import Image
import torchvision.transforms as T

import gzip
from io import BytesIO
import base64
from torch.utils.data import  IterableDataset
import random
import pdb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from torchvision.transforms.functional import crop
from pathlib import Path
import multiprocessing
import pickle
import numpy as np 
from multiprocessing import Pool
import glob
from functools import partial

def default_train(n_px):
    return [
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize(n_px),  # Image.BICUBIC
        T.RandomCrop(n_px),
        # T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]

class AigcExtractDataset(IterableDataset):
    def __init__(self, 
            args,
            **kwargs) -> None:
        super().__init__()
        self.sub_path_list = []
        for txt in [
                    '/home/jovyan/myh-data-ceph-0/code/pixart_alpha/PixArt-alpha-master/Aigc_laion.txt',  # 2b

                    ############ D3 : AIGC_15e, Laion_2b_16.8e, Flux_gen_5kw;  #################
                    # '/home/jovyan/boomcheng-data-shcdt/chengbo/datasets/AIGC_data/AIGC-all-1.5B/AIGC_list.txt',
                    # '/home/jovyan/boomcheng-data-shcdt/chengbo/datasets/laion_2b/laion_2b_All.txt',
                    # '/home/jovyan/maao-data-cephfs-0/dataspace/maao/projects/huggingface/FLUX/resources/images_and_annotations/laion_2b/training_lists/flux_v1.7_20241028.txt',
                   ]:
            with open(txt, 'r') as f:
                sub_path_list = f.read().strip().split('\n')
                self.sub_path_list.extend(sub_path_list)
        self.folder_path = '/home/jovyan/liushanyuan-sh-ceph/project/sub_project/lujunda/flux_ipa/project_diffusers/work/extract_file_paths1200w.txt'
        random.shuffle(self.sub_path_list)
        #self.sub_path_list*=100
        self.args=args
        self.resolution = args.resolution
        self.proportion_empty_prompts=args.proportion_empty_prompts
        self.image_transform = transforms.Compose(default_train(self.resolution))
        if 'clip_image_processor' in kwargs: 
            self.clip_image_processor = kwargs['clip_image_processor'] 
        else:
            self.clip_image_processor = None
        print (self.sub_path_list[:2])
        self.mult_GPU=True
        self.rank_res = 0
        self.world_size = 1


        # with Pool(num_workers) as pool:
        #     self.data_list = pool.map(self.load_pth_file, self.file_paths)
    
        try:
            self.rank_res = int(os.environ.get('RANK'))
            self.world_size = int(os.environ.get('WORLD_SIZE'))
            print('word_size, ', self.world_size)
            print('rank_res, ', self.rank_res)
            print('file_num, ', len(self.sub_path_list))
            #worker_info = torch.utils.data.get_worker_info()
            #print ('worker_info num_workers/id, ', worker_info.num_workers, worker_info.id)
        except Exception as e:
            self.mult_GPU=False
            print('mult_GPU Error', e)

    @staticmethod
    def load_pth_file(file_path):
        """
        加载单个 .pth 文件
        
        Args:
            file_path: .pth 文件路径
        
        Returns:
            加载的数据
        """
        try:
            data = torch.load(file_path)
            return data
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return None        
 
    def _sample_generator(self, intval_num):
        worker_info = torch.utils.data.get_worker_info()
        
        with open(self.folder_path, "r") as fr:
            for file_index, line in enumerate(fr):
                if file_index % (self.world_size*worker_info.num_workers) == worker_info.num_workers * intval_num + worker_info.id:    
                    try:
                        sub_path = line.strip() # self.sub_path_list[file_index]
                        succ = 0
                        # if file_index % (self.world_size*worker_info.num_workers) == worker_info.num_workers * intval_num + worker_info.id:                            
                        
                        data_info = {'img_hw': torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
                            'aspect_ratio': torch.tensor(1.)}
                        
                        #loaded_data = torch.load(sub_path)
                        loaded_data = torch.load(sub_path, map_location=torch.device('cpu'))
                        # with open(sub_path, 'rb') as f:
                        #     loaded_data = pickle.load(f)
                        #loaded_data = np.load(sub_path,  allow_pickle=True) 


                        #print(loaded_data.keys())
                        #print(loaded_data['t5'])
                        # print(sub_path)
                        # print(loaded_data['original'])
                        #print(loaded_data['clip'])

                    
                        line = loaded_data['original']
                        linesp = line.strip().split("#;#")
                        imgid, img_path, img_info = linesp

                        try:
                            img_info = json.loads(img_info.replace('\'', '"'))
                            prompt = img_info["en_caption"]
                            visual_quality_score = img_info["visual_quality_score"]
                        except:
                            visual_quality_score = 6.0
                            prompt = img_info.split("en_caption")[1][4:].split("text_video_align_score")[0][:-4]

                        if float(visual_quality_score) < 5:
                            continue
                        img = Image.open(img_path)
                        if self.clip_image_processor is not None: 
                            image_prompt = self.clip_image_processor(
                                images=img,
                                return_tensors="pt"
                            ).pixel_values
                        img = self.image_transform(img)

                        if random.random() < self.proportion_empty_prompts:
                            prompt = ""
                        
                        yield (img, prompt, image_prompt, loaded_data['clip'], loaded_data['t5'])
                        #return (img, prompt, data_info)
                        succ = 1
                    except Exception as e:
                        print (e, sub_path)

        
                
            
    def __iter__(self):
        sample_iterator = self._sample_generator(self.rank_res)
        return sample_iterator

    def __len__(self):
        return int(2e8*100)