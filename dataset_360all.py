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

def default_train(n_px):
    return [
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize(n_px),  # Image.BICUBIC
        T.RandomCrop(n_px),
        # T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
import functools
import threading

def debug_thread(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import pdb
            pdb.set_trace()
            return func(*args, **kwargs)
        except:
            import traceback
            print(f"Error in thread {threading.current_thread().name}")
            traceback.print_exc()
    return wrapper
class AigcDataset(IterableDataset):
    def __init__(self, 
            args,
            **kwargs) -> None:
        super().__init__()
        self.sub_path_list = []
        for txt in [##'/home/jovyan/myh-data-ceph-0/code/pixart_alpha/PixArt-alpha-master/Aigc_journeydb.txt', 
                    ##'/home/jovyan/myh-data-ceph-0/code/pixart_alpha/PixArt-alpha-master/Aigc_list_9000w_aethestic6.txt',
                    #'/home/jovyan/myh-data-ceph-0/code/pixart_alpha/PixArt-alpha-master/Aigc_list_9000w.txt',
                    '/home/jovyan/myh-data-ceph-0/code/pixart_alpha/PixArt-alpha-master/Aigc_laion.txt',  # 2b
                    #'/home/jovyan/liushanyuan-sh-ceph-new/project/train/sdxl_bdm/to_others/to_mayuhang/project-152/kolors_list_en.txt',
                    #'/home/jovyan/liushanyuan-sh-ceph-new/data/kolors/generate/kolors_list_1600_yuhang.txt',
                    #'/home/jovyan/boomcheng-data-shcdt/chengbo/datasets/GRIT-20M/grit_20m_all_12m_T2I_traindata_detail.json',
                    #'/home/jovyan/boomcheng-data-shcdt/chengbo/datasets/Flux_laion_2b_traininglist/flux_v1.6_20241016.txt', # 3000w
                    
                    ############ D3 : AIGC_15e, Laion_2b_16.8e, Flux_gen_5kw;  #################
                    # '/home/jovyan/boomcheng-data-shcdt/chengbo/datasets/AIGC_data/AIGC-all-1.5B/AIGC_list.txt',
                    # '/home/jovyan/boomcheng-data-shcdt/chengbo/datasets/laion_2b/laion_2b_All.txt',
                    # '/home/jovyan/maao-data-cephfs-0/dataspace/maao/projects/huggingface/FLUX/resources/images_and_annotations/laion_2b/training_lists/flux_v1.7_20241028.txt',
                   ]:
            with open(txt, 'r') as f:
                sub_path_list = f.read().strip().split('\n')
                self.sub_path_list.extend(sub_path_list)
                
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

   
    def _sample_generator(self, intval_num):
        worker_info = torch.utils.data.get_worker_info()
        for file_index in range(len(self.sub_path_list)):
            sub_path = self.sub_path_list[file_index]
            succ = 0
            if file_index % (self.world_size*worker_info.num_workers) == worker_info.num_workers * intval_num + worker_info.id:                            
               
                data_info = {'img_hw': torch.tensor([self.resolution, self.resolution], dtype=torch.float32),
                    'aspect_ratio': torch.tensor(1.)}
                
                # if 'journeydb' in sub_path:
                #     try:
                #         with open(sub_path) as fr:
                #             data = json.load(fr)
                #         for item in data:
                #             try:
                #                 img_path = os.path.join('/home/jovyan/diffusion-data-cephfs-new/liushanyuan/data/JourneyDB/OpenDataLab___JourneyDB/raw/JourneyDB/train/imgs/', item['img_path'])
                #                 img_path = img_path.replace('.png', '.jpg') #img_path[:-4]+'.jpg'
                #                 #if random.random() < 0.5:
                #                 #    prompt = item['prompt']
                #                 #else:
                #                 try:
                #                     prompt = item['Task2']['Caption']
                #                 except Exception as e:
                #                     print (e, sub_path)
                #                     prompt = item['prompt']
                #                 img = Image.open(img_path)
                #                 img = self.image_transform(img)

                #                 if random.random() < self.proportion_empty_prompts:
                #                     prompt = ""
                #                 yield (img, prompt)
                #                 #return (img, prompt, data_info)
                #                 succ = 1
                #                 #break
                #             except Exception as e:
                #                 print (e, sub_path)
                #     except Exception as e:
                #         print (e)
                
                # elif '/kolors/generate_en/' in sub_path:
                #     try:
                #         json_list = os.listdir(sub_path)
                #         for item in json_list:
                #             try:
                #                 with open(os.path.join(sub_path, item)) as fr:
                #                     prompt_dic = json.load(fr)
                #                     prompt = prompt_dic["caption"]
                                    
                #                 img_path = os.path.join('/home/jovyan/liushanyuan-sh-ceph-new/data/kolors/generate_en/image/caption_withAS_3B/', 
                #                                         '/'.join(sub_path.split('/')[-2:]), item.split('.')[0]+'.png')
                #                 img = Image.open(img_path)
                #                 img = self.image_transform(img)

                #                 if random.random() < self.proportion_empty_prompts:
                #                     prompt = ""
                #                 yield (img, prompt)
                #                 #return (img, prompt, data_info)
                #                 succ = 1
                #             except Exception as e:
                #                 print (e, sub_path)
                #     except Exception as e:
                #         print (e)
                    
                # elif '/kolors/generate/' in sub_path:
                #     try:
                #         json_list = os.listdir(sub_path)
                #         for item in json_list:
                #             try:
                #                 if 'low_res' in sub_path:
                #                     tmp_path = '/'.join(sub_path.split('/')[-4:])
                #                 else:
                #                     tmp_path = '/'.join(sub_path.split('/')[-3:])
                #                 with open(os.path.join(sub_path, item)) as fr:
                #                     prompt_dic = json.load(fr)
                #                     prompt = prompt_dic["caption"]
                                    
                #                 img_path = os.path.join('/home/jovyan/liushanyuan-sh-ceph-new/data/kolors/generate/image/', tmp_path, item.replace('.json', '.png'))
                #                 img = Image.open(img_path)
                #                 img = self.image_transform(img)

                #                 if random.random() < self.proportion_empty_prompts:
                #                     prompt = ""
                #                 yield (img, prompt)
                #                 #return (img, prompt, data_info)
                #                 succ = 1
                #             except Exception as e:
                #                 print (e, sub_path)
                #     except Exception as e:
                #         print (e)
                if 'diffusion-data-1' in sub_path:
                    sub_path = "/home/jovyan/%s" % sub_path
                    # pass
                    try:
                        with gzip.open(sub_path, 'r') as f:
                            for line in f:
                                try:
                                    datas = line.decode("utf-8").strip().split("\t")
                                    imgkey, title, imgb64, class_key = datas[:4]
                                    #bs = float(class_key.split(';')[1].split(':')[1])
                                    #if bs < 6.0:
                                    #    continue
                                    img = Image.open(BytesIO(base64.urlsafe_b64decode(imgb64)))
                                    prompt = title

                                    base_w, base_h = img.size
                                    if min(base_w, base_h) < self.resolution:
                                        continue

                                    img = self.image_transform(img)

                                    if random.random() < self.proportion_empty_prompts:
                                        prompt = ""

                                    yield (img, prompt)
                                    succ = 1
                                except Exception as e:
                                    print (e, sub_path)
                    except Exception as e:
                        print (e)
                elif 'laion-2b-decompress' in sub_path:
                    # pass
                    
                    gzip_path = sub_path
                    try:
                        with gzip.open(gzip_path,'r') as f:
                            for line in f:
                                
                                try:
                                    line = line.decode("utf-8").strip()
                                    datas = line.split("\t")
                                    img_url, img_width, img_height, title, imgb64 = datas[:5]

                                    prompt = title
                                    img = Image.open(BytesIO(base64.urlsafe_b64decode(imgb64)))
                                    base_w, base_h = img.size
                                    if min(base_w, base_h) < self.resolution:
                                        continue

                                    img = self.image_transform(img)
                                    if random.random() < self.proportion_empty_prompts:
                                        prompt = ""

                                    yield (img, prompt)
                                    succ = 1
                                except Exception as e:
                                    print (e, sub_path)
                    except Exception as e:
                        print (e)


                # elif 'aev_AIGC' in sub_path:
                #     try:
                #         each_recaption_path = '/home/jovyan/liushanyuan-sh-ceph-new/data/AIGC/recaption/AIGC-high-quality-90M/longcap_output/' +'/'.join(sub_path.split('/')[-2:])+'.json'
                #         with open(each_recaption_path, 'r') as f:
                #             recaption_dict=json.load(f)
                            
                #         each_recaption_path_short = '/home/jovyan/liushanyuan-sh-ceph/data/AIGC/recaption-liangdawei/output_data/' +'/'.join(sub_path.split('/')[-2:])+'.json'
                #         with open(each_recaption_path_short, 'r') as f:
                #             recaption_short_dict=json.load(f)
                #     except Exception as e:
                #         recaption_dict={}
                #         recaption_short_dict = {}

                #     try:
                #         with gzip.open(sub_path, 'r') as f:
                #             for line in f:
                #                 try:
                #                     datas = line.decode("utf-8").strip().split("\t")
                #                     imgkey, title, imgb64, class_key = datas[:4]
                #                     #bs = float(class_key.split(';')[1].split(':')[1])
                #                     #if bs < 6.0:
                #                     #    continue
                #                     img = Image.open(BytesIO(base64.urlsafe_b64decode(imgb64)))

                #                     base_w, base_h = img.size
                #                     if min(base_w, base_h) < self.resolution:
                #                         continue

                #                     img = self.image_transform(img)
                #                     if recaption_dict == {}:
                #                         prompt = title
                #                     else:
                #                         try:
                #                             rd = random.random()
                #                             if rd < 0.4:
                #                                 prompt = title+' '+recaption_dict[imgkey+title]['long_caption']
                #                             elif rd < 0.7:
                #                                 prompt = title
                #                             else:
                #                                 prompt = recaption_short_dict[imgkey+title]['caption']
                #                         except:
                #                             prompt = title

                #                     if random.random() < self.proportion_empty_prompts:
                #                         prompt = ""

                #                     yield (img, prompt)
                #                     #return (img, prompt, data_info)
                #                     succ = 1
                #                 except Exception as e:
                #                     print (e, sub_path)
                #     except Exception as e:
                #         print (e)
                            
                elif 'laion2b' in sub_path:
                    #import pdb;pdb.set_trace()
                    #import threading;threading.settrace()
                    print(f'sub_path:{sub_path}')
                    try:
                        with open(sub_path) as fr:
                            http_dict = json.load(fr)
                            
                        gzip_path = '/home/jovyan/lmm-shcdt-datasets/laion-2b-decompress-as-gzip/' + sub_path.split('/')[-2] + '/' + sub_path.split('/')[-1].split('.')[0] + '.gz'
                        print(f'gzip_path:{gzip_path}')
                        with gzip.open(gzip_path,'r') as f:
                            for line in f:
                                #print(f'line:{line}')
                                try:
                                    line = line.decode("utf-8").strip()
                                    datas = line.split("\t")
                                    url = datas[0].split(': ')[1].replace('"', '')
                                    if url not in http_dict:
                                        continue
                                    #aesthetic_score = http_dict[url]['aesthetic_score']
                                    #if float(aesthetic_score) < 6.0:
                                    #    continue
                                    #pdb.set_trace()
                                    
                                    #width = int(datas[1].split(': ')[1])
                                    #height = int(datas[2].split(': ')[1])
                                    imgb64 = datas[-1]
                                    img = Image.open(BytesIO(base64.urlsafe_b64decode(imgb64)))

                                    base_w, base_h = img.size
                                    if min(base_w, base_h) < self.resolution:
                                        continue
                                    if self.clip_image_processor is not None: 
                                        image_prompt = self.clip_image_processor(
                                            images=img,
                                            return_tensors="pt"
                                        ).pixel_values
                                    img = self.image_transform(img)

                                    if random.random() < 0.5:
                                        prompt = http_dict[url]['long_caption']
                                    else:
                                        prompt = http_dict[url]['title']
                                    #aesthetic_score = http_dict[url]['aesthetic_score']

                                    if random.random() < self.proportion_empty_prompts:
                                        prompt = ""
                                    yield (img, prompt) if self.clip_image_processor is None else (img, prompt, image_prompt)
                                    #return (img, prompt, data_info)
                                    succ = 1
                                except Exception as e:
                                    print (e, gzip_path)
                    except Exception as e:
                        print(e)
                elif 'FLUX' in sub_path:
                    try:
                        with open(sub_path, "r") as fr:
                            for line in fr:
                                linesp = line.strip().split("#;#")
                                imgid, img_path, img_info = linesp

                                #pdb.set_trace()
                                #img_info = json.loads(img_info)
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
                                img = self.image_transform(img)

                                if random.random() < self.proportion_empty_prompts:
                                    prompt = ""
                                yield (img, prompt)
                                #return (img, prompt, data_info)
                                succ = 1
                    except Exception as e:
                        print (e, sub_path)
                # elif 'GRIT-20M' in sub_path:
                #     #
                #     try:
                #         with open(sub_path) as fr:
                #             data_dict = json.load(fr)
                #         root_image_grit = "/home/jovyan/liushanyuan-sh-ceph-new/data/"
                #         for dot in data_dict:
                #             _key, _w, _h, _path_image, caption_base_en, caption_base_cn, caption_long_en, caption_long_cn = dot
                #             list_caption = [caption_base_en, caption_base_cn, caption_long_en, caption_long_cn]
                #             prompt = random.sample(list_caption, 1)[0]
                #             path_image = os.path.join(root_image_grit, _path_image)
                #             try:
                #                 img = Image.open(path_image)
                #             except Exception as e:
                #                 print (e, path_image)
                #                 continue
                #             base_w, base_h = img.size
                #             if min(base_w, base_h) < self.resolution:
                #                 continue
                #             img = self.image_transform(img)

                #             if random.random() < self.proportion_empty_prompts:
                #                 prompt = ""
                #             yield (img, prompt)
                #     except Exception as e:
                #         print (e, sub_path)
                            
                #if 0:
                    #if random.random() < self.proportion_empty_prompts:
                    #    prompt = ""
                    #out_info = {}
                    #out_info.setdefault("prompts", prompt)
                    #out_info.setdefault("images", img)
                    #yield out_info
                    #print ("=====")
                    #yield (img, prompt)
                    #return (img, prompt)
                    #return (img, prompt, data_info)
                #else:
                #    print ("-------- None Return. %s" % sub_path)
            
    def __iter__(self):
        sample_iterator = self._sample_generator(self.rank_res)
        return sample_iterator

    def __len__(self):
        return int(2e8*100)