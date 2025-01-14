#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import os
from pathlib import Path

# save_folder = '/home/jovyan/liushanyuan-sh-ceph/project/sub_project/lujunda/flux_ipa/project_diffusers/work/extract_text_emded_test/'

# loaded_data = torch.load(f'{save_folder}101.pth')
# print(loaded_data.keys())
# print(loaded_data['t5'])
# print(loaded_data['original'])
# print(loaded_data['clip'])

# print('1')
folder_path = '/home/jovyan/liushanyuan-sh-ceph/project/sub_project/lujunda/flux_ipa/project_diffusers/work/extract_text_emded/'
# print('2')
for root, dirs, files in os.walk(folder_path):
    for file in files:
        full_path = os.path.join(root, file)
        import pdb;pdb.set_trace()
        if full_path.endswith(".pt"):
            
            print(full_path)





