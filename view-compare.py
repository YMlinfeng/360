import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_paths', nargs='+')  # 接受多个文件夹路径
parser.add_argument('-o','--output_paths',default='gallery')  # 接受多个文件夹路径
parser.add_argument('-s','--step',default=1,type=int)  # 跳过几个文件
parser.add_argument('-r','--if_rank',action='store_true')  # 是否排序
args = parser.parse_args()

# def get_matching_images(input_paths):
#     image_dict = {}
#     for path in input_paths:
#         images = sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))],key=lambda x:int(x.split('.')[0]))
#         for image in images:
#             if image not in image_dict:
#                 image_dict[image] = [os.path.join(path, image)]
#             else:
#                 image_dict[image].append(os.path.join(path, image))
#     return image_dict
def get_matching_images(input_paths):
    image_dict = {}
    for path in input_paths:
        images = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        for image in images:
            if image not in image_dict:
                image_dict[image] = [os.path.join(path, image)]
            else:
                image_dict[image].append(os.path.join(path, image))
    return image_dict

def generate_html(image_dict,input_paths, output_file='gallery.html'):
    # 创建HTML内容
    html_content = '<html>\n<head>\n<meta charset="UTF-8">\n<title>图片库</title>\n'
    html_content += '<style>\n'
    html_content += '    .image-row { display: flex; }\n'
    html_content += '    .image-row img { margin-right: 10px; }\n'
    html_content += '</style>\n'
    rename="<br>".join([item.split('/')[-1] for item in input_paths])
    html_content += f'</head>\n<body>\n<h1>{rename}</h1>\n'
    # html_content += '</head>\n<body>\n<h1>图片库</h1>\n'
    
    for image, paths in image_dict.items():
        html_content += '<div class="image-row">'
        for path in paths:
            html_content += f'<img src="{path}" alt="{image}">\n'
        html_content += f'<p>{image}</p></div>\n'
    
    html_content += '</body>\n</html>'
    
    # 将HTML内容写入文件
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(html_content)
    
    print(f'HTML图片库已生成: {output_file}')

# 获取匹配的图片
args.input_paths=[item.strip('/') for item in args.input_paths]
args.input_paths=args.input_paths[::args.step]
if args.if_rank:
    args.input_paths = sorted(args.input_paths, key=lambda x: int(x.split('e')[-1].strip('/')))
image_dict = get_matching_images(args.input_paths)
# 使用函数，生成 HTML 图片库
generate_html(image_dict,args.input_paths,args.output_paths+'.html')
