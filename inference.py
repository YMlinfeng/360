import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn

#from pipeline_flux_ipa import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

from diffusers.models.attention_processor import IPAFluxAttnProcessor2_0, ConcatFluxAttnProcessor2_0
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

from transformers import AutoProcessor, SiglipVisionModel
from train_dit_all_control import IPAdapter_FLUX
#from train_dit_all_control import IPAdapter_FLUX
from safetensors.torch import load_file
from einops import rearrange,repeat
from tqdm import tqdm



ipadapter_path = "./ip-adapter.bin"
ipadapter_path = "/home/jovyan/liushanyuan-sh-ceph/project/sub_project/lujunda/flux_ipa/project_diffusers/project-3/output/checkpoint-23000/ip.safetensors"
ipadapter_path = "/home/jovyan/liushanyuan-sh-ceph/project/sub_project/lujunda/flux_ipa/project_diffusers/project-3/output/checkpoint-3000/ip.safetensors"
ipadapter_path = "/home/jovyan/liushanyuan-sh-ceph/project/sub_project/lujunda/flux_ipa/project_diffusers/project-6/output/checkpoint-11000/pytorch_model_fsdp.bin"
ipadapter_path = "/home/jovyan/liushanyuan-sh-ceph/project/sub_project/lujunda/flux_ipa/project_diffusers/project-6/output/checkpoint-5/pytorch_model_fsdp.bin"
ipadapter_path = "/home/jovyan/liushanyuan-sh-ceph/project/sub_project/lujunda/flux_ipa/project_diffusers/project-7/output/checkpoint-6000/pytorch_model_fsdp.bin"
ipadapter_path = "/liushanyuan-sh-ceph/project/sub_project/mengzijie/FLUX/project/project_9/output/checkpoint-6000/pytorch_model_fsdp.bin"
ipadapter_path = "/home/jovyan/liushanyuan-sh-ceph/project/sub_project/mengzijie/FLUX/project/project_11/output/checkpoint-104000/pytorch_model_fsdp.bin"




image_dir = "./assets/images/2.png"
image_dir = "/home/jovyan/liushanyuan-sh-ceph/project/sub_project/lujunda/flux_ipa/project/project-5/assets/images/elon.jpeg"
#image_dir = "/liushanyuan-sh-ceph/project/sub_project/lujunda/flux_ipa/project/project-5/assets/images/animation.jpeg"
image_dir = "/home/jovyan/liushanyuan-sh-ceph/project/sub_project/lujunda/flux_ipa/project/project-5/assets/images/ai_face2.png"
image_dir = "/home/jovyan/liushanyuan-sh-ceph/project/sub_project/lujunda/flux_ipa/project/project-5/assets/images/elon.jpeg"

prompt = "a young girl"
prompt = "woman with red hair, playing chess at the park, bomb going off in the background"
prompt = "A Side View of a majestic horse galloping across a field, showcasing its powerful muscles and elegant movement."
prompt = "A figure stands in a misty landscape, wearing a mask with antlers and dark, embellished attire, exuding mystery and otherworldlines"
prompt = "A Side View of a man lost in deep thought, with the warm glow of the setting sun casting dramatic shadows on their face"


current_device = torch.cuda.current_device()
#device = "cuda"
device = torch.device(f'cuda:{current_device}')

class MLPProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x



def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image





transformer = FluxTransformer2DModel.from_pretrained(
    "/home/jovyan/liushanyuan-sh-ceph/model/cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/303875135fff3f05b6aa893d544f28833a237d58", subfolder="transformer", torch_dtype=torch.bfloat16
)
image_encoder_path = "/home/jovyan/liushanyuan-sh-ceph/model/cache/huggingface/hub/models--google--siglip-so400m-patch14-384"

pipe = FluxPipeline.from_pretrained(
    "/home/jovyan/liushanyuan-sh-ceph/model/cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/303875135fff3f05b6aa893d544f28833a237d58", 
    transformer=transformer, torch_dtype=torch.bfloat16,
).to(device, dtype=torch.bfloat16)

image_encoder = SiglipVisionModel.from_pretrained(image_encoder_path).to(device, dtype=torch.bfloat16)
clip_image_processor = AutoProcessor.from_pretrained(image_encoder_path)

#ip_model = IPAdapter(pipe, image_encoder_path, ipadapter_path, device="cuda", num_tokens=128)




# ipa 模型加载

loaded = torch.load(ipadapter_path, map_location="cpu")


num_tokens=4 # 128?

transformer = pipe.transformer
ip_attn_procs = {} # 19+38=57
for name in transformer.attn_processors.keys():
    if name.startswith("transformer_blocks.") or name.startswith("single_transformer_blocks"):
        # ip_attn_procs[name] = ConcatFluxAttnProcessor2_0(
        ip_attn_procs[name] = IPAFluxAttnProcessor2_0(
            hidden_size=transformer.config.num_attention_heads * transformer.config.attention_head_dim,
            cross_attention_dim=transformer.config.joint_attention_dim,
            num_tokens=num_tokens,
        ).to(device, dtype=torch.bfloat16)
    else:
        ip_attn_procs[name] = transformer.attn_processors[name]

transformer.set_attn_processor(ip_attn_procs)


adapter_modules = torch.nn.ModuleList(transformer.attn_processors.values())
project_model = MLPProjModel(
    cross_attention_dim=transformer.config.joint_attention_dim, # 4096
    id_embeddings_dim=1152, 
    num_tokens=num_tokens,
).to(transformer.device, dtype=torch.bfloat16)
flux_ipa = IPAdapter_FLUX(adapter_modules, project_model)

flux_ipa.load_state_dict(loaded, strict=True)
flux_ipa = flux_ipa.to(device, dtype=torch.bfloat16)










# 推理




prompt_list = [
    "A figure stands in a misty landscape, wearing a mask with antlers and dark, embellished attire, exuding mystery and otherworldlines",
    #"woman with red hair, playing chess at the park, bomb going off in the background",
    "a woman holding a coffee cup, in a beanie, sitting at a cafe",
    "A Side View of a man lost in deep thought, with the warm glow of the setting sun casting dramatic shadows on their face.",
    #"First-person view, in front of a waterfall",
    #"A Side View of a majestic horse galloping across a field, showcasing its powerful muscles and elegant movement.",
    #"A brave young hero standing on a cliff overlooking a vast ocean, with a determined expression and the wind blowing through their hair. Studio Ghibli animation style. Blend of realism and fantasy with meticulous attention to detail. Vibrant color palette featuring soft, natural tones. Expressive character designs with emotive faces and relatable personalities. Intricate, hand-drawn backgrounds that create immersive worlds. Emphasis on nature as a central theme, often serving as a source of magic and wonder. Whimsical and imaginative storytelling that explores the human experience. Seamless integration of 2D and 3D animation techniques.",
]



folder_path = '/liushanyuan-sh-ceph/project/sub_project/lujunda/flux_ipa/project/project-5/assets/images'


for p in tqdm(prompt_list):
    prompt = p
    # Iterate through files in the folder
    for filename in tqdm(os.listdir(folder_path)):
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            # Construct full file path
            file_path = os.path.join(folder_path, filename)
            
            try:
                # Open the image


                image_name = file_path.split("/")[-1]
                image = Image.open(file_path).convert("RGB")
                image = resize_img(image)

                scale=0.5

                seed=42
                guidance_scale=3.5
                num_inference_steps=24


                if isinstance(image, Image.Image):
                    pil_image = [image]
                clip_image = clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
                clip_image_embeds = image_encoder(clip_image.to(image_encoder.device, dtype=image_encoder.dtype)).pooler_output
                clip_image_embeds = clip_image_embeds.to(dtype=torch.bfloat16)
                project_embeds = clip_image_embeds

                project_embeds=flux_ipa.project_model(project_embeds)
                # Predict the noise residual

                h = project_embeds.shape[1]
                w = project_embeds.shape[1]
                cond_ids = torch.zeros(h // 2, w // 2, 3)
                cond_ids[..., 1] = cond_ids[..., 1] + torch.arange(h // 2)[:, None]
                cond_ids[..., 2] = cond_ids[..., 2] + torch.arange(w // 2)[None, :]
                cond_ids = repeat(cond_ids, "h w c -> b (h w) c", b=project_embeds.shape[0])

                width=960
                height=1280
                width=256
                height=256
                generator = torch.Generator(device).manual_seed(seed)
                images = pipe(
                        prompt=prompt,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        width=width, height=height,
                        image_emb=project_embeds, bdm_ids=cond_ids,
                    ).images

                output_path = f"./result/{ipadapter_path.split('/')[-2]}"
                os.makedirs(output_path, exist_ok=True)
                ind = len(os.listdir(output_path))
                images[0].save(f"{output_path}/{ind}_{prompt[:30].replace(' ','_')}_{scale}__width_{width}_{image_name}")


                
            except Exception as e:
                print(f"Error processing {filename}: {e}")


