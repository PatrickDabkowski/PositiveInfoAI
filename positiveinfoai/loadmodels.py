from transformers import BartTokenizer, BartForConditionalGeneration
from diffusers import StableDiffusionPipeline
from transformers import pipeline
import torch

def load_bart(is_fast=True, is_quant=False, max_length=None, truncation=False):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', max_length=max_length, truncation=truncation, is_fast=True, clean_up_tokenization_spaces=False, padding='longest')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    
    if is_quant:
        model = torch.ao.quantization.quantize_dynamic(model, dtype=torch.qint8) 

    return tokenizer, model

def load_positive_bart(path: str, device: str, is_fast=True, is_quant=False, max_length=None, truncation=False):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', max_length=max_length, truncation=truncation, is_fast=True, clean_up_tokenization_spaces=False, padding='longest')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    
    if is_quant:
        model = torch.ao.quantization.quantize_dynamic(model, dtype=torch.qint8) 

    return tokenizer, model

def load_stablediffusion(device: str = 'cpu', is_quant: bool = False):
    # torch.ao is not compatible with MPS device (30.09.2024)
    
    if is_quant:
        pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype= torch.float32, device=device)
        print(pipeline)
        
        pipeline.unet = torch.ao.quantization.quantize_dynamic(pipeline.unet, dtype=torch.qint8) 
        pipeline.text_encoder = torch.ao.quantization.quantize_dynamic(pipeline.text_encoder, dtype=torch.qint8) 
        pipeline.vae = torch.ao.quantization.quantize_dynamic(pipeline.vae, dtype=torch.qint8) 
        
        return pipeline
    
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", device=device)
        
    return pipeline.to(device)

def load_positive_classifier(device):
    
    return pipeline("text-classification", device=device)


if __name__ == "__main__":
    print(f"file: {__name__}")    