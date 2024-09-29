import torch
from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration

def load_bart(is_fast=True, is_quant=False):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', is_fast=is_fast)
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    
    if is_quant:
        model = torch.ao.quantization.quantize_dynamic(model, dtype=torch.qint8) 

    return tokenizer, model

def load_positive_classifier(device):
    
    return pipeline("text-classification", device=device)


if __name__ == "__main__":
    print(f"file: {__name__}")