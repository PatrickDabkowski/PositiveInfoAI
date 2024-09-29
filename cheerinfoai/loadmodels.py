import torch
from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration

def load_bart(is_fast=True, is_quant=False, max_length=None, truncation=False):
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', max_length=max_length, truncation=truncation, is_fast=True, clean_up_tokenization_spaces=False)
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    
    if is_quant:
        model = torch.ao.quantization.quantize_dynamic(model, dtype=torch.qint8) 

    return tokenizer, model

def load_positive_classifier(device):
    
    return pipeline("text-classification", device=device)


if __name__ == "__main__":
    print(f"file: {__name__}")