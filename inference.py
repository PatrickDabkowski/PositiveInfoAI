import positiveinfoai
import os

class Bot():
    def __init__(self, device_bart='cpu', device_sd='cpu', is_fast=True, is_quant=False):
        if os.path.exists("positiveinfoai/models/positive_bart.pt"):
            print("run positive BART")
            self.tokenizer, self.model = positiveinfoai.load_positive_bart("positiveinfoai/models/positive_bart.pt", 
                                                                           device_bart, is_fast, is_quant)
        else:
            print("run standard BART")
            self.tokenizer, self.model = positiveinfoai.load_bart("positiveinfoai/models/positive_bart.pt", 
                                                                  device_bart, is_fast, is_quant)
            
        self.sd = positiveinfoai.load_stablediffusion(device_sd, is_quant)
        self.wpapi = positiveinfoai.WikipediaAPI()
        
    def wrapp_title(self):
        # uses WikipediaAPI to extract title and abstract of most positive among most pupular Wikipedia artivles
        info = self.wpapi.get_article_extracts()
        # unpack dict
        key = next(iter(info))
        # html -> text
        val = self.wpapi.return_text(info[key])
        
        return key, val
    
    def generate(self, is_new: bool = None):
        
        if is_new == True:  
            self.title, self.abstract = self.wrapp_title()
        else:
            if not hasattr(self, 'title') and not hasattr(self, 'abstract'):
               self.title, self.abstract = self.wrapp_title()  
                
        # text 
        inputs = self.tokenizer(self.abstract, return_tensors="pt")
        outputs = self.model.generate(inputs['input_ids'], max_length=150, num_beams=1, early_stopping=True)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # image
        img = self.sd(self.title)
        
        return text, img
 
if __name__ == "__main__":
    print(f"file: {__name__}")      
    # Apple MPS test (BART doesn't work with MPS well)   
    print(Bot('cpu', 'mps').generate(False))