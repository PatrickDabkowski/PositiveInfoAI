import torch
from . import data
from tqdm import tqdm
from . import loadmodels
    
class FineTune:

    def __init__(self, device="cpu"):
        
        self.device = torch.device(device)
        print('device: ', self.device)
   
        self.train, self.valid, self.test = data.make_dataloaders(1)

        self.tokenizer, self.model = loadmodels.load_bart(max_length=50)
        self.tokenizer = self.tokenizer
        self.model = self.model.to(self.device)
        
        # froze encoder
        for param in self.model.model.encoder.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam([
                {'params': self.model.model.decoder.parameters(), 'lr': 0.00002}])
        
        self.positiveness_loss = loadmodels.load_positive_classifier(device)

    def epoch(self):
        self.model.model.encoder.eval()
        self.model.model.decoder.train()
  
        progress_bar = tqdm(self.train, desc=f"Train", unit="batch")
    
        for i, x in enumerate(self.train):
            
            # original sentence
            tokens = self.tokenizer(x, return_tensors="pt", padding=True).to(self.device)
            # model forward
            answer = self.model.generate(input_ids=tokens["input_ids"])
            reconstruction = self.model(input_ids=tokens["input_ids"], labels=tokens["input_ids"])
            
            # decoding sentence to check positiveness
            sentence = self.tokenizer.decode(answer[0], skip_special_tokens=True)

            # list comprehension to extract positiveness score, that should be negative if entence is not positive
            positiveness = sum([-sample['score'] if sample['label'] == 'NEGATIVE' else sample['score'] for sample in self.positiveness_loss(sentence)])
            # minization of reconstruction error, maximization of positiveness
            loss = 0.7 * reconstruction.loss - 0.3 * positiveness
            
            # zero value of the gradient, prevents accumulation of gradients from different iterations
            self.optimizer.zero_grad()
            # computes the gradient of current tensor
            loss.backward() 
            # performs a single optimization step, update weights
            self.optimizer.step()
            
            # update progressbar
            progress_bar.set_postfix({"Loss": loss.item()})
            progress_bar.update()
             
            if i % 10 and i != 0:
                torch.save(self.model.state_dict(), 'models/positive_bart.pt')
                
            torch.save(self.model.state_dict(), 'models/positive_bart.pt')
    
    def valid_test(self, test=True):
        
        total_loss = 0
        
        if test:
            data = self.test
        else:
            data = self.valid
        
        self.model.model.encoder.eval()
        self.model.model.decoder.eval()
        
        desc = f"Test" if test else f"Validation"
        progress_bar = tqdm(data, desc=desc, unit="batch")
    
        for i, x in enumerate(data):
            
            # original sentence
            tokens = self.tokenizer(x, return_tensors="pt", padding=True).to(self.device)
            # model forward
            answer = self.model.generate(input_ids=tokens["input_ids"])
            reconstruction = self.model(input_ids=tokens["input_ids"], labels=tokens["input_ids"])
            
            # decoding sentence to check positiveness
            sentence = self.tokenizer.decode(answer[0], skip_special_tokens=True)

            # list comprehension to extract positiveness score, that should be negative if entence is not positive
            positiveness = sum([-sample['score'] if sample['label'] == 'NEGATIVE' else sample['score'] for sample in self.positiveness_loss(sentence)])
            # minization of reconstruction error, maximization of positiveness
            loss = 0.7 * reconstruction.loss - 0.3 * positiveness
            total_loss += loss.item()
            
            progress_bar.set_postfix({"Loss": loss.item(), "Total Loss": total_loss})
            progress_bar.update()
            
        return total_loss
        
    def tune(self, epochs, early_stopping_limit: int = None):
        
        if early_stopping_limit:
            early_stopping_count = 0
            previous_loss = torch.inf
        
        # perfromes model fine-tunning
        
        for epoch in range(epochs):
            # training 
            #self.epoch()
            # validation
            total_loss = self.valid_test(False)
            print(f"Epoch {epoch}, loss: {total_loss}")
            
            if early_stopping_limit:
                if total_loss > previous_loss:
                    early_stopping_count += 1
                    if early_stopping_count >= early_stopping_limit:
                        break
                else:
                    early_stopping_count = 0
                previous_loss = total_loss
                    
        test_loss = self.valid_test()
        print(f"Traning end after {epoch}s with test loss: {test_loss}")
            
if __name__ == "__main__":
    print(f"file: {__name__}")       