from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets

class CombinedDataset(Dataset):
    def __init__(self, set: str):
        """
        Initializaties new iterable dataset object based on arxiv-summarization and wikitext datasets 
        from HuggingFace repository
        
        Args:
            set (str): one of 'train', 'validation', 'test' """
            
        self.set = set
        # load and standarize columns' names
        self.arxiv = load_dataset("ccdv/arxiv-summarization", "section").rename_column("abstract", "text").remove_columns("article")
        self.wp = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
        self.arxiv.set_format("torch")
        self.wp.set_format("torch")
        self.arxiv = self.arxiv[self.set]
        self.wp = self.wp[self.set]

        self.dataset = concatenate_datasets([self.arxiv, self.wp])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
    
        return self.dataset[idx]['text']


def make_dataloaders(batch_size: int):
    '''Creates PyTorch DataLoaders from iterable Datasets, here CombinedDatasets objects'''
    
    train =  DataLoader(CombinedDataset('train'), batch_size=batch_size, shuffle=True)
    valid =  DataLoader(CombinedDataset('validation'), batch_size=batch_size, shuffle=True)
    test =  DataLoader(CombinedDataset('test'), batch_size=batch_size, shuffle=True)
    
    print(f'train set: {len(train)}, valid set: {len(valid)}, test set: {len(test)}')
    
    return train, valid, test