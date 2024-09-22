from loadmodels import load_positive_classifier
from bs4 import BeautifulSoup
import requests

class WikipediaAPI:
    
    def __init__(self):
        self.classifier = load_positive_classifier("mps")

    def get_popular_articles(self):
        '''downloads list of currently 20 most viewed Wikipedia articles
        self.articles (list): list of titles of most viewed Wikipedia articles without 3 first elements (main page etc.)'''
        
        endpoint = "https://en.wikipedia.org/w/api.php"
        
        params = {
            "action": "query",
            "format": "json",
            "list": "mostviewed",
            "pvimlimit": "20",  
        }

        response = requests.get(endpoint, params=params)
        
        if response.status_code == 200:
    
            data = response.json()
            
            if "mostviewed" in data["query"]:
                
                for article in data["query"]["mostviewed"]:
                    self.articles = data["query"]["mostviewed"][3:]
            else:
                print("No Data")
        else:
            print("API Connection Faild")
            
    def most_positive_title(self):
        '''clssifies positivness of all most viewed articles on Wikipedia (or any other list of strs)
        and picks title where positive score is the highest
        
        Args:
            classifier: callable model/classifier that returns dict with keys "lable" that are "POSITIVE" or "NEGATIVE" 
            and 'score' being a floating number between 0 and 1
            self.articles (list): list of str titles 
            
            self.titles (str): title with highest positive socre'''
        
        if not hasattr(self, 'articles'):
            self.get_popular_articles()
            
        max_score = -1
        most_positive = None
        # generator expression to uncpack article titles and filter positive ones by classifier, and return tuple (title, score)
        for article in ((article["title"], self.classifier(article["title"])[0]['score']) for article in self.articles if self.classifier(article["title"])[0]['label'] == 'POSITIVE'):
            if article[1] > max_score:
                
                max_score = article[1]
                most_positive = article[0]
        
        self.titles = most_positive
            
    def get_article_extracts(self):
        '''downloads abstracts of given list of titles or given str title
        
        Args:
            self.titles: callable model/classifier that returns dict with keys "lable" that are "POSITIVE" or "NEGATIVE" 
            and 'score' being a floating number between 0 and 1
            
        Returns: 
            extracts (dict): dict of abstracts from given titles with key "title" and str text as value'''
        
        if not hasattr(self, 'titles'):
            self.most_positive_title()
            
        if isinstance(self.titles, str):
            self.titles = [self.titles]
            
        endpoint = "https://en.wikipedia.org/w/api.php"
        
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True, 
            "titles": "|".join(self.titles)  
        }
        
        response = requests.get(endpoint, params=params)
        
        if response.status_code == 200:
            data = response.json()
            extracts = {}
            
            for page_id, page_info in data["query"]["pages"].items():
                if "extract" in page_info:
                    extracts[page_info["title"]] = page_info["extract"]
            
            return extracts
        else:
            print("API Connection Failed")
            
    def return_text(self, html_text):

        soup = BeautifulSoup(html_text, 'html.parser')
        text = soup.get_text()
        text = ' '.join(text.split())

        return text

        
if __name__ == "__main__":
    wa = WikipediaAPI()
    print(wa.get_article_extracts())