from loadmodels import load_positive_classifier
import requests

def get_popular_articles():
    '''downloads list of currently 20 most viewed Wikipedia articles
    Returns: 
        data (list): list of titles of most viewed Wikipedia articles without 3 first elements (main page etc.)'''
    
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
                return data["query"]["mostviewed"][3:]
        else:
            print("No Data")
    else:
        print("API Connection Faild")
        
def most_positive_title(classifier, articles):
    '''clssifies positivness of all most viewed articles on Wikipedia (or any other list of strs)
    and picks title where positive score is the highest
    
    Args:
        classifier: callable model/classifier that returns dict with keys "lable" that are "POSITIVE" or "NEGATIVE" 
        and 'score' being a floating number between 0 and 1
        articles (list): list of str titles 
        
    Returns: 
        most_positive (str): title with highest positive socre'''
    
    max_score = -1
    most_positive = None
    # generator expression to uncpack article titles and filter positive ones by classifier, and return tuple (title, score)
    for article in ((article["title"], classifier(article["title"])[0]['score']) for article in articles if classifier(article["title"])[0]['label'] == 'POSITIVE'):
        if article[1] > max_score:
            
            max_score = article[1]
            most_positive = article[0]
    
    return most_positive
        
def get_article_extracts(titles):
    '''downloads abstracts of given list of titles or given str title
    
    Args:
        titles: callable model/classifier that returns dict with keys "lable" that are "POSITIVE" or "NEGATIVE" 
        and 'score' being a floating number between 0 and 1
        
    Returns: 
        extracts (dict): dict of abstracts from given titles with key "title" and str text as value'''
    
    if isinstance(titles, list):
        pass
    elif isinstance(titles, str):
        titles = [titles]
        
    endpoint = "https://en.wikipedia.org/w/api.php"
    
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": True, 
        "titles": "|".join(titles)  
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
        return None

        
articles = get_popular_articles()
classifier = load_positive_classifier("mps")

if __name__ == "__main__":
    print(get_article_extracts(most_positive_title(classifier, articles)))
