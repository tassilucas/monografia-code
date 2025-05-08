import pandas as pd
from google import genai
from pydantic import BaseModel
import json
import time
import csv

class News(BaseModel):
    title: str
    corpus: str

def remove_duplicates(filename):
    df = pd.read_csv(filename, names=["idx", "title", "corpus"])
    df = df.drop_duplicates(subset=["idx"])
    df.to_csv('deduplicated_generations.csv', index=False)

def write_to_file(idx, title, corpus):
    with open('generated.csv', 'a+') as csvfile:
        content_writer = csv.writer(csvfile, delimiter=',')
        content_writer.writerow([idx, title, corpus])

def truncate_text(text):
    try:
        if len(text) > 500:
            tokens = text.split(' ')
            ts = tokens[:500]
            return ' '.join(ts)
    except:
        print("ERROR: ", text)

def generateContentFromGemini(iterator):
    global checkpoint
    for idx, row in iterator:
        prompt = f"""
            Você esta criando notícias.
        
            Considerando o titulo de notícia:
        
            {row['title']}
        
            E o seguinte trecho que fornece contexto para criação da notícia:
        
            {row['text']}
        
            Realize a criação de uma notícia inspirada no titulo e com auxilio
            do trecho fornecidos, de maneira clara, que tenha até 300 palavras.
        
            Não insira caracteres especiais como '\n', '\t' no texto da sua resposta.
            Ela deve ser em texto corrido.
        
            Use esse schema JSON:
        
            News = {{'title': str, 'corpus': str}}
        """
    
        max_retries = 3
        retries = 0
        success = False
        while not success and retries < max_retries:
            try:
                response = client.models.generate_content(
                        model="gemini-2.0-flash-lite",
                        config={'response_mime_type': 'application/json', 'response_schema': News},
                        contents=prompt
                        )
                success = True
                parsed_content = json.loads(response.text.strip())
                print(f"""({idx}) Created news: {parsed_content['title']}
                    ======================================= {checkpoint}""")
                checkpoint = checkpoint + 1
                time.sleep(0.3) # give time to the model
                write_to_file(idx, parsed_content['title'], parsed_content['corpus'])
            except Exception as e:
                print(e)
                print("Retrying after 5 seconds...")
                time.sleep(5)
                retries += 1

remove_duplicates("generated.csv")

# gemini client
client = genai.Client(api_key="AIzaSyCa_-ulJVoRg6XscSa8xjP_joFqfmtOz7o")

# env variables
checkpoint = 0

# data
print("Reading data")
articles = pd.read_csv("articles.csv", delimiter=',')

articles = articles[~articles['text'].isnull()]
articles = articles[(articles['title'].str.len() > 40) & (articles['text'].str.find('.') > 150)]
articles['title'] = articles['title'].apply(lambda x : x.replace("\n", " "))
articles['text'] = articles['text'].apply(lambda x : x.replace("\t", " "))
articles['title'] = articles['title'].apply(lambda x : x.strip())
articles['text'] = articles['text'].apply(lambda x : x.strip())

news_size = 1000
arc = articles.sample(n=news_size)
arc['text'] = arc['text'].apply(truncate_text)

print("Starting generation...")
while True:
    if checkpoint > 1000:
        print("All rows retrieved, ready to export...")
        break

    resp = generateContentFromGemini(arc[checkpoint:].iterrows())
    print("Exiting")
