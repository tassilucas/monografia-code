import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import time
import csv

def write_to_file(idx, title, corpus):
    with open('generated_deepseek.csv', 'a+') as csvfile:
        content_writer = csv.writer(csvfile, delimiter=',')
        content_writer.writerow([idx, title, corpus])

def retrieving_news():
    print("Retrieving news")
    gemini_news = pd.read_csv("../dataset/sample_generated.csv")
    return gemini_news.to_dict('records')

def generateContentFromDeepseek(news):
    global checkpoint
    load_dotenv()
    client = OpenAI(api_key=os.getenv("DEEPSEEK_KEY"),
                    base_url="https://api.deepseek.com")

    for new in news:
        retries = 0
        success = False
        while not success and retries < 3:
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "user",
                         "content": 
                            f"""
                                Você esta criando notícias.
                            
                                Considerando o titulo de notícia:
                            
                                {new['title']}
                            
                                E o seguinte trecho que fornece contexto para criação da notícia:
                            
                                {new['text']}
                            
                                Realize a criação de uma notícia inspirada no titulo e com auxilio
                                do trecho fornecidos, de maneira clara, que tenha até 300 palavras.
                            
                                Não insira caracteres especiais como '\n', '\t' no texto da sua resposta.
                                Ela deve ser em texto corrido.
                            
                                Use esse schema JSON:
                            
                                News = {{'title': str, 'corpus': str}}
                            """
                         },
                    ],
                    stream=False,
                    response_format={"type": "json_object"}
                )

                parsed_content = json.loads(response.choices[0].message.content)
                write_to_file(new['ID'], parsed_content['title'], parsed_content['corpus'])
                checkpoint += 1
                success = True
                print(f"""({new['ID']}) Created news: {parsed_content['title']}
                    ======================================= {checkpoint}""")
            except Exception as e:
                print(e)
                print("Retrying after 3 seconds...")
                time.sleep(3)
                retries += 1


news = retrieving_news()

## env variables
checkpoint = 1691

print("Starting generation...")
while True:
    if checkpoint >= 1700:
        print("All rows retrieved, ready to export...")
        break

    resp = generateContentFromDeepseek(news[checkpoint:])
    print("Exiting")
