import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup 
import time
from tqdm.auto import tqdm
import json

data_list = []


cnt = 0

for alphabet, num in mapping_alphabets.items():

    URL = "url_url_url"
    URL += alphabet

    for i in tqdm(range(1, num + 1)):
        cnt += 1
        page_url = URL + '/page/' + str(i)

        response = requests.get(page_url)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        word_list = soup.find_all('h2', 'elementor-post__title')
        meaning_list = soup.find_all('div', 'elementor-post__excerpt')

        
        for word, meaning in zip(word_list, meaning_list):
            temp_dict = {
                'word': word.text.strip(),
                'meaning': meaning.text.strip()
            }
            data_list.append(temp_dict)

        if cnt % 60 == 0:
            print("Sleep 90seconds. Count:" + str(cnt)
                +",  Local Time:"+ time.strftime('%Y-%m-%d', time.localtime(time.time()))
                +" "+ time.strftime('%X', time.localtime(time.time()))
                +",  Data Length:"+ str(len(data_list))
                +",  Data Ratio:"+ str(len(data_list) / cnt))
            # time.sleep(90)

    with open(f'./law_dictionary_{alphabet}.json', 'w', encoding='utf-8') as file:
        json.dump(data_list, file, indent='\t')