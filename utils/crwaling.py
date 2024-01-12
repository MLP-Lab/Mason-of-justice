import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup 
import time
from tqdm.auto import tqdm
import json

mapping_alphabets = {'a': 167,
 'b': 110,
 'c': 227,
 'd': 121,
 'e': 94,
 'f': 99,
 'g': 46,
 'h': 44,
 'i': 60,
 'j': 21,
 'k': 9,
 'l': 71,
 'm': 100,
 'n': 59,
 'o': 55,
 'p': 174,
 'q': 18,
 'r': 99,
 's': 109,
 't': 92,
 'u': 32,
 'v': 33,
 'w': 36,
 'x': 1,
 'y': 3,
 'z': 3}

data_list = []

URL = "https://thelawdictionary.org/letter/"

cnt = 0

for alphabet, num in mapping_alphabets.items():

    URL = "https://thelawdictionary.org/letter/"
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