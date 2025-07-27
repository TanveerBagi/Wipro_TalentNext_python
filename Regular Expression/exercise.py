#1. Write a program to find if a string has only octal digits. Given: ['789','123','004']
import re

strings = ['789', '123', '004']

for s in strings:
    if re.match(r'^[0-7]+$', s):
        print(f"'{s}' - Valid octal")
    else:
        print(f"'{s}' - Not octal")

#2. Extract the user id, domain name, and suffix from the following email addresses:
    """
    emails = """
    zuck@facebook.com
    sunder33@google.com
    jeff42@amazon.com """
    """

import re

emails = """
zuck@facebook.com
sunder33@google.com
jeff42@amazon.com
"""

pattern = r'([\w\.]+)@([\w]+)\.([\w]+)'
matches = re.findall(pattern, emails)
print(matches)  

#3. Split the following irregular sentence into proper words:
    """ sentence = """A, very    very; irregular_sentence""" 
    """

import re


sentence = """A, very    very; irregular_sentence"""
cleaned = re.sub(r'[^a-zA-Z\s]', ' ', sentence)
words = re.split(r'\s+', cleaned.strip())
print(' '.join(words))  

#4.Clean up the following tweet to retain only the user’s message (remove URLs, mentions, RTs, hashtags, etc.):
   #tweet = '''Good advice! RT @TheNextWeb: What I would do differently if I was learning to code today http://t.co/lbwej0pxOd cc: @garybernhardt #rstats'''

import re

tweet = '''Good advice! RT @TheNextWeb: What I would do differently if I was learning to code today http://t.co/lbwej0pxOd cc: @garybernhardt #rstats'''
clean = re.sub(r'http\S+|@\S+|#\S+|RT|cc:|[:!]', '', tweet)
clean = re.sub(r'\s+', ' ', clean).strip()
print(clean)  

#5.Extract all the text portions between HTML tags from this URL: https://raw.githubusercontent.com/selva86/datasets/master/sample.html
import requests
import re
def extract_text_from_html(url):
    r = requests.get(url)
    html_text = r.text
  
    pattern = r'>([^<]+)<'        
    matches = re.findall(pattern, html_text)
    cleaned_matches = [match.strip() for match in matches if match.strip()]
        
    return cleaned_matches
        
url = "https://raw.githubusercontent.com/selva86/datasets/master/sample.html"
        
extracted_text = extract_text_from_html(url)
print("Extracted text from sample HTML:")
for text in extracted_text:
    print(f"'{text}'")


#6. Given below list of words, identify the words that begin and end with the same character.
  #civic
  #trust
  #widows
  #maximum
  #museums
  #aa
  #as

import re

words = ['civic', 'trust', 'widows', 'maximum', 'museums', 'aa', 'as']

for word in words:
    if re.match(r'^(.).*\1$', word):
        print(f"'{word}' - Same start/end")
    else:
        print(f"'{word}' - Different start/end")




