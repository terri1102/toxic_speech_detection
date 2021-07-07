import re
import demoji
demoji.download_codes()
from demoji import replace_with_desc

def clean_number_tag(text):
    re_pattern = r'\d\. '
    new_text = re.sub(re_pattern, '', text)
    return new_text

def clean_newline_tab(text):
    re_pattern = r'[\n\t]'
    new_text = re.sub(re_pattern, ' ', text)
    return new_text

def delete_deleted(text):
    patterns = '[deleted]'
    new_text = re.sub(patterns, '', text)
    return new_text

def only_english(text):
    re_pattern = r'[^A-Za-z0-9]'
    new_text = re.sub(re_pattern, '', text)
    return new_text

def cleanText(text):
 
    #텍스트에 포함되어 있는 특수 문자 제거 #점이랑 쉼표는 살리기. 문장이 여러 개인 것도 있으니까 점이 있는 게 나을듯?
 
    new_text = re.sub('[-=+#/\:^$@*☑⋅——♦ε←\"•███※~&%ㆍ』\\‘|\(\)\[\]\<\>`\…》]', '', text)
 
    return new_text

def phone_number_filter(text):
    re_pattern = r'\d{2,3}[-\.\s]*\d{3,4}[-\.\s]*\d{4}(?!\d)'
    new_text = re.sub(re_pattern, 'tel', text)
    re_pattern = r'\(\d{3}\)\s*\d{4}[-\.\s]??\d{4}'
    new_text = re.sub(re_pattern, 'tel', new_text)
    return new_text
    
    
def url_filter(text):
    re_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),|]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    new_text = re.sub(re_pattern, 'url', text)
    return new_text

def delete_repeated_sign(text):
    new_text =  re.sub("'+", "'", text)
    return new_text

def username_filter(text):
    re_pattern = r"@\w*\b"
    new_text = re.sub(re_pattern, 'username', text) #user? person?
    return new_text

def convert_emoji(text):
    new_text = replace_with_desc(text, sep="'")
    return new_text


def preprocess_text(data): #data = train['text']
  data = data.map(url_filter)
  data = data.map(cleanText)
  data = data.map(delete_repeated_sign)
  data = data.map(username_filter)
  data = data.map(convert_emoji)
  return data

if __name__ == '__main__':
    demoji.download_codes()
    #preprocess_text(data) # 데이터 이름 parser에 넣어서..