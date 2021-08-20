import re
import argparse

class preprocess():
    def __init__(self, datadir):
        self.datadir = datadir

    def clean_number_tag(self, text):
        re_pattern = r'\d\. '
        new_text = re.sub(re_pattern, '', text)
        return new_text

    def clean_newline_tab(self, text):
        re_pattern = r'[\n\t]'
        new_text = re.sub(re_pattern, ' ', text)
        return new_text

    def delete_deleted(self, text):
        patterns = '[deleted]'
        new_text = re.sub(patterns, '', text)
        return new_text

    def only_english(self, text):
        re_pattern = r'[^A-Za-z0-9]'
        new_text = re.sub(re_pattern, '', text)
        return new_text

    def cleanText(self, text):
    
        new_text = re.sub('[-=+#/\:^$@*☑⋅——♦ε←\"•███※~&%ㆍ』\\‘|\(\)\[\]\<\>`\…》]', '', text)
    
        return new_text

    def phone_number_filter(self, text):
        re_pattern = r'\d{2,3}[-\.\s]*\d{3,4}[-\.\s]*\d{4}(?!\d)'
        new_text = re.sub(re_pattern, 'tel', text)
        re_pattern = r'\(\d{3}\)\s*\d{4}[-\.\s]??\d{4}'
        new_text = re.sub(re_pattern, 'tel', new_text)
        return new_text
        
        
    def url_filter(self, text):
        re_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),|]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        new_text = re.sub(re_pattern, 'url', text)
        return new_text

    def delete_repeated_sign(self, text):
        new_text =  re.sub("'+", "'", text)
        return new_text

    def username_filter(self, text):
        re_pattern = r"@\w*\b"
        new_text = re.sub(re_pattern, 'username', text) #user? person?
        return new_text


    def preprocess_text(self, data): #data = train['text']
        data = datadir
        data = data.map(url_filter)
        data = data.map(cleanText)
        data = data.map(delete_repeated_sign)
        data = data.map(username_filter)
        return data

def main():
    data = 
    preprocessed_data = preprocess()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run this to preprocess dataset.")
    parser.add_argument("--path_to_data")
    args = parser.parse_args()
    preprocess_text(args) # 데이터 이름 parser에 넣어서..