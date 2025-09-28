import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from nltk import FreqDist, pos_tag
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re


spell = SpellChecker()
en_stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()

def spell_check(text_list):
    result = []
    for word in text_list:
        correct_word = spell.correction(word)
        if correct_word is None:   # 如果 spellchecker 找不到，保留原字
            correct_word = word
        result.append(correct_word)
    return result

def remove_stopwords(text_list):
    return [token for token in text_list if token not in en_stopwords]

def remove_punct(text_list):
    tokenizer = RegexpTokenizer(r"\w+")
    text_list = [str(token) for token in text_list if token is not None]
    return tokenizer.tokenize(' '.join(text_list))

def lemmatization(text_list):
    result = []
    for token, tag in pos_tag(text_list):
        pos = tag[0].lower()
        if pos not in ['a', 'r', 'n', 'v']:
            pos = 'n'
        result.append(lemmatizer.lemmatize(token, pos))
    return result

def stemming(text_list):
    return [porter.stem(word) for word in text_list]

def remove_tag(text_str):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text_str)

def remove_urls(text_str):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text_str)

def frequent_words(df_column):
    all_words = []
    for tokens in df_column:
        all_words.extend(tokens)
    fdist = FreqDist(all_words)
    return fdist.most_common(10)

if __name__ == "__main__":
    df = pd.read_csv("all_annotated.tsv", sep="\t")
    df_text = df[['Tweet']].copy()
    
    # 小寫化 & 去空格
    df_text['Tweet'] = df_text['Tweet'].str.lower().str.split().str.join(" ")

    # 下載 nltk 模型
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    # 分詞
    df_text['Tweet'] = df_text['Tweet'].apply(word_tokenize)

    # 拼字修正
    df_text['Tweet'] = df_text['Tweet'].apply(spell_check)

    # 去停用詞
    df_text['Tweet'] = df_text['Tweet'].apply(remove_stopwords)

    # 去標點
    df_text['Tweet'] = df_text['Tweet'].apply(remove_punct)

    # 顯示前 10 個最常用詞
    freq10 = frequent_words(df_text['Tweet'])
    print("Top 10 frequent words:", freq10)

    # 詞型還原
    df_text['Tweet'] = df_text['Tweet'].apply(lemmatization)

    # 詞幹化
    df_text['Tweet'] = df_text['Tweet'].apply(stemming)

    # list 轉回字串
    df_text['Cleaned_Tweet'] = df_text['Tweet'].apply(lambda x: ' '.join(x))

    # 移除 HTML 標籤和 URL
    df_text['Cleaned_Tweet'] = df_text['Cleaned_Tweet'].apply(remove_tag)
    df_text['Cleaned_Tweet'] = df_text['Cleaned_Tweet'].apply(remove_urls)

    df_text.to_csv("all_annotated_cleaned.csv", index=False, encoding='utf-8-sig')
    print("已成功匯出處理過後的檔案: all_annotated_cleaned.csv")
