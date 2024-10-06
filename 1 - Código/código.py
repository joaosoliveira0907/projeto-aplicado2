# Importar pacotes necessários
import re
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Certifique-se de ter baixado os pacotes e modelos necessários:
# !pip install nltk spacy
# !python -m spacy download pt_core_news_sm

# Carregar o modelo de linguagem português do spaCy
nlp = spacy.load('pt_core_news_sm')

# Definir as stop words em português (precisa ter baixado o conjunto de stopwords do nltk)
stop_words = set(stopwords.words('portuguese'))

# Função para remover stop words
def remove_stopwords(text):
return ' '.join([word for word in text.split() if word.lower() not in stop_words])

# Função para tokenizar o texto
def tokenize_text(text):
return word_tokenize(text)

# Função para lematizar o texto
def lemmatize_text(text):
doc = nlp(text)
return ' '.join([token.lemma_ for token in doc])

# Função para remover stop words
def remove_stopwords(text):
return ' '.join([word for word in text.split() if word.lower() not in stop_words])


# Função para normalizar (converter para letras minúsculas)
def normalize_text(text):
return text.lower()

# Função geral para aplicar todas as etapas de pré-processamento
def preprocess_text(text):
text = normalize_text(text) # Passo 1: Normalização
text = remove_special_chars(text) # Passo 2: Remover pontuação
text = remove_stopwords(text) # Passo 3: Remover stopwords
text = lemmatize_text(text) # Passo 4: Lematização
tokens = tokenize_text(text) # Passo 5: Tokenização (opcional, se necessário)
return tokens

# Exemplo de aplicação em uma coluna de DataFrame usando pandas
import pandas as pd

# Suponha que você tenha uma base de dados com uma coluna 'text'
data = pd.DataFrame({
'text': ['Este é um exemplo de frase para ser processada.',
'Outro exemplo com palavras que precisam ser limpas!']
})

# Aplicar a função de pré-processamento à coluna 'text'
data['processed_text'] = data['text'].apply(preprocess_text)

# Exibir os textos processados
print(data[['text', 'processed_text']])