# sentiment analysis
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


def main():
    # чтение из файла
    source_file = open('data.txt', 'r', encoding='utf8')
    output_file = open('out.txt', 'w', encoding='utf8')
    src = source_file.read()

    # дробление текста на слова с помощью токенайзера по словам из nltk
    words = nltk.word_tokenize(src)

    # удаление нейтраньлых слов используя stopwords.words("Russian") set
    # и сет значков
    redundant = (',', '.', '#', '...', '!', '?', ':', '…')
    stop_words = set(stopwords.words("Russian")).union(redundant)

    filtered_words = [word for word in words if word not in stop_words]
    # print(len(words) - len(filtered_words))
    words = filtered_words
    for word in words:
        output_file.writelines(word + "\n")
    source_file.close()
    output_file.close()


if __name__ == '__main__':
    main()
