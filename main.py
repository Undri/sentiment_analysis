# sentiment analysis
import nltk
from nltk.corpus import stopwords
import pymorphy2

morph = pymorphy2.MorphAnalyzer()


# чистка твита, освобождение от лишних слов, знаков, чисел
# лемматизация
def clean(twit):
    # дробление текста на слова с помощью токенайзера по словам из nltk
    words = nltk.word_tokenize(twit)

    # удаление нейтраньлых слов используя stopwords.words("Russian") set
    # и сет значков
    redundant = (',', '.', '#', '...', '!', '?', ':', '…', '(', ')', '``',
                 '«', '»', '@', "''", '-', '+')
    stop_words = set(stopwords.words("Russian")).union(redundant)

    # не берем во внимание дату[0] и время [1]
    filtered_words = [word for word in words[2:] if word not in stop_words]
    # print(len(words) - len(filtered_words))

    # лемматизация
    words = [morph.parse(word)[0].normal_form for word in filtered_words]
    return words


def twit_length(twits):
    twit_lengths = list()
    for twit in twits:
        twit_lengths.append(len(twit))
    twit_lengths.sort()


def main():
    # чтение из файла
    source_file = open('test.txt', 'r', encoding='utf8')
    output_file = open('out.txt', 'w', encoding='utf8')

    src = source_file.readlines()
    # убираем пустые строки
    src = [line for line in src if line != '\n']
    # чистим каждый твит
    lines = [clean(line) for line in src]
    twit_length(lines)

    for word in lines:
        output_file.writelines(word)

    source_file.close()
    output_file.close()


if __name__ == '__main__':
    main()
