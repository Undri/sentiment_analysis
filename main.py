# sentiment analysis
import nltk
from nltk.corpus import stopwords
import pymorphy2
import string

morph = pymorphy2.MorphAnalyzer()
twit_lengths = list()
twit_len_stat = dict()
word_frequencies = dict()


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
    # исключаем личшние слова и знаки
    filtered_words = [word for word in words if word not in stop_words and word not in string.printable]

    # удаление ссылок (ссылка - всегда последний элемент в твите)

    for letter in filtered_words[-1]:
        if letter in string.printable:
            del filtered_words[-1]
            break

    # print(len(words) - len(filtered_words))

    # лемматизация
    words = [morph.parse(word)[0].normal_form for word in filtered_words]
    return words


def twit_length(twits):
    for twit in twits:
        twit_lengths.append(len(twit))
    for l in sorted(twit_lengths):
        if l in twit_len_stat:
            twit_len_stat[l] += 1
        else:
            twit_len_stat[l] = 1
    output_file = open('twits_length.txt', 'w', encoding='utf8')
    output_file.writelines("Twit length: frequency\n")
    for key, value in twit_len_stat.items():
        output_file.writelines(str(key) + ": " + str(value) + "\n")


def word_frequency(twits):
    for twit in twits:
        for word in twit:
            if word in word_frequencies:
                word_frequencies[word] += 1
            else:
                word_frequencies[word] = 1
    output_file = open('frequency.txt', 'w', encoding='utf8')
    output_file.writelines('Word: frequency\n')
    for key, value in word_frequencies.items():
        output_file.writelines(str(key) + ": " + str(value) + "\n")


def main():
    # чтение из файла
    source_file = open('data.txt', 'r', encoding='utf8')
    output_file = open('out.txt', 'w', encoding='utf8')

    src = source_file.readlines()
    # убираем пустые строки
    src = [line for line in src if line != '\n']
    # чистим каждый твит
    # lines - массив массивов слов
    twits = [clean(line) for line in src]
    twit_length(twits)
    word_frequency(twits)
    for line in twits:
        for word in line:
            output_file.writelines(word)
            output_file.writelines(" ")
        output_file.writelines("\n")

    source_file.close()
    output_file.close()


if __name__ == '__main__':
    main()
