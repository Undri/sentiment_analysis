# sentiment analysis
import nltk
from nltk.corpus import stopwords
import pymorphy2
import string
import re

morph = pymorphy2.MorphAnalyzer()
raw_twits = list()          # лист необработанных твитов
twits = list()              # лист твитов
twit_lengths = list()
twit_len_stat = dict()      # словарь типа (длина твита - частота - %)
word_frequencies = dict()   # словарь типа (слово - твиты с этим словом - %)
word_sentiment = dict()     # словарь типа (слово - оценка)
twit_sentiment1 = list()    # словарь типа (твит - оценка) classification 1


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
    # удаляем все слова которые состоят не из русских букв
    r = re.compile("[а-яА-Я]+")
    filtered_words = list()

    # дату и время нужно оставить поэтому [2:]
    filtered_words.append(words[0])
    filtered_words.append(words[1])
    for word in words[2:]:
        if word not in stop_words and word not in string.printable and len(word) > 2 and word in filter(r.match, words):
            filtered_words.append(word)

    # print(len(words) - len(filtered_words))
    # лемматизация
    words = [morph.parse(word)[0].normal_form for word in filtered_words]
    return words


def twit_length():
    length = len(twits)
    for twit in twits:
        twit_lengths.append(len(twit))
    for l in sorted(twit_lengths):
        if l in twit_len_stat:
            twit_len_stat[l] += 1
        else:
            twit_len_stat[l] = 1
    output_file = open('twits_length.txt', 'w', encoding='utf8')
    output_file.writelines("Twit length - frequency - %\n")
    for key, value in twit_len_stat.items():
        output_file.writelines(str(key) + " - " + str(value) + " - " + str(round(value / length * 100, 2)) + "%\n")
    output_file.close()


def word_frequency():
    # how frequent is every word in all twits
    length = len(twits)
    for twit in twits:
        # make set from twit to delete repeated words
        for word in set(twit):
            if word in word_frequencies:
                word_frequencies[word] += 1
            else:
                word_frequencies[word] = 1

    output_file = open('frequency.txt', 'w', encoding='utf8')
    output_file.writelines('Word: frequency\n')
    for key, value in reversed(sorted(word_frequencies.items(), key=lambda x: x[1])):
        output_file.writelines(str(key) + " - " + str(value) + " - " + str(round(value / length * 100, 2)) + "%\n")
    output_file.close()


def set_sentiment():
    output_file = open('estimations.txt', 'w')
    n = 0
    for key, value in word_frequencies.items():
        if value >= 2:
            print(key + ": ")
            n = int(input())
            word_sentiment[key] = n
        else:
            n = 0
            word_sentiment[key] = n
        output_file.writelines(str(key) + " " + str(n) + '\n')
    output_file.close()


def classification():
    file = open('estimations.txt')
    src = file.readlines()
    line = dict()
    file.close()
    # проходимся по оценкам и разбираем обратно в словарь для дальнейшего использования
    for raw_line in src:
        line[raw_line.split(' ')[0]] = raw_line.split(' ')[1]
    i = 0

    # --------- FIRST --------------
    # находим оценку настроения твита, складывая оценки слов, из которых состоит твит
    for twit in twits:
        for word in twit:
            i += int(line[word])
        twit_sentiment1.append(i)
        i = 0
    # задать пороги, для определения good/bad/neutral твитов
    up = 1
    bot = -1
    # распределяем, согласно заданным порогам
    for_classification1 = {'good': 0, 'neutral': 0, 'bad': 0}
    for num in twit_sentiment1:
        if num >= up:
            for_classification1['good'] += 1
        if bot < num < up:
            for_classification1['neutral'] += 1
        if num <= bot:
            for_classification1['bad'] += 1

    # не было в задании, но вывел для наглядности
    output_file = open('classification1.txt', 'w')
    output_file.writelines('Twit = sentiment estimation, according to good/bad/neutral words\n\n')
    i = 0
    for raw_twit in raw_twits:
        output_file.writelines(str(raw_twit) + ' : ' + str(twit_sentiment1[i]) + '\n')
        i += 1
    output_file.close()



    # добавляем полученные результаты классификации в файл
    output_file = open('classifications.txt', 'a')
    length = len(raw_twits)
    output_file.writelines('Sum of estimations\n')

    value = for_classification1['good']
    output_file.write("Good - " + str(value) + " - " + str(round(value / length * 100, 2)) + '%\n')

    value = for_classification1['neutral']
    output_file.write("Neutral - " + str(value) + " - " + str(round(value / length * 100, 2)) + '%\n')

    value = for_classification1['bad']
    output_file.write("Bad - " + str(value) + " - " + str(round(value / length * 100, 2)) + '%\n\n')

    output_file.close()


def main():
    # чтение из файла
    source_file = open('test.txt', 'r', encoding='utf8')
    output_file = open('out.txt', 'w', encoding='utf8')
    src = source_file.readlines()
    # убираем пустые строки
    src = [line for line in src if line != '\n']
    for line in src:
        if line != '\n':
            raw_twits.append(line)
    # чистим каждый твит
    # lines - массив массивов слов
    for line in src:
        twits.append(clean(line))

    twit_length()
    word_frequency()
    for twit in twits:
        for word in twit:
            output_file.writelines(word)
            output_file.writelines(" ")
        output_file.writelines("\n")
    # функция set_sntiment() вызывается только 1 раз для создания словаря с оценками
    # для каждого слова!!

    # set_sentiment()
    classification()
    source_file.close()
    output_file.close()


if __name__ == '__main__':
    main()
