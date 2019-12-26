# sentiment analysis
import nltk
from nltk.corpus import stopwords
import pymorphy2
import string
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

morph = pymorphy2.MorphAnalyzer()
raw_twits = list()          # лист необработанных твитов
twits = list()              # лист твитов
twit_lengths = list()
twit_len_stat = dict()      # словарь типа (длина твита - частота)
word_frequencies = dict()   # словарь типа (слово - твиты с этим словом)
word_sentiment = dict()     # словарь типа (слово - оценка)
twit_sentiment1 = list()    # словарь типа (твит - оценка) classification 1
twit_sentiment2 = list()
estimations = dict()
pos_adjectives = dict()
neg_adjectives = dict()


# чистка твита (одного), освобождение от лишних слов, знаков, чисел
# но оставляю дату и время
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
    # дату и время добавляем отдельно
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
    # слова отсортированны по частоте появления
    for key, value in reversed(sorted(word_frequencies.items(), key=lambda x: x[1])):
        output_file.writelines(str(key) + " - " + str(value) + " - " + str(round(value / length * 100, 2)) + "%\n")
    output_file.close()


def set_sentiment():
    output_file = open('estimations.txt', 'w')
    for key, value in word_frequencies.items():
        print(key + ": ")
        n = int(input())
        word_sentiment[key] = n
        output_file.writelines(str(key) + " " + str(n) + '\n')
    output_file.close()


def classification():
    file = open('estimations.txt')
    src = file.readlines()
    file.close()
    # проходимся по оценкам и разбираем обратно в словарь для дальнейшего использования
    for raw_line in src:
        estimations[raw_line.split(' ')[0]] = raw_line.split(' ')[1].replace('\n', '')
    i = 0

    # --------- FIRST --------------
    # находим оценку настроения твита, складывая оценки слов, из которых состоит твит
    for twit in twits:
        for word in twit:
            i += int(estimations[word])
        twit_sentiment1.append(i)
        i = 0
    # задать пороги, для определения good/bad/neutral твитов
    up = 1
    bot = -1
    # распределяем, согласно заданным порогам
    for_classification = {'good': 0, 'neutral': 0, 'bad': 0}
    for num in twit_sentiment1:
        if num >= up:
            for_classification['good'] += 1
        if bot < num < up:
            for_classification['neutral'] += 1
        if num <= bot:
            for_classification['bad'] += 1

    # не было в задании, но вывел для наглядности
    output_file = open('classification1.txt', 'w')
    output_file.writelines('Twit = sentiment estimation, according to good/bad/neutral words\n\n')
    i = 0
    for raw_twit in raw_twits:
        output_file.writelines(str(raw_twit) + ' : ' + str(twit_sentiment1[i]) + '\n')
        i += 1
    output_file.close()

    # добавляем полученные результаты классификации в файл
    output_file = open('classifications.txt', 'w')
    length = len(raw_twits)
    output_file.writelines('1) Sum of estimations\n')

    good_value = for_classification['good']
    neutral_value = for_classification['neutral']
    bad_value = for_classification['bad']

    output_file.write("Good - " + str(good_value) + " - " + str(round(good_value / length * 100, 2)) + '%\n')
    output_file.write("Neutral - " + str(neutral_value) + " - " + str(round(neutral_value / length * 100, 2)) + '%\n')
    output_file.write("Bad - " + str(bad_value) + " - " + str(round(bad_value / length * 100, 2)) + '%\n\n')

    # barplot
    good = 'good - {}%'.format(round(good_value / length * 100, 2))
    neutral = 'neutral - {}%'.format(round(neutral_value / length * 100, 2))
    bad = 'bad - {}%'.format(round(bad_value / length * 100, 2))
    names = good, neutral, bad
    size_of_groups = [for_classification['good'], for_classification['neutral'], for_classification['bad']]
    plt.pie(size_of_groups, labels=names, colors=['red', 'green', 'blue'])
    # add a circle at the center
    my_circle = plt.Circle((0, 0), 0.7, color='white')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.show()

    # --------- SECOND --------------
    good = 0
    neutral = 0
    bad = 0
    for twit in twits:
        for word in twit:
            if estimations[word] == '1':
                good += 1
            if estimations[word] == '0':
                neutral += 1
            if estimations[word] == '-1':
                bad += 1
        result = max(good, bad, neutral)
        if result == good:
            twit_sentiment2.append('good')
        elif result == neutral:
            twit_sentiment2.append('neutral')
        elif result == bad:
            twit_sentiment2.append('bad')
        good = 0
        bad = 0
        neutral = 0

    # now put results in dict
    for_classification = {'good': 0, 'neutral': 0, 'bad': 0}
    for word in twit_sentiment2:
        for_classification[word] += 1

    # добавляем полученные результаты классификации в файл
    output_file = open('classifications.txt', 'a')
    length = len(raw_twits)
    output_file.writelines('2) Second classification\n')

    good_value = for_classification['good']
    neutral_value = for_classification['neutral']
    bad_value = for_classification['bad']

    output_file.write("Good - " + str(good_value) + " - " + str(round(good_value / length * 100, 2)) + '%\n')
    output_file.write("Neutral - " + str(neutral_value) + " - " + str(round(neutral_value / length * 100, 2)) + '%\n')
    output_file.write("Bad - " + str(bad_value) + " - " + str(round(bad_value / length * 100, 2)) + '%\n\n')

    output_file.close()

    # barplot
    good = 'good - {}%'.format(round(good_value / length * 100, 2))
    neutral = 'neutral - {}%'.format(round(neutral_value / length * 100, 2))
    bad = 'bad - {}%'.format(round(bad_value / length * 100, 2))
    names = good, neutral, bad
    size_of_groups = [for_classification['good'], for_classification['neutral'], for_classification['bad']]
    plt.pie(size_of_groups, labels=names, colors=['red', 'green', 'blue'])
    # add a circle at the center
    my_circle = plt.Circle((0, 0), 0.7, color='white')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.show()


def cout_adjectives():
    length = len(twits)
    for twit in twits:
        for word in set(twit):
            if morph.parse(word)[0].tag.POS == 'ADJF':
                if estimations[word] == '1':
                    if word in pos_adjectives:
                        pos_adjectives[word] += 1
                    else:
                        pos_adjectives[word] = 1
                if estimations[word] == '-1':
                    if word in neg_adjectives:
                        neg_adjectives[word] += 1
                    else:
                        neg_adjectives[word] = 1
    output_file = open('adjectives.txt', 'w')
    # отбираем первые 5
    pos_adjectives_sorted = reversed(sorted(pos_adjectives.items(), key=lambda x: x[1]))
    neg_adjectives_sorted = reversed(sorted(neg_adjectives.items(), key=lambda x: x[1]))
    i = 0
    for key, value in pos_adjectives_sorted:
        if i != 5:
            output_file.writelines(str(key) + " - " + str(value) + ' - ' + str(round(value / length * 100, 2)) + '%\n')
            i += 1
        else:
            break
    output_file.writelines('\n')
    i = 0
    for key, value in neg_adjectives_sorted:
        if i != 5:
            output_file.writelines(str(key) + " - " + str(value) + ' - ' + str(round(value / length * 100, 2)) + '%\n')
            i += 1
        else:
            break
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
        twits.append(clean(line))   # вызываем клинер

    twit_length()
    word_frequency()

    for twit in twits:
        for word in twit:
            output_file.writelines(word)
            output_file.writelines(" ")
        output_file.writelines("\n")
    # функция set_sentiment() вызывается только 1 раз для создания словаря с оценками
    # для каждого слова!!

    # set_sentiment()
    classification()
    cout_adjectives()
    source_file.close()
    output_file.close()


if __name__ == '__main__':
    main()
