# sentiment analysis
import nltk
from matplotlib.dates import DateFormatter
from nltk.corpus import stopwords
import pymorphy2
import string
import re
import matplotlib.pyplot as plt
from matplotlib import dates
import numpy as np

morph = pymorphy2.MorphAnalyzer()
raw_twits = list()          # лист необработанных твитов
twits = list()              # лист твитов
twit_lengths = list()       # лист длин твитов
twit_len_stat = dict()      # словарь типа (длина твита - частота)
word_frequencies = dict()   # словарь типа (слово - твиты с этим словом)
word_sentiment = dict()     # словарь типа (слово - оценка)
twit_sentiment1 = list()    # лист оценок classification 1
twit_sentiment2 = list()    # лист оценок classification 2
estimations = dict()        # словарь типа (слово - оценка)
pos_adjectives = dict()     # словарь типа (положительное прилагательное - количество твитов с ним)
neg_adjectives = dict()     # словарь типа (отрицательное прилагательное - количество твитов с ним)


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
    output_file = open('estimations.txt', 'w', encoding='utf8')

    for key, value in word_frequencies.items():
        if word_frequencies[key] >= 0:
            w = morph.parse(key)[0].tag.POS
            while True:
                print(key + ": ")
                n = input()
                if n == '1' or n == '0' or n == '-1':
                    word_sentiment[key] = int(n)
                    output_file.writelines(str(key) + " " + str(n) + '\n')
                    break
    output_file.close()


def classification():
    file = open('estimations.txt', 'r', encoding='utf8')
    src = file.readlines()
    file.close()
    # проходимся по оценкам и разбираем обратно в словарь для дальнейшего использования
    for raw_line in src:
        estimations[raw_line.split(' ')[0]] = raw_line.split(' ')[1].replace('\n', '')
    i = 0

    # --------- FIRST --------------
    # находим оценку настроения твита, складывая оценки слов, из которых состоит твит
    for twit in twits:
        for word in twit[2:]:
            if word in estimations:
                i += int(estimations[str(word)])
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
        for word in twit[2:]:
            if word in estimations:
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
    for_classification2 = {'good': 0, 'neutral': 0, 'bad': 0}
    for word in twit_sentiment2:
        for_classification2[word] += 1

    # добавляем полученные результаты классификации в файл
    output_file = open('classifications.txt', 'a')
    length = len(raw_twits)
    output_file.writelines('2) Second classification\n')

    good_value = for_classification2['good']
    neutral_value = for_classification2['neutral']
    bad_value = for_classification2['bad']

    output_file.write("Good - " + str(good_value) + " - " + str(round(good_value / length * 100, 2)) + '%\n')
    output_file.write("Neutral - " + str(neutral_value) + " - " + str(round(neutral_value / length * 100, 2)) + '%\n')
    output_file.write("Bad - " + str(bad_value) + " - " + str(round(bad_value / length * 100, 2)) + '%\n\n')

    output_file.close()

    # barplot
    good = 'good - {}%'.format(round(good_value / length * 100, 2))
    neutral = 'neutral - {}%'.format(round(neutral_value / length * 100, 2))
    bad = 'bad - {}%'.format(round(bad_value / length * 100, 2))
    names = good, neutral, bad
    size_of_groups = [for_classification2['good'], for_classification2['neutral'], for_classification2['bad']]
    plt.pie(size_of_groups, labels=names, colors=['red', 'green', 'blue'])
    # add a circle at the center
    my_circle = plt.Circle((0, 0), 0.7, color='white')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.show()


def cout_adjectives():
    length = len(twits)
    for twit in twits:
        # делаем сет из слов в твите чтобы считать твиты а не все появления слова
        for word in set(twit[2:]):
            if morph.parse(word)[0].tag.POS == 'ADJF':
                if word in estimations:
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
    pos_adjectives_sorted = dict(reversed(sorted(pos_adjectives.items(), key=lambda x: x[1])))
    neg_adjectives_sorted = dict(reversed(sorted(neg_adjectives.items(), key=lambda x: x[1])))
    i = 0
    output_file.writelines("Top 5 Positive: \n")
    for key, value in pos_adjectives_sorted.items():
        if i != 5:
            output_file.writelines(str(key) + " - " + str(value) + ' - ' + str(round(value / length * 100, 2)) + '%\n')
            i += 1
        else:
            break

    output_file.writelines("\nTop 5 Negative: \n")
    i = 0
    for key, value in neg_adjectives_sorted.items():
        if i != 5:
            output_file.writelines(str(key) + " - " + str(value) + ' - ' + str(round(value / length * 100, 2)) + '%\n')
            i += 1
        else:
            break
    output_file.close()

    figure = plt.figure(figsize=(11, 8))
    first_bar = figure.add_axes([0.555, 0.325, 0.365, 0.352273])
    first_bar.set_title('Top 5 positive')
    second_bar = figure.add_axes([0.125, 0.325, 0.365, 0.352273])
    second_bar.set_title('Top negative')
    labels1 = list()
    pos = list()
    i = 0
    for key, value in pos_adjectives_sorted.items():
        if i != 5:
            labels1.append(key)
            pos.append(value)
            i += 1
        else:
            break
    labels2 = list()
    neg = list()
    i = 0
    for key, value in neg_adjectives_sorted.items():
        if i != 5:
            labels2.append(key)
            neg.append(value)
            i += 1
        else:
            break
    width = 0.35

    first_bar.bar(labels1, pos, width, label='Positive')
    first_bar.set_xticklabels(labels1, fontsize=9)

    second_bar.bar(labels2, neg, width, label='Negative')
    second_bar.set_xticklabels(labels2, fontsize=9)

    plt.show()


def dates_counter():
    from datetime import timedelta, datetime
    start_time = datetime.strptime(twits[-1][0] + ' ' + twits[-1][1], "%Y-%m-%d %H:%M")
    shag = timedelta(minutes=10)
    new_time = start_time
    output_file = open('hours.txt', 'w')
    x, yg, yb, yn, y = list(), list(), list(), list(), list()
    while new_time < datetime.strptime(twits[0][0] + ' ' + twits[0][1], "%Y-%m-%d %H:%M"):
        new_time = new_time + shag
        i = 0
        counter = 0
        new_twit_sent1 = list(reversed(twit_sentiment1))
        good = 0
        bad = 0
        neutral = 0
        for twit in twits[-1::-1]:
            d = datetime.strptime(twit[0] + ' ' + twit[1], "%Y-%m-%d %H:%M")
            if d < new_time:
                if new_twit_sent1[i] > 0:
                    good += 1
                elif new_twit_sent1[i] == 0:
                    neutral += 1
                elif new_twit_sent1[i] < 0:
                    bad += 1
                counter += 1
            i += 1
        stt = str(start_time)
        nt = str(new_time)
        g = str(round(good / counter, 5))
        b = str(round(bad / counter, 5))
        n = str(round(neutral / counter, 5))
        output_file.writelines(stt + ' - ' + nt + ' - ' + str(counter) + ' ' + g + '/' + n + '/' + b + '\n')
        shag += timedelta(minutes=10)
        x.append(new_time)
        y.append(counter)
        yg.append(float(g))
        yn.append(float(n))
        yb.append(float(b))

    fmt = DateFormatter('%H:%M:%S')
    x = np.array(x)
    y = np.array(y)
    yg = np.array(yg)
    yn = np.array(yn)
    yb = np.array(yb)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.set_title('Distribution of tweets classes in time')
    ax1.plot(x, yg, "ro-", label='N_pos')
    ax1.plot(x, yn, "go:", label='N_0')
    ax1.plot(x, yb, "bo--", label='N_neg')
    ax1.grid()
    ax1.legend()
    ax1.set_ylabel('Fraction')

    ax2.bar(x, y, width=0.03, color='black')
    ax2.grid()
    ax2.set_ylabel('Number of tweets')
    ax2.set_xlabel('Time window')

    ax1.xaxis.set_major_formatter(fmt)
    ax2.xaxis.set_major_formatter(fmt)
    fig.autofmt_xdate()

    plt.show()
    output_file.close()


def main():
    # чтение из файла
    source_file = open('data.txt', 'r', encoding='utf8')
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

    dates_counter()


if __name__ == '__main__':
    main()
