# sentiment analysis
import nltk


def main():

    source_file = open('data.txt', 'r', encoding='utf8')
    src = source_file.readlines()
    text = 'Hello, Mr. Andrey. How are you doing?'
    print(nltk.sent_tokenize(text))
    for word in nltk.word_tokenize(text):
        print(word)
    lines = list()
    for line in src:
        if line != "\n":
            lines.append(line)
    print(len(lines))
    source_file.close()


if __name__ == '__main__':
    main()
