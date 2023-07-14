from canine.config import WORDS, FREQ
import pandas as pd
import re

# run char_frquency.py first

def get_freqs(word_list, lang):

    word_list = [str(m).lower() for m in word_list]
    special_chars = ["«","´","»", ".", ",", ":", "?", "!", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    word_list = [re.sub('[\.\,:;\!?«»\\+\(\)\/\-0-9\'%&´\t@ ]+', '', m) for m in word_list]

    char_freq = {}
    for w in word_list:
        for c in w:
            if c not in char_freq:
                char_freq[c] = 0
            else:
                char_freq[c] += 1
    print(char_freq)
    total_chars = sum(char_freq.values())
    print(total_chars)
    for c, f in char_freq.items():
        f = float(f/total_chars)
        char_freq[c] = f
    print(char_freq)
    print()
    print()
    f_df = pd.DataFrame.from_dict(char_freq, orient='index')
    f_df['language'] = lang
    #print(f_df)
    return f_df


char_freq_df = pd.DataFrame()

# danish
frequent_words = pd.read_csv("canine/frequency_lists/danish/lemma-10k-2017-ex.txt", sep="\t", header=None, names=["POS", "lemma", "freq"])
freq_words_list = frequent_words["lemma"].tolist()[:FREQ]
print(len(freq_words_list))

danish_freq = get_freqs(freq_words_list, "da")


# faroese
frequent_words = pd.read_csv("canine/frequency_lists/faroese_freq_list.csv")
freq_words_list = frequent_words["Unnamed: 0"].tolist()[:FREQ]
print(len(freq_words_list))

faroese_freq = get_freqs(freq_words_list, "fa")

# finnish
frequent_words = pd.read_csv("canine/frequency_lists/parole_frek_3-utf8.txt", delimiter=" ", header=None, names=["rank", "freq", "word", "perc", "last"])
freq_word_list = frequent_words["word"].tolist()[:FREQ]

finnish_freq = get_freqs(freq_words_list, "fi")

# icelandic
frequent_words = pd.read_csv("canine/frequency_lists/MIMO_freq/frequency_list.tsv", delimiter="\t")
freq_word_list = frequent_words["word"].tolist()[:FREQ]

icelandic_freq = get_freqs(freq_words_list, "is")

# norwegian
frequent_words = pd.read_excel("canine/frequency_lists/Norwegian-Kelly.xls")
freq_word_list = frequent_words["Norwegian\n"].tolist()[:FREQ]

norwegian_freq = get_freqs(freq_words_list, "no")

# swedish
frequent_words = pd.read_xml("canine/frequency_lists/swedish_kelly.xml")
frequent_words['wpm'] = frequent_words['wpm'].str.replace(',','.').astype(float)
frequent_words = frequent_words.sort_values(by=['wpm'], ascending=False)
freq_word_list = frequent_words["gf"].tolist()[:FREQ]

swedish_freq = get_freqs(freq_words_list, "sv")


char_freq_df = char_freq_df.append(danish_freq)
char_freq_df = char_freq_df.append(faroese_freq)
char_freq_df = char_freq_df.append(finnish_freq)
char_freq_df = char_freq_df.append(icelandic_freq)
char_freq_df = char_freq_df.append(norwegian_freq)
char_freq_df = char_freq_df.append(swedish_freq)

#char_freq_df = char_freq_df.transpose()

char_freq_df = char_freq_df.rename({0: 'freq'}, axis=1)
#print(char_freq_df)
print(char_freq_df)
char_freq_df.to_csv("char_freqs.csv")
