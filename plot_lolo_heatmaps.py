import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import sys
import os

lang_abbrevs = {"swedish": "sv", "danish": "da", "faroese": "fo", "finnish": "fi", "norwegian": "no", "icelandic": "is"}
lang_abbrevs = {"sv": "swedish", "da": "danish", "fo": "faroese", "fi": "finnish", "no": "norwegian", "is": "icelandic"}
#lang_abbrevs = {'vi': 'vi', 'esc': 'esc', 'uk': 'uk', 'hyw': 'hyw', 'lt': 'lt', 'fi': 'fi', 'bg': 'bg', 'da': 'da', 'hye': 'hye', 'esl': 'esl', 'de': 'de', 'mk': 'mk', 'hi': 'hi', 'eo': 'eo', 'la': 'la', 'ro': 'ro', 'tl': 'tl', 'nl': 'nl', 'en': 'en', 'jak': 'jak', 'cs': 'cs', 'ru': 'ru', 'fo': 'fo', 'hu': 'hu'}

feature_set = "global_type_vowel" #diacritic, consonant, vowel

results_all = pd.DataFrame()
for file in os.listdir("probing_results/"):
    if file.endswith("lolo-allChars.csv"):
        print (file)
        results = pd.read_csv("probing_results/"+file)
        lang = file.split("_")[-1][:-18]
        print(results['feature'].unique())
        results = results.loc[results['feature'] == feature_set]
        results = results.drop(columns=['feature'])
        results = results.transpose()
        results.columns = results.iloc[1]
        results = results.drop(index='char', axis=0)
        results = results.drop(index='language', axis=0)
        results['language'] = lang
        results = results.reset_index()
        #
        del results["index"]
        print(results)
        results_all = results_all.append(results)

#print(results_all)


#print(results_all)
print(len(results_all))

chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y','z']
chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y','z']
#print(results_all.columns)
results_all = results_all[['language']+chars]
#results_all = results_all[['language', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'x', 'y', 'á', 'å', 'æ', 'é', 'í', 'ð', 'ó', 'ö', 'ø', 'ú', 'ý']]
language_order = results_all['language']
results_all = results_all.drop(columns=['language'])
results_all = results_all.reset_index(drop=True)
results_all = results_all.apply(pd.to_numeric, errors='coerce')

fig, axarr = plt.subplots(2, 1, figsize=(14,7), sharex=True, gridspec_kw={'height_ratios': [4, 1]})



sns.color_palette("vlag", as_cmap=True)
g = sns.heatmap(data=results_all, annot=True, annot_kws={"size":6}, cbar=False, fmt=".2f", linewidth=.2, cmap=sns.color_palette("coolwarm", as_cmap=True),ax=axarr[0])# palette=sns.color_palette("Set2"))
g.set_ylabel("F1 score")
g.xaxis.label.set_visible(False)
g.set_yticks(range(len(language_order))) #MANEX Small addition
g.set_yticklabels([lang_abbrevs[l] for l in language_order])
g.set_xticks(range(len(chars)))
g.set_xticklabels(chars)

freqs = pd.read_csv("char_freqs.csv", header=0)#, index_col="language")
#freqs = freqs.apply(pd.to_numeric, errors='coerce')
freqs = freqs.rename({"Unnamed: 0": 'char'}, axis=1)
freqs = freqs.sort_values(by=['char'])
freqs = freqs.drop(freqs[freqs.char == "w"].index)
freqs = freqs.drop(freqs[freqs.char == "z"].index)
freqs = freqs.drop(freqs[freqs.char == "ü"].index)
freqs = freqs.drop(freqs[freqs.char == "q"].index)
freqs = freqs.drop(freqs[freqs.char == "ý"].index)
#print(freqs)


#f = sns.barplot(data=freqs, x="char", y="freq", hue="language", palette=sns.color_palette("Paired"),ax=axarr[1])
#axarr[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#          fancybox=True, shadow=True, ncol=6,prop={'size': 12})
##f.xaxis.set_visible(False) # whole axis
#f.xaxis.tick_top()
#f.xaxis.label.set_visible(False)
plt.suptitle(feature_set + " LOLO")
plt.tight_layout()
plt.savefig("plots/"+"heatmap-lolo-"+feature_set+".pdf")
plt.show()



#ax[0].set_xticks(np.arange(0, len(important_pos)), labels=important_pos, rotation=45)
#ax[0].set_yticks(np.arange(0, len(ylabels)), labels=ylabels)
