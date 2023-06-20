import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import sys
import os

feature_set = "global"
include_features = ["global_type_consonant", "global_type_vowel"] # "global_type_diacritic"
short_names = ["consonant", "vowel"] #"diacritic"
result_avg = 'weighted avg'

results_all = pd.DataFrame()
for file in os.listdir("probing_results/"):
    if file.endswith("-8020.csv"):
        lang = file.split("_")[-1][:-9]
        print(lang)
        results = pd.read_csv("probing_results/"+file)
        results['language'] = lang
        results_all = results_all.append(results)

print(results_all)
results_all = results_all.loc[results_all['result_type'] == result_avg]
results_all = results_all.loc[results_all['feature'].isin(include_features)]
print(len(results_all))
print(results_all)

sns.color_palette("Set2")
g = sns.barplot(data=results_all, x="feature", y="f1-score", hue='language', palette=sns.color_palette("Set2"))
#g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend([],[], frameon=False)
plt.title(feature_set + " 80/20")
g.set_xticklabels(short_names)
plt.ylim(0.2,1)
#plt.xticks(rotation = 90)
g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3,prop={'size': 12})
plt.tight_layout()
plt.savefig("plots/"+"results-8020-"+feature_set+".pdf")
#plt.show()
plt.close()

results_lolo = pd.DataFrame()
for file in os.listdir("probing_results/"):
    if file.endswith("-lolo.csv"):
        lang = file.split("_")[-1][:-9]
        print(lang)
        results = pd.read_csv("probing_results/"+file)
        results['language'] = lang
        results_lolo = results_lolo.append(results)

print(results_lolo)
results_lolo = results_lolo.loc[results_lolo['result_type'] == result_avg]
results_lolo = results_lolo.loc[results_lolo['feature'].isin(include_features)]
print(len(results_lolo))
print(results_lolo)

sns.color_palette("Set2")
g = sns.barplot(data=results_lolo, x="feature", y="f1-score", hue='language', palette=sns.color_palette("Set2"))
#g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(feature_set + " LOLO")
g.set_xticklabels(short_names)
plt.ylim(0.2,1)
#plt.xticks(rotation = 90)
g.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3,prop={'size': 12})
plt.tight_layout()
plt.savefig("plots/"+"results-lolo-"+feature_set+".pdf")
#plt.show()
plt.close()


results_all["eval"] = "8020"
results_lolo["eval"] = "lolo"
all_df = pd.concat([results_all, results_lolo])
vowel = all_df[all_df["feature"] == "global_type_vowel"]
fig = plt.figure(figsize=(8, 4))
print(all_df)
b = sns.barplot(data=vowel, x="language", y="f1-score", hue='eval', palette=sns.color_palette("Set2"))
plt.ylim(0.6,1)
b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=2,prop={'size': 12})
plt.savefig("plots/"+"results-overview-vowel.pdf")
plt.show()
plt.close()

consonant = all_df[all_df["feature"] == "global_type_consonant"]
print(all_df)
fig = plt.figure(figsize=(8, 4))
b = sns.barplot(data=consonant, x="language", y="f1-score", hue='eval', palette=sns.color_palette("Set2"))
plt.ylim(0.6,1)
b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=2,prop={'size': 12})
plt.savefig("plots/"+"results-overview-consonant.pdf")
plt.show()
