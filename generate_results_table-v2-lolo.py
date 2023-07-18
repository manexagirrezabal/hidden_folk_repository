import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import sys
import os

results_all = pd.DataFrame()
for file in os.listdir("probing_results/"):
    if file.endswith("lolo.csv"):
#    if file.endswith(".csv") and "lolo" not in file:
        lang = file.split("_")[-1][:-9]
        print(file)
        print(lang)
        results = pd.read_csv("probing_results/"+file)
        results['language'] = lang
        results_all = results_all.append(results)

print(results_all)
results = results_all[results_all["result_type"] == "weighted avg"]
results = results.drop(["result_type", "support"], axis=1)


#include_features = ["global_type_consonant", "global_type_vowel"]
#results = results[results["feature"].isin(include_features)]
results = results[["language", "feature", "f1-score"]] # "precision", "recall",

final_results = {}
feature_set = set(results.feature)
#for lang in set(results.language):
#    feature_set = feature_set.intersection(results[results.language==lang])
#table = []
#for lang in set(results.language):
#    final_results[lang] = results[results.language==lang].set_index("feature")
#    table.append(results[results.language==lang].set_index("feature"))
#    final_results[lang] = final_results[lang]


#table = pd.DataFrame(0,index=list(set(results.language)),columns=list(set(feature_set)))
table = {}
for lang in list(set(results.language)):
    for feature in feature_set:
        tmpr = results[results.language==lang]
        if sum(tmpr.feature==feature)>0:
#            table.loc[lang,feature] = tmpr[tmpr.feature==feature]['f1-score']
            if lang not in table.keys():
                table[lang] = {}
            if feature not in table[lang].keys():
                table[lang][feature]="-1"
            table[lang][feature] = tmpr[tmpr.feature==feature]['f1-score'].values[0]
#            table.loc[lang,feature] = tmpr[tmpr.feature==feature]['f1-score']
table_df = pd.DataFrame(table)

#Filter out features that don't have info in more than 10 languages
table_df = table_df[table_df.isna().sum(axis=1)<10]

table_df = table_df.fillna(-1)
sns.heatmap(table_df,yticklabels=True)
plt.show()

#sprint(results.to_latex(index=False, float_format="%.2f"))


#print(len(results["feature"].unique()))

print(results.to_csv("test-table.csv"))
