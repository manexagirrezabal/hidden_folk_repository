import pandas as pd
import sys
import numpy as np
from tqdm import tqdm

lang = sys.argv[1]
input_path = "canine/contextual_canine_char_embeddings_{}_3000.csv".format(lang)

labels_path = "ipa/ipa_embeddings_{}.csv".format(lang)

input_embeddings = pd.read_csv(input_path)
print(len(input_embeddings))
labels = pd.read_csv(labels_path)

input_embeddings["index_input"] = input_embeddings.index
labels["index_labels"] = labels.index

# Merge input and labels
data_merge = pd.merge(input_embeddings, labels, how="left", on=["char", "word", "position"])

data = data_merge[~data_merge["index_labels"].isnull()]
print("{} of the most-frequent words ({} characters) has labels.".format(len(set(data["word"])), len(data)))
print()

# Check if there exists duplicates (due to homographs). Should these be removed?
duplicates = data.duplicated(subset="index_input")
if sum(duplicates)>1:
    print("Following duplicates were found:")
    print(data[duplicates].iloc[:, [0, 1, 2]])

features = labels.columns[6:-1]

check_for = features
assert all(f in features for f in check_for)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score,f1_score

filter_global_type = True

results_df = pd.DataFrame()

indiv_letter_results = pd.DataFrame()

for feature in check_for:

    if filter_global_type and (not feature.startswith("global_type")):
        if feature.startswith("consonant_"):
            global_type="global_type_consonant"
        elif feature.startswith("vowel_"):
            global_type="global_type_vowel"
        elif feature.startswith("diacritic_"):
            global_type="global_type_diacritic"
        elif feature.startswith("suprasegmental_"):
            global_type="global_type_suprasegmental"
        elif feature.startswith("tone_"):
            global_type="global_type_tone"
        else:
            raise AssertionError("Unknown feature type in feature: [{}]".format(feature))
        data_filter = data[data[global_type]!=0]
    else:
        global_type="global"
        data_filter = data

    print("Testing {}:".format(feature))
    print("..........................")
    if filter_global_type:
        print("Filtered instances from {} to {} (global type {})".format(len(data), len(data_filter), global_type))
    try:
        positives = data_filter[data_filter[feature]==1.0]
        negatives = data_filter[data_filter[feature]==0.0]

        pos_examples = positives["char"].to_list()
        neg_examples = negatives["char"].to_list()

        pos_evidence = [pos_examples.count(c) for c in set(pos_examples)]
        neg_evidence = [neg_examples.count(c) for c in set(neg_examples)]

        print("Number of unique positives: {} ({})".format(len(set(pos_examples)),set(pos_examples)))
        print("\twith evidence from {} samples".format("{}--{}".format(min(pos_evidence),max(pos_evidence))))
        print("Number of unique negatives: {} ({})".format(len(set(neg_examples)),set(neg_examples)))
        print("\twith evidence from {} samples".format("{}--{}".format(min(neg_evidence),max(neg_evidence))))

        results = []
        sample_weight = []
        all_true = []
        all_predicted = []
        all_test_chars = []
        for char in tqdm(set(data_filter["char"]),leave=True):
            train_instances = data_filter[data_filter["char"]!=char]
            test_instances = data_filter[data_filter["char"]==char]

            train_indices, train_labels = train_instances.index, train_instances[feature]
            test_indices, test_labels = test_instances.index, test_instances[feature]

            # Get input embeddings
            x = data_filter.loc[train_indices, ["d{}".format(i) for i in range(1,768)]]
            y = data_filter.loc[test_indices, ["d{}".format(i) for i in range(1,768)]]

            clf = LogisticRegression(random_state=0, max_iter=10000).fit(x, train_labels)

            pred = clf.predict(y)
            results.append(f1_score(test_labels, pred,average="weighted"))
            all_true += test_labels.tolist()
            all_predicted += pred.tolist()
            all_test_chars += [char] * len(test_labels.tolist())
            sample_weight.append(len(pred))

            f_results = pd.DataFrame({'f1swe': [f1_score(test_labels, pred,average="weighted")], 'char': [char], 'feature' : [feature], 'language' : [lang]})

            print(f_results)
            indiv_letter_results = pd.concat([indiv_letter_results, f_results])

        print("Average accuracy: {} (std {})".format(np.average(results), np.std(results)))
        print("Average accuracy (weighted): {} (std {})".format(np.average(results, weights=sample_weight), np.std(results)))
        print()
        print()

        print(classification_report(all_true, all_predicted))
        print()
        print()

        #f_results = pd.DataFrame({'true': all_true, 'pred': all_predicted, 'char': all_test_chars, 'feature' : [str(feature)] * len(all_true), 'language' : [lang]* len(all_true)})
        #print(f_results)
        #indiv_letter_results = pd.concat([indiv_letter_results, f_results])

        task_results = classification_report(all_true, all_predicted, output_dict=True)
        #print(task_results)

        for type, res in task_results.items():
            #print(res)
            if type != 'accuracy':
                res['feature'] = feature
                res['result_type'] = type
                results_df = results_df.append(res, ignore_index = True)

    except ValueError as e:
        print(e)
        print()
        print()

#print(results_df)
results_df.to_csv("probing_results/"+lang+"-lolo.csv", index=False)

print(indiv_letter_results)
indiv_letter_results.to_csv("probing_results/"+lang+"-lolo-allChars.csv", index=False)

#sns.heatmap(indiv_letter_results, annot=True)
