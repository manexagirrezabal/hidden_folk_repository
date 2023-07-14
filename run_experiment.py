from cProfile import label
import pandas as pd
import sys

lang = sys.argv[1]
input_path = "canine/contextual_canine_char_embeddings_{}_3000.csv".format(lang)

labels_path = "ipa/ipa_embeddings_{}.csv".format(lang)

input_embeddings = pd.read_csv(input_path)
labels = pd.read_csv(labels_path)

input_embeddings["index_input"] = input_embeddings.index
labels["index_labels"] = labels.index

# Check that we have labels for all words --> OBS: For some reason this gets stuck in a loop
# Strategy if we don't?
# assert all(word in labels["word"].tolist() for word in input_embeddings["word"].tolist())

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
from sklearn.metrics import classification_report

filter_global_type = True

results_df = pd.DataFrame()

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
        train_indices, test_indices, train_labels, test_labels = train_test_split(data_filter.index, data_filter[feature], stratify= data_filter[feature], test_size=0.3, random_state=0)

        print(len(train_indices), len(test_indices))

        # Get input embeddings
        x = data_filter.loc[train_indices, ["d{}".format(i) for i in range(1,768)]]
        y = data_filter.loc[test_indices, ["d{}".format(i) for i in range(1,768)]]

        clf = LogisticRegression(random_state=0, max_iter=10000).fit(x, train_labels)

        pred = clf.predict(y)

        print(classification_report(test_labels, pred))
        print()
        print()

        task_results = classification_report(test_labels, pred, output_dict=True)
        print(task_results)

        for type, res in task_results.items():
            print(res)
            if type != 'accuracy':
                res['feature'] = feature
                res['result_type'] = type
                results_df = results_df.append(res, ignore_index = True)

    except ValueError as e:
        print(e)
        print()
        print()

print(results_df)
results_df.to_csv("probing_results/"+lang+"-8020.csv", index=False)
