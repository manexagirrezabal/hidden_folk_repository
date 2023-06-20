import pandas as pd
import numpy as np
#import panphon
from tqdm import tqdm
from ipaembedding import IPAEmbedding

import sys
import os

langs = {
#    "dan":"danish",
#    "fao":"faroese",
#    "fin":"finnish",
#    "isl":"icelandic",
#    "nob":"norwegian",
#    "swe":"swedish",
#    "bog":"bogstavlydDK",
#    "eng":"english",
#    "eus":"basque",
"ru":"russian",
"fi":"finnish",
"esc":"spanish (cast)",
"esl":"spanish (latin)",
"hu":"hungarian",
"mk":"macedonian",
"la":"latin",
"cs":"czech",
"jah":"japanese hiragana",
"ko":"korean",
"vi":"vietnamese",
"hye":"east armenian",
"hyw":"west armenian",
"lt":"lithuanian",
"de":"german",
"eo":"esperanto",
"uk":"ukrainian",
"hi":"hindi",
"da":"danish",
"tl":"tagalog",
"bg":"bulgarian",
"ro":"romanian",
"jak":"japanese katakana",
"fo":"faroese",
"en":"english",
"nl":"dutch"
}


path = sys.argv[1]
file_name = os.path.basename(path)
lang_code = file_name[0:3]
lang = langs[lang_code]

m2m_converter = lambda x:x.split(" ")

converters = {
    "graphs":m2m_converter,
    "phones":m2m_converter
}

# Read data
words = pd.read_csv("../wikipron/"+path, usecols=[0,1], sep="\t", header=None, names=["graphs", "phones"], converters=converters)

# Ipapy features
E = IPAEmbedding()
features = E.feature_names
columns = ["char", "word", "position", "combined_phon", "combined_graph", "error"]+features
#    ft = panphon.FeatureTable()

embedding_table = []
b = False
for i, word in tqdm(words.iterrows(), total=len(words)):
    graphs = word["graphs"]
    phones = word["phones"]

    # Expand [m:m, år:ɒˀ, er:ɒ, s:s] to [m:m, å:ɒˀ, r:ɒˀ, e:ɒ, r:ɒ, s:s]
    graphs_unfold = []
    phones_unfold = []
    combined_graph = []
    for g, p in zip(graphs, phones):
        if len(list(g)) > 1:
            combined_graph.append(True)
        else:
            combined_graph.append(False)

        graphs_unfold.extend(list(g))
        phones_unfold.extend([p,]*len(list(g)))
        
    for i, (g, p, combined_g) in enumerate(zip(graphs_unfold, phones_unfold, combined_graph)):
        combined_p=False
        
        feats, combined = E[p]
        error=False
        if sum(feats)==0:error=True

        embedding_table.append([g, "".join(graphs), i, combined_p, combined_g, error]+feats)
            

ipa_embeddings = pd.DataFrame(embedding_table, columns=columns)
ipa_embeddings.to_csv("ipa_embeddings_{}.csv".format(lang_code), index=False, header=True)
