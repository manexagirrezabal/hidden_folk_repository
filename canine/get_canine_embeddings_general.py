import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from transformers import CanineTokenizer, CanineModel
from tqdm import tqdm
import random
from config import WORDS, FREQ

import sys

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

LANG = sys.argv[1]
# Load English frequency list
#frequent_words = pd.read_csv("frequency_lists/lemmasfreq_english.txt", sep="\t", header=None, names=["lemma", "freq"])
#print(len(frequent_words))
#freq_words_list = frequent_words["lemma"].tolist()[:FREQ]
f=open("frequency_lists/"+LANG+".mostFreq5000",encoding="utf8")
freq_words_list = [line.strip() for line in f]
f.close()

print(len(freq_words_list))
freq_word_list_rand = random.sample(freq_words_list, WORDS)
print(len(freq_word_list_rand))

# load CANINE model
model = CanineModel.from_pretrained('google/canine-c')
tokenizer = CanineTokenizer.from_pretrained('google/canine-c')
print("Model and tokenizer loaded.")

contextual_canine_char_embeddings = []

for group in tqdm(chunker(freq_word_list_rand, 200)):
    inputs = group
    encoding = tokenizer(inputs, padding="longest", truncation=True, return_tensors="pt")

    outputs = model(**encoding) # forward pass
    pooled_output = outputs.pooler_output
    sequence_output = outputs.last_hidden_state

    dimension_columns = ["d"+str(dim) for dim in list(range(sequence_output.shape[2]))]

    # extract individual character representations
    for word, hidden_reps in zip(inputs, sequence_output):
        # ignore beginning-of-sequence and end-of-sequence representations
        hidden_reps = hidden_reps[1:len(word)+1]
        for pos, (char, char_rep) in enumerate(zip(word, hidden_reps)):
            char_dict = {"char": char, "word": word, "position": pos}
            char_data = pd.DataFrame([char_dict])
            char_emb = pd.DataFrame([char_rep.detach().numpy().tolist()], columns=dimension_columns)
            char_data = pd.concat([char_data, char_emb], axis=1)
            contextual_canine_char_embeddings.append(char_data)
    print()
    print(len(contextual_canine_char_embeddings), " character embeddings added.")

contextual_canine_char_embeddings = pd.concat(contextual_canine_char_embeddings)
#contextual_canine_char_embeddings = pd.DataFrame(contextual_canine_char_embeddings, columns=["char", "word", "position"])
contextual_canine_char_embeddings.to_csv("contextual_canine_char_embeddings_"+LANG+"_"+str(len(freq_word_list_rand))+".csv", index=False, mode='w')

print(contextual_canine_char_embeddings)
