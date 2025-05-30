from datasets import load_dataset
import nltk
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from collections import Counter
from itertools import combinations
import tqdm
import pickle

# Download required NLTK data files
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt_tab')

def get_noun_cooccurrences(captions):
    lemmatizer = WordNetLemmatizer()
    cooccur = Counter()
    for caption in tqdm.tqdm(captions):
        tokens = word_tokenize(caption)
        pos_tags = pos_tag(tokens)
        nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
        noun_lemmas = set(lemmatizer.lemmatize(noun.lower(), pos='n') for noun in nouns)
        for noun1, noun2 in combinations(sorted(noun_lemmas), 2):
            cooccur[(noun1, noun2)] += 1
    return cooccur

def main():
    ds = load_dataset("sentence-transformers/coco-captions")
    # Assuming default split 'train'
    captions = []
    # Limit to first 1000 examples for testing
    # ds["train"] = ds['train'].select(range(1000))
    for ex in ds['train']:
        captions.append(ex['caption1'])
        captions.append(ex['caption2'])
    cooccurrences = get_noun_cooccurrences(captions)
    # Print top 10 co-occurring noun pairs
    with open('noun_cooccurrences.pkl', 'wb') as f:
        pickle.dump(cooccurrences, f)

    with open('noun_cooccurrences.csv', 'a') as f:
        f.write("noun1,noun2,count\n")
        for (noun1, noun2), count in cooccurrences.most_common(1000):
            print(f"{noun1}, {noun2}: {count}")
            f.write(f"{noun1},{noun2},{count}\n")

        


if __name__ == "__main__":
    main()