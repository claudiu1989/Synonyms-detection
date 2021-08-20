"""Popescu Claudiu, 2021.

Code for a synonyms detection experiment. 
Synonyms detection is based on word2vec embeddings and cosine similarity.

Usage:

python ./synonyms_detection.py
"""

from gensim.models import Word2Vec

def evaluate_synonyms(model, th):
    with open('sinonime.txt', 'r', encoding='UTF-8') as data_file:
        TP = 0
        OVS = 0
        j = 0
        for line in data_file:
            j = j + 1
            left_word, right_word = line.split('=')
            try:
                result = model.wv.similarity(left_word.strip(), right_word.strip())
                if result >= th:
                    TP = TP +1 
            except:
                OVS = OVS + 1

    with open('non_sinonime.txt', 'r', encoding='UTF-8') as data_file:
        TN = 0
        OVN = 0
        i = 0
        for line in data_file:
            i = i + 1
            if i > j: # Limit the number of non-synonyms to that of the synonyms
                break
            left_word, right_word = line.split('=')
            try:
                result = model.wv.similarity(left_word.strip(), right_word.strip())
                if result < th:
                    TN = TN +1 
            except:
                OVN = OVN + 1
    print(f'True positives: {TP/float(j-OVS)}')
    print(f'Out-of-vocabulary synonyms: {OVS/float(j)}')
    print(f'Out-of-vocabulary non-synonyms: {TN/float(i-OVN)}')
    print(f'OVN: {OVN/float(i)}')

if __name__ == '__main__':
    # Be sure you have the models and data files in the current dir
    
    model_name = "word2vec-ro-100-cbow-negative_sampling.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-200-cbow-negative_sampling.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-300-cbow-negative_sampling.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.05)

    model_name = "word2vec-ro-100-preprocessed-cbow-negative_sampling.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-200-preprocessed-cbow-negative_sampling.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-300-preprocessed-cbow-negative_sampling.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-100-skipgram-negative_sampling.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.35)

    model_name = "word2vec-ro-200-skipgram-negative_sampling.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-300-skipgram-negative_sampling.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-100-preprocessed-skipgram-negative_sampling.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-200-preprocessed-skipgram-negative_sampling.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-300-preprocessed-skipgram-negative_sampling.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-100-cbow-hierarchical_softmax.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-200-cbow-hierarchical_softmax.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-300-cbow-hierarchical_softmax.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-100-preprocessed-cbow-hierarchical_softmax.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-200-preprocessed-cbow-hierarchical_softmax.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-300-preprocessed-cbow-hierarchical_softmax.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-100-skipgram-hierarchical_softmax.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-200-skipgram-hierarchical_softmax.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-300-skipgram-hierarchical_softmax.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-100-preprocessed-skipgram-hierarchical_softmax.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-200-preprocessed-skipgram-hierarchical_softmax.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)

    model_name = "word2vec-ro-300-preprocessed-skipgram-hierarchical_softmax.model"
    model = Word2Vec.load(model_name)
    print(f'Results for {model_name}')
    evaluate_synonyms(model, 0.2)