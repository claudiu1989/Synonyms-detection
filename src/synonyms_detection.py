"""Popescu Claudiu, 2021.

Code for a synonyms detection experiment. 
Synonyms detection is based on word2vec embeddings and cosine similarity.

Usage:

python ./synonyms_detection.py
"""

from gensim.models import Word2Vec


def evaluate_synonyms(model, th):
    """The main function for synonyms experiment.
       TODO: add type hints

        Args:
            model: The word-vector model.
            th: The cosine similarity threshold.

    """
    with open('sinonime.txt', 'r', encoding='UTF-8') as data_file:
        TP = 0  # True positives
        OVS = 0  # Out-of-vocabulary synonyms
        j = 0
        for line in data_file:
            j = j + 1
            left_word, right_word = line.split('=')
            try:
                result = model.wv.similarity(left_word.strip(), right_word.strip())
                if result >= th:
                    TP = TP + 1 
            except:  # TODO: use a specific exception
                OVS = OVS + 1

    with open('non_sinonime.txt', 'r', encoding='UTF-8') as data_file:
        TN = 0  # True negatives
        OVN = 0  # Out-of-vocabulary non-synonyms
        i = 0
        for line in data_file:
            i = i + 1
            if i > j:  # Limit the number of non-synonyms to that of the synonyms
                break
            left_word, right_word = line.split('=')
            try:
                result = model.wv.similarity(left_word.strip(), right_word.strip())
                if result < th:
                    TN = TN + 1
            except:  # TODO: use a specific exception
                OVN = OVN + 1
    print(f'True positives rate: {TP/float(j-OVS)}')
    print(f'Out-of-vocabulary synonyms rate: {OVS/float(j)}')
    print(f'True negatives rate: {TN/float(i-OVN)}')
    print(f'Out-of-vocabulary non-synonyms rate: {OVN/float(i)}')


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