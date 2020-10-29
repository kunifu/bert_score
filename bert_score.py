import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import BertJapaneseTokenizer

def pairwise_cos_sim(reference_matrix, candidate_matrix):

    d = reference_matrix @ tf.transpose(candidate_matrix)
    reference_matrix_norm = tf.reduce_sum(reference_matrix * reference_matrix, 1, keepdims=True) ** .5
    candidate_matrix_norm = tf.reduce_sum(candidate_matrix * candidate_matrix, 1, keepdims=True) ** .5

    return d / reference_matrix_norm / tf.transpose(candidate_matrix_norm)

def bert_score(
    model: tf.keras.Model,
    tokenizer: BertJapaneseTokenizer,
    reference: str,
    candidate: str,
    use_idf: bool = False
) -> float:

    # tokenize and encode
    reference_tokens = np.array(tokenizer.encode(reference))
    candidate_tokens = np.array(tokenizer.encode(candidate))
    reference_input = reference_tokens[np.newaxis, :]
    candidate_input = candidate_tokens[np.newaxis, :]

    # model output
    reference_output = model(reference_input)[0]
    candidate_output = model(candidate_input)[0]

    # compute cosine similarity for each word pair and get max score
    pairwise_score = pairwise_cos_sim(reference_output[0], candidate_output[0])
    reference_score = np.amax(pairwise_score, axis=1)
    candidate_score = np.amax(pairwise_score, axis=0)

    # compute metrics
    if use_idf:

        token2idf = pd.read_pickle('token2idf.pkl')
        reference_idf = np.array([token2idf[token] for token in reference_tokens])
        candidate_idf = np.array([token2idf[token] for token in candidate_tokens])
        recall = np.sum(reference_score * reference_idf) / np.sum(reference_idf)
        precision = np.sum(candidate_score * candidate_idf) / np.sum(candidate_idf)
        f1 = 2 * precision * recall / (precision + recall)

    else:

        recall = np.sum(reference_score) / reference_score.shape[0]
        precision = np.sum(candidate_score) / candidate_score.shape[0]
        f1 = 2 * precision * recall / (precision + recall)

    return recall, precision, f1