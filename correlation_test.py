import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import network_helpers
import scipy

def correlation_diagnostics(decoded_PC_names, correlation_indices, dupe_indices, r_vals):
    """ returns F1 score of classification of correlated PCs via dupe names """

    positive_results = dupe_indices.intersection(correlation_indices)

    # these need to be computed from the pairs because otherwise results are skewed by overlapping cases
    num_correct = len(set(list(positive_results)))
    num_dupes = len(set(list(dupe_indices)))
    num_correlated = len(set(list(correlation_indices)))

    decode_indices = lambda indices: set([(decoded_PC_names[idx[0]], decoded_PC_names[idx[1]]) for idx in indices])

    positive_names = decode_indices(positive_results)

    correlated_non_dupes = correlation_indices - dupe_indices
    correlated_non_dupes = decode_indices(correlated_non_dupes)

    non_correlated_dupes = dupe_indices - set(correlation_indices)
    non_correlated_dupes = decode_indices(non_correlated_dupes)

    print('avg r_val= ', np.mean(r_vals))
    print("decoded_PC_names: ", decoded_PC_names)
    print("positive_names: ", positive_names)  # names of correlated & duplicate named embeddings

    print("non_correlated_dupes: ", non_correlated_dupes)
    print("correlated_non_dupes: ", correlated_non_dupes)

    print("num_dupes: ", num_dupes)
    print("num_correlated: ", num_correlated)

    try:
        precision = num_correct / num_dupes  # accuracy of identified dupes
        print("precision: ", precision)
        recall = num_correct / num_correlated  # portion of true correlated cases that it identified
        print("recall: ", recall)
        F1_score = 2 / (1 / precision + 1 / recall) if precision and recall else 0   # harmonic mean of recall & precision
        print("F1 score: ", F1_score)
        return F1_score
    except ZeroDivisionError:
        return None


def correlation_test(embedder, PC_name_vecs, log_fn = './logs/correlation_test.csv'):
    """ determines portion of PCs with redundant names that are also correlated out of all redundant names """

    decoded_PC_names, _ = embedder.soft_inverse(PC_name_vecs)
    recoded_PC_name_vecs = embedder.embed(decoded_PC_names)

    r_vals_df = pd.DataFrame(columns=['name1', 'name2', 'dist_r', 'name_sim', 'recoded_name_sim'])

    for i in range(embedder.vocab_embeddings.shape[1]):
        for j in range(i, embedder.vocab_embeddings.shape[1]):
            if i == j: continue  # we don't want correlation with itself
            r, p_val = scipy.stats.pearsonr(embedder.vocab_embeddings[:, i],
                                            embedder.vocab_embeddings[:, j])
            # ^ p-value for: x[:,i].T.dot(x[:,j])

            # transpose not really necessary, but good reminder for matrix case
            name_sim = float(network_helpers.np_cos_similarity(PC_name_vecs[i].T, PC_name_vecs[j].T))
            recoded_name_sim = float(network_helpers.np_cos_similarity(recoded_PC_name_vecs[i].T, recoded_PC_name_vecs[j].T))

            r_vals_df = r_vals_df.append({'name1': decoded_PC_names[i], 'name2': decoded_PC_names[j],
                              'dist_r': r, 'dist_p': p_val, 'name_sim': name_sim,
                              'recoded_name_sim': recoded_name_sim}, ignore_index=True)
    r_vals_df.to_csv(log_fn)


def compare_PC_name_algs(embedder, PC_names_A, PC_names_B):
    decoded_PC_names_A, _ = embedder.soft_inverse(PC_names_A)
    decoded_PC_names_B, _ = embedder.soft_inverse(PC_names_B)

    print("correlation test (A) % accuracy: ", correlation_test(embedder, PC_names_A, './logs/correlation_testA.csv'))
    print("correlation test (B) % accuracy: ", correlation_test(embedder, PC_names_B, './logs/correlation_testB.csv'))

    stacked_comparison = np.stack([decoded_PC_names_A, decoded_PC_names_B]).T
    print("decoded_PC_names_A stacked next to decoded_PC_names_B: ", stacked_comparison)

    # this should roughly match! It doesn't!!
    print("portion of stemmed_PC_names_A & stemmed_PC_names_B that match:",
          np.mean([word1 == word2 for word1, word2 in zip(decoded_PC_names_A, decoded_PC_names_B)]))

    # cosine similarities operates on axis=0
    similarities = network_helpers.np_cos_similarity(PC_names_A.T, PC_names_B.T)
    plt.hist(similarities)
    plt.xlabel('cosine similarity')
    plt.title('similarities of A/B PC names')
    plt.savefig('similarities_of_AB_PC_names.png')
    print("similarities: ", similarities)
    print("mean similarity: ", np.mean(similarities))
