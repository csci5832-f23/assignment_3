import nltk
from nltk.corpus import indian
import numpy as np
import random
from typing import List

nltk.download('indian')

UNK_TOKEN = '<unk>'


def get_observation_ids(observations, ob_dict):
    return [[ob_dict[e] if e in ob_dict else ob_dict[UNK_TOKEN] for e in es] for es in observations]


def get_state_ids(tags, state_dict):
    return [[state_dict[t] for t in ts] for ts in tags]


def generate_hmm_data(tagged_sentences):
    """
    create the state, observation dict from nltk tagged sentences
    each tag sentence is of the form list[(word, pos_tag)]
    """
    words = list(set([word for tagged_sent in tagged_sentences for word, tag in tagged_sent]))
    words.sort()
    tags = list(set([tag for tagged_sent in tagged_sentences for word, tag in tagged_sent]))
    tags.sort()

    observation_dict = {word: i for i, word in enumerate(words)}
    state_dict = {state: i for i, state in enumerate(tags)}

    sentence_words = [[word for word, tag in tagged_sent] for tagged_sent in tagged_sentences]
    sentence_tags = [[tag for word, tag in tagged_sent] for tagged_sent in tagged_sentences]

    return words, tags, observation_dict, state_dict, \
           get_observation_ids(sentence_words, observation_dict), \
           get_state_ids(sentence_tags, state_dict)


def get_hindi_dataset():
    """
    generate the inputs required to train an hmm pos tagger for hindi dataset
    """
    all_tagged_sentences = indian.tagged_sents('hindi.pos')
    return generate_hmm_data(all_tagged_sentences)


class HMM:
    def __init__(self, num_states, num_observations):
        self.n = num_states
        self.m = num_observations

        # the small number added is to avoid log(0) error!
        self.pi = np.zeros(num_states) + 0.0000000001
        self.A = np.zeros((num_states, num_states)) + 0.0000000001
        self.B = np.zeros((num_states, num_observations)) + 0.0000000001

    def fit(self, state_ids, observation_ids):
        """
        ENTER CODE HERE: complete the code
        populate the parameters (self.pi, self.A, self.B) by counting the bi-grams
        for self.A use bi-grams of states and states
        for self.B use bi-grams of states and observations
        """
        for s_ids, o_ids in zip(state_ids, observation_ids):
            # count initial states
            # count state->state transitions
            # count state->observations emissions
            # HINT: use zip for creating bi-grams
            raise NotImplementedError

        # normalize the rows of each probability matrix
        self.pi = np.log(self.pi / np.sum(self.pi))
        self.A = np.log(self.A / np.sum(self.A, axis=1).reshape((-1, 1)))
        self.B = np.log(self.B / np.sum(self.B, axis=1).reshape((-1, 1)))

    def path_log_prob(self, state_ids, observation_ids) -> np.array:
        """
        A debugging helper function to calculate the path probability of a given sequence of states and observations
        """
        all_path_log_probs = []
        for sent_states, sent_observations in zip(state_ids, observation_ids):
            # initial prob and all transition probs
            transition_log_probs = np.array([self.pi[sent_states[0]]] +
                                            [self.A[t_1, t_2]
                                             for t_1, t_2 in zip(sent_states[:-1], sent_states[1:])])

            observation_log_probs = np.array([self.B[t, e] for t, e in zip(sent_states, sent_observations)])

            all_path_log_probs.append(transition_log_probs.sum() + observation_log_probs.sum())

        return np.array(all_path_log_probs)

    def decode(self, observation_ids: List[List[int]]) -> List[List[int]]:
        """
        ENTER CODE HERE: complete the code
        Viterbi Algorithm: Follow the algorithm in Jim's book:
        Figure 8.10 at https://web.stanford.edu/~jurafsky/slp3/8.pdf
        """
        # store the decoded states here
        all_predictions = []
        for obs_ids in observation_ids:
            T = len(obs_ids)  # Sequence length
            viterbi = np.zeros((self.n, T))  # The viterbi table
            back_pointer = np.zeros((self.n, T))   # backpointers for each state+sequence id
            # TODO: Fill the viterbi table, back_pointer. Get the optimal sequence by backtracking
            ...
            raise NotImplementedError
        return all_predictions


def test_fit():
    # Assume a set of observations: {0, 1, 2} and set of states {0, 1}
    test_n = 2
    test_m = 3

    test_hmm_fit = HMM(test_n, test_m)

    train_states = [[0, 1, 1], [1, 0, 1], [0, 0, 1]]
    train_obs = [[1, 0, 2], [0, 0, 1], [2, 1, 0]]

    test_hmm_fit.fit(train_states, train_obs)

    assert np.round(np.exp(test_hmm_fit.pi)[0], 3) == 0.667

    assert np.round(np.exp(test_hmm_fit.A)[0, 1], 2) == 0.75

    assert np.round(np.exp(test_hmm_fit.B)[1, 0], 1) == 0.6

    print('All Test Cases Passed!')


def test_decode():
    # Assume a set of observations: {0, 1, 2} and set of states {0, 1}
    test_n = 2
    test_m = 3

    test_hmm_tagger = HMM(test_n, test_m)
    test_hmm_tagger.pi = np.log([0.5, 0.5])
    test_hmm_tagger.A = np.log([[0.3, 0.7], [0.6, 0.4]])
    test_hmm_tagger.B = np.log([[0.2, 0.5, 0.3], [0.3, 0.1, 0.6]])

    test_state_observation_ids = [[[1, 0, 1], [2, 1, 1]]]
    test_state_ids = [[1, 0, 1]]
    test_obs_ids = [[2, 1, 1]]
    test_forwards = test_hmm_tagger.path_log_prob(test_state_ids, test_obs_ids)

    expected_prob = 0.5 * 0.6 * 0.6 * 0.5 * 0.7 * 0.1

    predicted_forward = np.exp(test_forwards)[0]

    assert np.round(predicted_forward, 4) == expected_prob

    ### TEST Viterbi Decoding Method ###

    test_observations = [[2, 1, 1]]

    all_possibilities = [[s1, s2, s3] for s1 in range(2) for s2 in range(2) for s3 in range(2)]

    all_observations = test_observations * len(all_possibilities)

    all_forwards = test_hmm_tagger.path_log_prob(all_possibilities, all_observations)

    best_forward_prob = np.max(all_forwards)
    best_forward_index = np.argmax(all_forwards)
    best_path_true = all_possibilities[best_forward_index]

    decoded_states = test_hmm_tagger.decode(test_observations)
    
    # Test the first and last states of the decoded states
    assert decoded_states[0][0] == best_path_true[0]
    assert decoded_states[0][-1] == best_path_true[-1]
    assert len(decoded_states[0]) == len(best_path_true)
    
    print('All Test Cases Passed!')


def run_tests():
    print('Testing the fit function of the HMM')
    test_fit()

    print('Testing the decode function of the HMM')
    test_decode()

    print('Yay! You have a working HMM. Now try creating a pos-tagger using this class.')
