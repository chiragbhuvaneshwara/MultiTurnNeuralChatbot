# Approaches
- objective function
- additional(?) reinforcement learning via idf
- predefined keyword / topic (e.g. Mou et al, Xing et al.)
- latent factors
- learn usage representation (how specific it is)

## Objective function (MMI)
goal: single diverse output

method: Replace log-likelihood loss function (here: maskNLLLoss) with a loss function that includes Mutual Information
- have a hyperparater lambda to adjust the tradeoff between p(S|T) and p(T|S)
- Trained only a maximum likelihood model and use MMI criterion only during testing
- Two (mathematicall equivalent?) notations MMI-antiLM and MMI-bidi
- Problem: answers of MMI-antiLM can be ungrammatical, decoding of MMI-bidi can be intractable

### MMI-antiLM
Problem: answers can be ungrammatical

p(T) is actually a language model
- don't use p(T) directly but replace it with a modified language model U(T)
- U(T) takes into account how long the sentence already is
- U(T) = \product_{k = 1}^N p(t_k | t_i, ... t_k - 1) * g(k)
- g(k) is a weight that decreases when k increases
- Language model becomes less important at the end of the sentence?
- here: g(k) with a threshold

- log p(T |S) − λ log U (T )

- include length penalty:
- Score(T ) = p(T |S) − λU (T ) + γNt
 


### MMI-bidi
 (1 − λ) log p(T |S) + λ log p(S|T )
- generate n-best lists with  (1 − λ) log p(T |S) as objective function
- rank n-best list with λ log p(S|T )

### Training
- optimize parameter with MERT (like Li et al. ) or grid search

Parameters from Li et al.
- beam size 200
- max response length 20


## Objective function (idf)
- Yao et al. (2016)
- reinforcement learning with idf-value of generated sentence as reward

## Specifity
- Zhang et al. (2018)
- have explicit control variable for specifity
- learn usage representation similar to word embeddings

# Schedule

# Paper

- Li et al. (2016): A Diversity-Promoting Objective Function for Neural Conversation Models
- Zhang et al. (2018): Learning to Control the Specificity in Neural Response Generation
- Li et al. (2017): Data Distillation for Controlling Specificity in Dialogue Generation
- Yao et al. (2016): An Attentional Neural Conversation Model with Improved Specificity
(- Shen et al. (2017): A Conditional Variational Framework for Dialog Generation)
(- Zhou et al. (2017): Mechanism-Aware Neural Machine for Dialogue Response Generation)

# TODO:
check pairs as output from load_data.py, read_vocabulary(corpus_file)



