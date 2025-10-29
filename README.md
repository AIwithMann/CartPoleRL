This repository is about implementing 1-step semi gradient Sarsa on the Cart-Pole environment using tile coding.
For environment construction, I use gymnasium here.

### Files:<br>
- `tilecoding.py`: Implements tilecoding <br>
- `sarsa.py`: Implements 1-step semi gradient Sarsa algorithm on cart-pole task with using tile coding from `tilecoding.py` <br>
- `evaluate.py`: Runs the sarsa.py with different hyperparameters to get the answer to the question - "what combination of hyperparameters are the best"
- `evaluation.md`: Summarizing the graph that I got from `evalute.py` in words.
