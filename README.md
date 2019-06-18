## JNTM
Source code for ACM TOIS 2017 paper "A Neural Network Approach to Joint Modeling Social Networks and Mobile Trajectories".

### Description

UserMain is the main body. NetworkModel handles the loss function of network structure and UserLSTM includes other parts (RNN, LSTM and fixed preference embedding).

### Input format

train, test: Each line contains a user id followed by a sequence of locations ids.

graph: Each line contains the ids of two connected nodes. An edge need to be inputed twice for undirected graphs.

### Cite

If you use the code, please cite this paper:

```
@article{yang2017neural,
  title={A neural network approach to jointly modeling social networks and mobile trajectories},
  author={Yang, Cheng and Sun, Maosong and Zhao, Wayne Xin and Liu, Zhiyuan and Chang, Edward Y},
  journal={ACM Transactions on Information Systems (TOIS)},
  volume={35},
  number={4},
  pages={36},
  year={2017},
  publisher={ACM}
}
```
### Contact
albertyang33@gmail.com
