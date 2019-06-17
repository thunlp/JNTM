# JNTM
code for ACM TOIS paper "A Neural Network Approach to Joint Modeling Social Networks and Mobile Trajectories"

UserMain is the main body. NetworkModel handles the loss function of network structure and UserLSTM includes other parts (RNN LSTM fixed embedding).

Input format:
train, test: Each line contains a user id followed by a sequence of locations ids.
graph: Each line contains the ids of two connected nodes.

