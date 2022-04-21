# Imports
import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
import networkx as nx
from model import SAGE
# from src.data import utils as ut
import config as cnf
import warnings
import shutil
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
warnings.filterwarnings("ignore")
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)
import random
import pandas as pd
import matplotlib.pyplot as plt
import copy
import pickle

# split the graph
def inductive_split(graph):

    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""

    train_g = graph.subgraph(g.ndata['train_mask'])
    val_g = graph.subgraph(graph.ndata['train_mask'] | graph.ndata['val_mask'])
    test_g = copy.deepcopy(graph)
    
    return train_g, val_g, test_g

# Evaluation while training
def al_evaluate(model, test_nfeat, test_labels, device, dataloader, loss_fcn):

    """
    Evaluate the model on the given data set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval() # change the mode

    test_acc = 0.0
    test_loss = 0.0
    class1acc = 0.0
    running_f1 = 0.0

    for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
        with th.no_grad():
            # Load the input features of all the required input nodes as well as output labels of seeds node in a batch
            batch_inputs, batch_labels = load_subtensor(test_nfeat, test_labels,
                                                        seeds, input_nodes, device)

            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)

            temp_pred = th.argmax(batch_pred, dim=1)
            current_acc = accuracy_score(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy() )
            test_acc = test_acc + ((1 / (step + 1)) * (current_acc - test_acc))

            current_f1 = f1_score(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy(), average='weighted')
            running_f1 = running_f1 + ((1 / (step + 1)) * (current_f1 - running_f1))

            # cnfmatrix = confusion_matrix(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy())
            # class1acc = class1acc + ((1 / (step + 1)) * (cnfmatrix[0][0] / np.sum(cnfmatrix[0, :]) - class1acc))

            # print(cnfmatrix)

            # correct = temp_pred.eq(batch_labels)
            # test_acc = test_acc + correct

            loss = loss_fcn(batch_pred, batch_labels)
            test_loss = test_loss + ((1 / (step + 1)) * (loss.data - test_loss))

    model.train() # rechange the model mode to training

    return test_acc, test_loss, running_f1

# Evaluation while testing
def al_evaluate_test(model, test_labels, device, dataloader, loss_fcn):

    """
    Evaluate the model on the given data set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval() # change the mode

    test_acc = 0.0
    test_loss = 0.0
    class1acc = 0.0
    running_f1 = 0.0

    # get intermediate output as node embeddings
    # activation = {}
    #
    # def get_activation(name):
    #     def hook(model, input, output):
    #         activation[name] = output.detach()
    #
    #     return hook

    # model.layers[2].fc_neigh.register_forward_hook(get_activation('layers[2].fc_neigh'))

    for step, (input_nodes, seeds, blocks) in enumerate(dataloader):

        with th.no_grad():
            # Load the input features of all the required input nodes as well as output labels of seeds node in a batch
            batch_inputs, batch_labels = load_subtensor(test_nfeat, test_labels,
                                                        seeds, input_nodes, device)

            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)

            # node_emb = activation['layers[2].fc_neigh']

            temp_pred = th.argmax(batch_pred, dim=1)
            current_acc = accuracy_score(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy() )
            test_acc = test_acc + ((1 / (step + 1)) * (current_acc - test_acc))

            current_f1 = f1_score(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy(), average='weighted')
            running_f1 = running_f1 + ((1 / (step + 1)) * (current_f1 - running_f1))

            # cnfmatrix = confusion_matrix(batch_labels.cpu().detach().numpy(), temp_pred.cpu().detach().numpy())
            # class1acc = class1acc + ((1 / (step + 1)) * (cnfmatrix[0][0] / np.sum(cnfmatrix[0, :]) - class1acc))

            # print(cnfmatrix)

            # correct = temp_pred.eq(batch_labels)
            # test_acc = test_acc + correct

            loss = loss_fcn(batch_pred, batch_labels)
            test_loss = test_loss + ((1 / (step + 1)) * (loss.data - test_loss))

    model.train() # rechange the model mode to training

    return test_acc, test_loss, running_f1

# Subtensor loading function
def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

# Checkpoint management
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    f_path = checkpoint_path
    th.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model):
    checkpoint = th.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    valid_loss_min = checkpoint['valid_loss_min']
    return model, valid_loss_min.item()

# Getting initial labelled dataset, unlabelled pool; and forming validation and test masks
def get_valtest_mask(graph, n_classes, val_size, test_size, base_size):
        array_lst_class = []
        for cclass in range(n_classes):
            array_lst_class.append([])

        for node in graph.nodes():
            temp = graph.ndata['label'][node]
            array_lst_class[temp].append(node.item())

        rng = random.Random(6984)
        array_lst_base_train = [rng.sample(array_lst_class[ind], base_size) for ind in range(n_classes)]

        array_lst_avail_val = []
        for ind in range(n_classes):
            array_lst_avail_val.append([elem for elem in array_lst_class[ind] if elem not in array_lst_base_train[ind]])

        array_lst_val = [rng.sample(array_lst_avail_val[ind], val_size) for ind in range(n_classes)]
        lst_val = [item for sublist in array_lst_val for item in sublist]

        array_lst_avail_test = []
        for ind in range(n_classes):
            array_lst_avail_test.append([elem for elem in array_lst_class[ind] if elem not in array_lst_base_train[ind] and elem not in array_lst_val[ind] ])

        array_lst_test = [rng.sample(array_lst_avail_test[ind], test_size) for ind in range(n_classes)]
        lst_test = [item for sublist in array_lst_test for item in sublist]

        array_lst_avail_train = []
        for ind in range(n_classes):
            array_lst_avail_train.append([elem for elem in array_lst_class[ind] if elem not in array_lst_base_train[ind]
                                          and elem not in array_lst_val[ind] and elem not in array_lst_test[ind] ])

        n_nodes = graph.num_nodes()
        val_mask = th.zeros(n_nodes, dtype=th.bool)
        test_mask = th.zeros(n_nodes, dtype=th.bool)

        val_mask[lst_val] = True
        test_mask[lst_test] = True

        graph.ndata['val_mask'] = val_mask
        graph.ndata['test_mask'] = test_mask
        
        array_lst_base_train = [item for sublist in array_lst_base_train for item in sublist]
        array_lst_avail_train = [item for sublist in array_lst_avail_train for item in sublist]

        return array_lst_base_train, array_lst_avail_train

# Active Node Selection
def get_query(array_lst_avail_train, query_batch_size, sorted_nodeIDs_all):
    flag1 = 0;
    i = 0;
    while (flag1 == 0):
        temp_indx = sorted_nodeIDs_all[i]
        sorted_nodeIDs_all.remove(temp_indx)
        if(temp_indx in array_lst_avail_train):
            flag1 = 1
            node_sel = [temp_indx]
        i = i+1
    return sorted_nodeIDs_all, node_sel

# Updating training mask
def update_train_mask(graph, n_classes, array_lst_base_train, array_lst_avail_train, node_sel):

    array_lst_base_train.extend(node_sel)

    n_nodes = graph.num_nodes()
    train_mask = th.zeros(n_nodes, dtype=th.bool)
    train_mask[array_lst_base_train] = True
    graph.ndata['train_mask'] = train_mask
    try:
        array_lst_avail_train.remove(node_sel[0])
    except:
        pass
    
    return array_lst_base_train, array_lst_avail_train

# Training
def run(args, device, data, checkpoint_path, best_model_path):

    # Unpacking the data
    n_classes, train_g, val_g, test_g, train_nfeat, train_labels, val_nfeat, val_labels, test_nfeat, test_labels, g = data

    in_feats = train_nfeat.shape[1]

    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]

    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]

    # test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    dataloader_device = th.device('cpu')

    if args.sample_gpu:
        train_nid = train_nid.to(device)
        # copy only the csc to the GPU
        train_g = train_g.formats(['csc'])
        train_g = train_g.to(device)
        dataloader_device = device

    # Define dataloader function
    def get_dataloader(train_g, train_nid, sampler):

        dataloader = dgl.dataloading.NodeDataLoader(
            train_g,
            train_nid,
            sampler,
            device=dataloader_device,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers)

        return dataloader

    # Define model and optimizer
    model = SAGE(in_feats, args.hidden_dim, n_classes, args.num_layers, F.relu, args.dropout)
    # print("== # model parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = model.to(device)

    # == customize loss function
    # custom loss function
    # class_weights = th.FloatTensor(weights).to(device)
    loss_fcn = nn.CrossEntropyLoss()
    # loss_fcn = nn.BCELoss(weight = class_weights)
    # loss_fcn = weighted_binary_cross_entropy()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])

    # Create PyTorch DataLoader for constructing blocks
    dataloader = get_dataloader(train_g, train_nid, sampler)

    # validata dataloader
    valdataloader = get_dataloader(val_g, val_nid, sampler)

    # Training loop
    valid_loss_min = np.Inf

    for epoch in range(args.num_epochs):

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        # tic_step = time.time()
        # epoch_loss = []
        model.train()

        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features of all the required input nodes as well as output labels of seeds node in a batch
            batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, device)

            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)

            loss = loss_fcn(batch_pred, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        train_acc, train_loss, train_f1 = al_evaluate(model, train_nfeat, train_labels, device, dataloader, loss_fcn)
        val_acc, val_loss, val_f1 = al_evaluate(model, val_nfeat, val_labels, device, valdataloader, loss_fcn)

        print('Epoch: {} \tTraining acc: {:.6f} \tValidation acc: {:.6f} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_acc,
            val_acc,
            train_loss,
            val_loss
            ))

        checkpoint = {
            'valid_loss_min': val_loss,
            'state_dict': model.state_dict(),
        }

        save_ckp(checkpoint, False, checkpoint_path, best_model_path)

        # TODO: save the model if validation loss has decreased

        if val_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, val_loss))
            save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = val_loss

    return train_acc, train_f1, val_acc, val_f1

# Testing
def run_test(args, device, data, best_model_path):

    # Unpacking the data
    n_classes, train_g, val_g, test_g, train_nfeat, train_labels, val_nfeat, val_labels, test_nfeat, test_labels, g = data

    in_feats = train_nfeat.shape[1]

    test_nid = th.nonzero(test_g.ndata['test_mask'], as_tuple=True)[0]
    # test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    dataloader_device = th.device('cpu')

    if args.sample_gpu:
        test_nid = test_nid.to(device)
        # copy only the csc to the GPU
        test_g = test_g.formats(['csc'])
        test_g = test_g.to(device)
        dataloader_device = device

    # define dataloader function
    def get_dataloader(train_g, train_nid, sampler):

        dataloader = dgl.dataloading.NodeDataLoader(
            train_g,
            train_nid,
            sampler,
            device=dataloader_device,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers)

        return dataloader

    # Define model and optimizer
    model = SAGE(in_feats, args.hidden_dim, n_classes, args.num_layers, F.relu, args.dropout)

    model = model.to(device)

    # == customize loss function
    # custom loss function
    # class_weights = th.FloatTensor(weights).to(device)
    loss_fcn = nn.CrossEntropyLoss()
    # loss_fcn = nn.BCELoss(weight = class_weights)
    # loss_fcn = weighted_binary_cross_entropy()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model, valid_loss_min = load_ckp(best_model_path, model)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

    # Create PyTorch DataLoader for constructing blocks
    dataloader = get_dataloader(test_g, test_nid, sampler)

    model.eval()

    test_acc, test_loss, test_f1 = al_evaluate_test(model, test_labels, device, dataloader, loss_fcn)

    print('Test acc: {:.6f} \tTess Loss: {:.6f}'.format(test_acc, test_loss))

    return test_acc, test_loss, test_f1
 
# evaluate node metrics for active selection
def node_information_metrics(graph):
    
    # Initializing the dictionary to store metrics
    nodeInfo_dict = {}
    # Degree centrality
    degCent_measures = nx.degree_centrality(graph)
    # PageRank centrality
    pgRnkCent_measures = nx.pagerank(graph)
    # Clustering coefficent
    clstCoeff_measures = nx.clustering(graph)
    # Eigenvector centrality
    eigCent_measures = nx.eigenvector_centrality(graph)
    # Betweenness centrality
    betCent_measures = nx.betweenness_centrality(graph)
    # CLoseness centrality
    clsCent_measures = nx.closeness_centrality(graph)
    # Aggregating all values
    for node in graph.nodes():
        # print (node)
        nodeInfo_dict[node] = {'degCent':degCent_measures[node], 'pgRnkCent':pgRnkCent_measures[node], 'clstCoeff':clstCoeff_measures[node], 'eigCent':eigCent_measures[node], 'betCent':betCent_measures[node], 'clsCent':clsCent_measures[node]}
    # Saving the metrics
    filepath = cnf.datapath + "\\nodeInfo_cora_deg_eig_bet_cls.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(nodeInfo_dict, f)
        
    return nodeInfo_dict

if __name__ == '__main__':
    # Arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=-1,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--n_class', type=int, default = 7)
    argparser.add_argument('--n_queries', type=int, default = 100)
    argparser.add_argument('--query_batch_size', type=int, default = 1)
    argparser.add_argument('--val_size_perclass', type=int, default = 2)
    argparser.add_argument('--test_size_perclass', type=int, default = 100)
    argparser.add_argument('--base_size_perclass', type=int, default = 5)
    argparser.add_argument('--queryStrategy', type=int, default = 1)

    argparser.add_argument('--num-epochs', type=int, default = 10)
    argparser.add_argument('--hidden_dim', type=int, default = 48)
    argparser.add_argument('--num-layers', type=int, default = 2)
    argparser.add_argument('--fan-out', type=str, default='8,10')
    argparser.add_argument('--batch-size', type=int, default = 5)
    argparser.add_argument('--log-every', type=int, default = 20)
    argparser.add_argument('--eval-every', type=int, default = 5)
    argparser.add_argument('--lr', type=float, default = 0.001)
    argparser.add_argument('--dropout', type=float, default = 0.1)
    argparser.add_argument('--num-workers', type=int, default = 4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--sample-gpu', action='store_true',
                           help="Perform the sampling process on the GPU. Must have 0 workers.")
    # argparser.add_argument('--inductive', action='store_true',
    #                        help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    args = argparser.parse_args()

    # Device initialization
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    # Loading the graph
    filepath = cnf.datapath + "\\cora_weighted.gpickle"
    graph_raw = nx.read_gpickle(filepath)
    
    # Define a mapping function to re-number node IDs
    mapping = dict(zip(graph_raw, range(0, len(graph_raw.nodes()))))
    # Renuumbering the nodes from 0, 1, 2, ...
    graph = nx.relabel_nodes(graph_raw, mapping)
    
    # Evaluating the node metrics
    # # This is required to be executed only once while loading a new graph
    # infoNodes = node_information_metrics(graph)
    
    # Load metrics corresponding to all nodes
    filepath = cnf.datapath + "\\nodeInfo_cora_deg_eig_bet_cls.pkl"
    with open(filepath, 'rb') as f:
        infoNodes_all = pickle.load(f)
        
    # Ranking nodes for active selection
    # nodeIDs_deg = sorted(infoNodes_all.items(), key=lambda x: x[1]['degCent'], reverse=True)
    # sorted_nodeIDs_deg = [ind[0] for ind in nodeIDs_deg]
    # nodeIDs_eig = sorted(infoNodes_all.items(), key=lambda x: x[1]['eigCent'], reverse=True)
    # sorted_nodeIDs_eig = [ind[0] for ind in nodeIDs_eig]
    # nodeIDs_bet = sorted(infoNodes_all.items(), key=lambda x: x[1]['betCent'], reverse=True)
    # sorted_nodeIDs_bet = [ind[0] for ind in nodeIDs_bet]
    # nodeIDs_cls = sorted(infoNodes_all.items(), key=lambda x: x[1]['clsCent'], reverse=True)
    # sorted_nodeIDs_cls = [ind[0] for ind in nodeIDs_cls]
    
    # Ranking nodes for active selection
    query_strategy_seq = args.queryStrategy
    if (query_strategy_seq == 1):
        query_strategy = 'Degree';
        nodeIDs_all = sorted(infoNodes_all.items(), key=lambda x: x[1]['degCent'], reverse=True)
        sorted_nodeIDs_all = [ind[0] for ind in nodeIDs_all]
    elif (query_strategy_seq == 2):
        query_strategy = 'PageRank';
        nodeIDs_all = sorted(infoNodes_all.items(), key=lambda x: x[1]['pgRnkCent'], reverse=True)
        sorted_nodeIDs_all = [ind[0] for ind in nodeIDs_all]
    elif (query_strategy_seq == 3):
        query_strategy = 'ClusteringCoefficient';
        nodeIDs_all = sorted(infoNodes_all.items(), key=lambda x: x[1]['clstCoeff'], reverse=True)
        sorted_nodeIDs_all = [ind[0] for ind in nodeIDs_all]
    elif (query_strategy_seq == 4):
        query_strategy = 'EigenVector';
        nodeIDs_all = sorted(infoNodes_all.items(), key=lambda x: x[1]['eigCent'], reverse=True)
        sorted_nodeIDs_all = [ind[0] for ind in nodeIDs_all]
    elif (query_strategy_seq == 5):
        query_strategy = 'Betweenness';
        nodeIDs_all = sorted(infoNodes_all.items(), key=lambda x: x[1]['betCent'], reverse=True)
        sorted_nodeIDs_all = [ind[0] for ind in nodeIDs_all]
    elif (query_strategy_seq == 6):
        query_strategy = 'Closeness';
        nodeIDs_all = sorted(infoNodes_all.items(), key=lambda x: x[1]['clsCent'], reverse=True)
        sorted_nodeIDs_all = [ind[0] for ind in nodeIDs_all]
    
    # Building DGL graph
    graph = nx.to_directed(graph)
    g = dgl.from_networkx(graph, node_attrs=['feature', 'label'])

    n_classes = args.n_class

    # features and labels in DGL graph
    g.ndata['feat'] = g.ndata['feature']
    g.ndata['label'] = g.ndata['label'].long()
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    # Getting initial labelled dataset, unlabelled pool; and forming validation and test masks
    array_lst_base_train, array_lst_avail_train = get_valtest_mask(g, n_classes, args.val_size_perclass, args.test_size_perclass, args.base_size_perclass)

    # Initializing variables to record performance
    train_size_list = []
    train_acc_list = []
    train_f1_list = []
    test_acc_list = []
    test_f1_list = []

    # Querying
    for cquery in range(args.n_queries+1):
        # No active selection for 0th query - just considering initial labelled dataset
        if cquery == 0:
            node_sel = []
        # Active Node selection for subsequent queries
        else:
            sorted_nodeIDs_all, node_sel = get_query(array_lst_avail_train, args.query_batch_size, sorted_nodeIDs_all)

        # Updating training mask - appending selected node to initial labelled dataset and removing it from the unlabelled pool
        array_lst_base_train, array_lst_avail_train = update_train_mask(g, n_classes, array_lst_base_train, array_lst_avail_train, node_sel)

        # Training preparation
        train_g, val_g, test_g = inductive_split(g)

        train_nfeat = train_g.ndata.pop('features')
        val_nfeat = val_g.ndata.pop('features')
        test_nfeat = test_g.ndata.pop('features')
        train_labels = train_g.ndata.pop('labels')
        val_labels = val_g.ndata.pop('labels')
        test_labels = test_g.ndata.pop('labels')

        print("No. of training, validation nodes:", train_nfeat.shape, val_nfeat.shape)
        print("Size of the input graph:", train_nfeat.shape)
        print("Query No.:", cquery)

        # Device manipulation
        if not args.data_cpu:
            train_nfeat = train_nfeat.to(device)
            val_nfeat = val_nfeat.to(device)
            test_nfeat = test_nfeat.to(device)
            train_labels = train_labels.to(device)
            val_labels = val_labels.to(device)
            test_labels = test_labels.to(device)

        # Pack data
        data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, val_nfeat, val_labels, test_nfeat, test_labels, g

        # Training model on the current labelled datset
        train_acc, train_f1, val_acc, val_f1 = run(args, device, data, cnf.modelpath + "\\current_checkpoint_cora.pt", cnf.modelpath + "\\cora_uc.pt")

        # Evaluating model on the test data
        test_acc, test_loss, test_f1 = run_test(args, device, data, cnf.modelpath + "\\cora_uc.pt")

        # Recording performance
        train_size_list.append(train_nfeat.shape[0])
        train_acc_list.append(train_acc)
        train_f1_list.append(train_f1)

        test_acc_list.append(test_acc)
        test_f1_list.append(test_f1)

    # Saving the results
    out_results = pd.DataFrame()
    out_results['train_size'] = train_size_list
    out_results['train_acc'] = train_acc_list
    out_results['train_f1'] = train_f1_list
    out_results['test_acc'] = test_acc_list
    out_results['test_f1'] = test_f1_list
    filepath = cnf.modelpath + "out_results_cora.csv"
    out_results.to_csv(filepath)











