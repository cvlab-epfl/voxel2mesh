import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import math
from IPython import embed
from itertools import product, combinations

def adjacency_matrix(vertices, faces):
    B, N, D = vertices.shape

    halfedges = torch.tensor(list(combinations(range(D), 2)))
    edges = torch.cat([halfedges, torch.flip(halfedges,dims=[1])], dim=0)


    A = torch.zeros(1, N, N, device=faces.device)

    all_edges = faces[:, :, edges].long()
    all_edges = all_edges.view(1, -1, 2)
    A[0, all_edges[0, :, 0], all_edges[0, :, 1]] = 1 
    D = torch.diag(1 / torch.squeeze(torch.sum(A, dim=1)))[None]

    A = A.repeat(B, 1, 1)
    D = D.repeat(B, 1, 1)

    return A, D 
 


class GraphConv(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, batch_norm=False):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features)
        self.neighbours_fc = nn.Linear(in_features, out_features)

        self.bc = nn.BatchNorm1d(out_features) if batch_norm else Non()

    def forward(self, input, A, Dinv, vertices, faces):
        # coeff = torch.bmm(torch.bmm(Dsqrtinv, A), Dsqrtinv)
        coeff = torch.bmm(Dinv, A) # row normalization 

        y = self.fc(input)
        y_neightbours = torch.bmm(coeff, input)
        y_neightbours = self.neighbours_fc(y_neightbours)
 
 
        # y_neightbours = self.bc(y_neightbours.permute(0, 2, 1)).permute(0, 2, 1)
        y = y + y_neightbours
        # y = self.bc(y.permute(0, 2, 1)).permute(0, 2, 1)
        return y

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features is not None
        )
 

class GraphConvEdgeLengthWeighted(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, batch_norm=False):
        super(GraphConvEdgeLengthWeighted, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features)
        self.neighbours_fc = nn.Linear(in_features, out_features)

        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else Non()

        self.sigma = torch.nn.Parameter(0.1*torch.ones(1))

    def forward(self, input, A, Dinv, vertices, faces): 

        B, N, D = vertices.shape
        halfedges = torch.tensor(list(combinations(range(D), 2)))
        edges = torch.cat([halfedges, torch.flip(halfedges,dims=[1])], dim=0)


        dist = torch.zeros(1, N, N, device=faces.device)

        all_edges = faces[:, :, edges].long()
        all_edges = all_edges.view(1, -1, 2)

 
        all_edges = faces[:, :, edges].long()
        all_edges = all_edges.view(1, -1, 2)
        dist[0, all_edges[0, :, 0], all_edges[0, :, 1]] = torch.exp(-torch.sum((vertices[0, all_edges[0, :, 0]] - vertices[0, all_edges[0, :, 1]])**2,dim=1)/(self.sigma ** 2))
        dist = torch.clamp(dist, min=1e-7) 
        Dinv = (1 / (torch.squeeze(torch.sum(dist, dim=1))))[None,:,None] 
        coeff = Dinv * dist
 

        y = self.fc(input)
        y_neightbours = torch.bmm(coeff, input)
        y_neightbours = self.neighbours_fc(y_neightbours)

 
         
        y = y + y_neightbours
        # y = self.batch_norm(y.permute(0, 2, 1)).permute(0, 2, 1)
        return y

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features is not None
        )

 

class Feature2VertexLayer(nn.Module):

    def __init__(self, in_features, hidden_layer_count, batch_norm=False):
        super(Feature2VertexLayer, self).__init__()
        self.gconv = []
        for i in range(hidden_layer_count, 1, -1):
            self.gconv += [GraphConv(i * in_features // hidden_layer_count, (i-1) * in_features // hidden_layer_count, batch_norm)]
        self.gconv_layer = nn.Sequential(*self.gconv)
        self.gconv_last = GraphConv(in_features // hidden_layer_count, 3, batch_norm)

    def forward(self, features, adjacency_matrix, degree_matrix, vertices, faces):
        for gconv_hidden in self.gconv:
            features = F.relu(gconv_hidden(features, adjacency_matrix, degree_matrix,vertices,faces))
        return self.gconv_last(features, adjacency_matrix, degree_matrix,vertices,faces)

class Features2Features(nn.Module):

    def __init__(self, in_features, out_features, hidden_layer_count=2, graph_conv=GraphConv):
        super(Features2Features, self).__init__()

        self.gconv_first = graph_conv(in_features, out_features)
        gconv_hidden = []
        for i in range(hidden_layer_count):
            gconv_hidden += [graph_conv(out_features, out_features)]
        self.gconv_hidden = nn.Sequential(*gconv_hidden)
        self.gconv_last = graph_conv(out_features, out_features)

    def forward(self, features, adjacency_matrix, degree_matrix, vertices, faces):
        features = F.relu(self.gconv_first(features, adjacency_matrix, degree_matrix, vertices,faces))
        for gconv_hidden in self.gconv_hidden:
            features = F.relu(gconv_hidden(features, adjacency_matrix, degree_matrix, vertices,faces))
        return self.gconv_last(features, adjacency_matrix, degree_matrix, vertices, faces)


class Non(nn.Module):
    def __init__(self):
        super(Non, self).__init__()

    def forward(self, x):
        return x
 