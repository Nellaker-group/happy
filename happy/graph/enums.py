from enum import Enum


class FeatureArg(str, Enum):
    predictions = "predictions"
    embeddings = "embeddings"


class MethodArg(str, Enum):
    k = "k"
    delaunay = "delaunay"
    intersection = "intersection"


class SupervisedModelsArg(str, Enum):
    sup_graphsage = "sup_graphsage"
    sup_clustergcn = "sup_clustergcn"
    sup_clustergin  = "sup_clustergin"
    sup_clustergine  = "sup_clustergine"
    sup_clustergcne = "sup_clustergcne"
    sup_jumping = "sup_jumping"
    sup_graphsaint = "sup_graphsaint"
    sup_sign = "sup_sign"
    sup_shadow = "sup_shadow"
    sup_gat = "sup_gat"
    sup_gatv2 = "sup_gatv2"
    sup_graphunet = "sup_graphunet"
    sup_graphunet_noskip = "sup_graphunet_noskip"
    sup_mlp = "sup_mlp"
    sup_clustergcn_multilabel = "sup_clustergcn_multilabel"
    sup_clustergcn_multihead = "sup_clustergcn_multihead"


class GraphClassificationModelsArg(str, Enum):
    top_k = "top_k"
    sag = "sag"
    asap = "asap"

class AutoEncoderModelsArg(str, Enum):
    fps = "fps"
    fps_cosine = "fps_cosine"
    random = "random"
    random_cosine = "random_cosine"
    one_hop = "one_hop"
    one_hop_cosine = "one_hop_cosine"

class GCNLayerArg(str, Enum):
    gcn = "gcn"
    gine = "gine"
    gat = "gat"
