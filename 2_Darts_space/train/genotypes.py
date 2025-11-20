from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

# Weight-sharing NAS method
# 19 ICLR
DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
DARTS = DARTS_V2

# Training-Free NAS methods
# 24 ICLR
RoBoT  = Genotype(normal=[('dil_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 2), ('skip_connect', 0), ('max_pool_3x3', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 4), ('skip_connect', 3)], reduce_concat=range(2, 6))
# 24 ICLR
SWAP_C = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 4), ('sep_conv_5x5', 1)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 4), ('sep_conv_5x5', 1)], reduce_concat=[2, 3, 4, 5])

# Weight-sharing robust NAS methods
# 20 Arxiv
RACL = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 3), ('sep_conv_3x3', 3), ('skip_connect', 4)], normal_concat=[2, 3, 4, 5], reduce=[('sep_conv_3x3',0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5',1), ('sep_conv_3x3', 2), ('dil_conv_3x3', 3)], reduce_concat=[2, 3, 4, 5])
# 21 ICCV
ADVRUSH = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 2), ('skip_connect', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))
# 24 TNNLS
LRNAS  = Genotype(normal=[('max_pool_3x3', 1), ('avg_pool_3x3', 0),('sep_conv_3x3', 1), ('skip_connect', 0),('sep_conv_3x3', 2), ('avg_pool_3x3', 1),('max_pool_3x3', 3), ('max_pool_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('none', 0), ('none', 1),('sep_conv_5x5', 2), ('max_pool_3x3', 0),('avg_pool_3x3', 2), ('skip_connect', 1),('max_pool_3x3', 3), ('skip_connect', 4)], reduce_concat=[2, 3, 4, 5])
# 25 TKDE
REP_PGD = Genotype(normal=[('skip_connect', 1), ('dil_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('dil_conv_5x5', 1), ('dil_conv_3x3', 3), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('max_pool_3x3', 4), ('max_pool_3x3', 3)], reduce_concat=range(2, 6))

# Training-Free robust NAS methods
# 24 NIPS CRoZe
CROZE  = Genotype(normal=[('avg_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 2), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('dil_conv_3x3', 3), ('sep_conv_5x5', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 2), ('skip_connect', 3), ('sep_conv_7x7', 4)], normal_concat=[4, 5, 3, 4], reduce= [('avg_pool_3x3', 0), ('sep_conv_5x5', 1),('conv_7x1_1x7', 0), ('sep_conv_5x5', 1),('sep_conv_5x5', 2), ('conv_7x1_1x7', 0),('sep_conv_7x7', 1), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_3x3', 0), ('sep_conv_7x7', 1), ('conv_7x1_1x7', 2),('dil_conv_3x3', 3), ('sep_conv_3x3', 4)], reduce_concat=[3, 2, 5, 5])
# 25 ICLR ZCPRob
RNTK       = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1),('sep_conv_3x3', 0), ('dil_conv_5x5', 1),('skip_connect', 2), ('sep_conv_3x3', 3),('dil_conv_3x3', 2), ('skip_connect', 4)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 0), ('sep_conv_3x3', 1),('dil_conv_5x5', 2), ('sep_conv_3x3', 1),('avg_pool_3x3', 0), ('skip_connect', 3),('max_pool_3x3', 4), ('skip_connect', 1)], reduce_concat=[2, 3, 4, 5])

# Ours (Choose two good architectures)
TRNAS_c10      = Genotype(normal=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 3), ('skip_connect', 1), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3)], reduce_concat=range(2, 6))
TRNAS_imagenet = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0),('sep_conv_5x5', 2), ('sep_conv_5x5', 4), ('sep_conv_5x5', 0)], reduce_concat=range(2, 6))
