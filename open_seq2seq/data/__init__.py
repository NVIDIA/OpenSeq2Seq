# Copyright (c) 2017 NVIDIA Corporation
from .data_layer import DataLayer, MultiGPUWrapper
from .text2text.text2text import ParallelDataInRamInputLayer
from .speech2text import Speech2TextPlaceholdersDataLayer, Speech2TextDataLayer
