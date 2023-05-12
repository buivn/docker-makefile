"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Dzung Bui
# DoC: 2022.04.02
# email: 
-----------------------------------------------------------------------------------
#
"""

import os

try: 
    env_var = os.environ['MAYAVI']
except KeyError:
    os.environ['MAYAVI'] = '/usr/bin/mayavi2'