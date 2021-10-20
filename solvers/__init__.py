#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 12:03:32 2021

@author: enfita
"""

from solvers.solvers import direct_solver, solve_bicg, solve_bicgstab,solve_cgs,\
    solve_gmres,solve_lgmres,solve_qmr
solvers = {"direct": direct_solver,          
           "bicgstab": solve_bicgstab,
           "bicg": solve_bicg,
           "cgs":solve_cgs,
           "gmres":solve_gmres,
           "gmres":solve_lgmres,
           "qmr":solve_qmr}