from operator import index

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


label  = pd.read_csv('labels.csv')
data = pd.read_csv('data.csv')

crossedtable = data.merge(label, left_index=True, right_index=True, how='inner')
#print(crossedtable)
BRCA_class = crossedtable[crossedtable['Class'] == 'BRCA']
#print(BRCA_class)
KIRC_class = crossedtable[crossedtable['Class'] == 'KIRC']
#print(KIRC_class)
COAD_class = crossedtable[crossedtable['Class'] == 'COAD']
#print(COAD_class)
LUAD_class = crossedtable[crossedtable['Class'] == 'LUAD']
#print(LUAD_class)
PRAD_class = crossedtable[crossedtable['Class'] == 'PRAD']
#print(PRAD_class)

########################turn into matrix########################
BRCA_Matrix = BRCA_class.drop(columns=BRCA_class.columns[BRCA_class.columns.str.startswith('Unnamed')])
BRCA_Matrix = BRCA_Matrix.drop(columns='Class')
#BRCA_Matrix = np.array(BRCA_Matrix)
#print(np.array(BRCA_Matrix))
#print(BRCA_Matrix)
KIRC_Matrix =  KIRC_class.drop(columns=KIRC_class.columns[KIRC_class.columns.str.startswith('Unnamed')])
KIRC_Matrix = KIRC_Matrix.drop(columns='Class')
#KIRC_Matrix = np.array(KIRC_Matrix)
#print(KIRC_Matrix)
COAD_Matrix = COAD_class.drop(columns=COAD_class.columns[COAD_class.columns.str.startswith('Unnamed')])
COAD_Matrix = COAD_Matrix.drop(columns='Class')
#COAD_Matrix = np.array(COAD_Matrix)
#print(COAD_Matrix)
LUAD_Matrix = LUAD_class.drop(columns=LUAD_class.columns[LUAD_class.columns.str.startswith('Unnamed')])
LUAD_Matrix = LUAD_Matrix.drop(columns='Class')
#LUAD_Matrix = np.array(LUAD_Matrix)
#print(LUAD_Matrix)
PRAD_Matrix = PRAD_class.drop(columns=PRAD_class.columns[PRAD_class.columns.str.startswith('Unnamed')])
PRAD_Matrix = PRAD_Matrix.drop(columns='Class')
#PRAD_Matrix = np.array(PRAD_Matrix)
#print(PRAD_Matrix)

########################CENTRE DES CLASSES######################
BRCA_Center = np.array(BRCA_Matrix.mean())
KIRC_Center = np.array(KIRC_Matrix.mean())
COAD_Center = np.array(COAD_Matrix.mean())
LUAD_Center = np.array(LUAD_Matrix.mean())
PRAD_Center = np.array(PRAD_Matrix.mean())
#print(BRCA_Center)
################################################################
################cohésion avec la intra classe###################
################################################################
tabDist = []

################################################################
#######################Euclidian distance#######################
################################################################


################COHÉSION CLASSE BRCA############################
for patient in np.array(BRCA_Matrix):
    dist = euclidean(patient, BRCA_Center)
    tabDist.append(dist)

BRCA_cohesion_eucludian =max(tabDist)
tabDist.clear()
###############################################################

################COHÉSION CLASSE KIRC############################
for patient in np.array(KIRC_Matrix):
    dist = euclidean(patient, KIRC_Center)
    tabDist.append(dist)

KIRC_cohesion_eucludian =max(tabDist)
tabDist.clear()
################################################################

################COHÉSION CLASSE COAD############################
for patient in np.array(PRAD_Matrix):
    dist = euclidean(patient, PRAD_Center)
    tabDist.append(dist)

PRAD_cohesion_eucludian =max(tabDist)
tabDist.clear()
################################################################

################COHÉSION CLASSE COAD############################
for patient in np.array(COAD_Matrix):
    dist = euclidean(patient, COAD_Center)
    tabDist.append(dist)

COAD_cohesion_eucludian =max(tabDist)
tabDist.clear()
################################################################


################COHÉSION CLASSE LUAD############################
for patient in np.array(LUAD_Matrix):
    dist = euclidean(patient, LUAD_Center)
    tabDist.append(dist)

LUAD_cohesion_eucludian =max(tabDist)
tabDist.clear()
################################################################

################COHÉSION CLASSE PRAD############################
for patient in np.array(PRAD_Matrix):
    dist = euclidean(patient, PRAD_Center)
    tabDist.append(dist)

PRAD_cohesion_eucludian =max(tabDist)
tabDist.clear()
################################################################

################################################################
###################Euclidian Distance Fin#######################
################################################################

# =======================Mahalanobis Distance=====================
#                       Avec N_GENES dimensions
# ==========================COVAVIANCE==============================

N_GENES_M = 20

genes_disponibles = BRCA_Matrix.columns

if len(genes_disponibles) < N_GENES_M:
    raise ValueError("Il y a moins de gènes disponibles que N_GENES_M.")

# Pour éviter une covariance singulière par manque de patients :
effectif_min = min(
    len(BRCA_Matrix),
    len(KIRC_Matrix),
    len(COAD_Matrix),
    len(LUAD_Matrix),
    len(PRAD_Matrix)
)

if N_GENES_M >= effectif_min:
    raise ValueError(
        f"N_GENES_M={N_GENES_M} est trop grand : "
        f"la plus petite classe a {effectif_min} patients. "
        "Diminue N_GENES_M."
    )

donnees_genes = data[genes_disponibles].apply(
    pd.to_numeric,
    errors="raise"
)

if donnees_genes.isna().any().any():
    raise ValueError("Des expressions génétiques sont manquantes.")

# Sélection des mêmes gènes pour toutes les classes Mahalanobis.
genes_M = donnees_genes.var(axis=0).nlargest(N_GENES_M).index.tolist()

print(f"Mahalanobis : {len(genes_M)} gènes utilisés")
print("Gènes sélectionnés :", genes_M)

BRCA_Matrix_M = BRCA_Matrix[genes_M].to_numpy(dtype=float)
KIRC_Matrix_M = KIRC_Matrix[genes_M].to_numpy(dtype=float)
COAD_Matrix_M = COAD_Matrix[genes_M].to_numpy(dtype=float)
LUAD_Matrix_M = LUAD_Matrix[genes_M].to_numpy(dtype=float)
PRAD_Matrix_M = PRAD_Matrix[genes_M].to_numpy(dtype=float)

# Centres calculés sur les mêmes gènes réduits.
BRCA_Center_M = BRCA_Matrix_M.mean(axis=0)
KIRC_Center_M = KIRC_Matrix_M.mean(axis=0)
COAD_Center_M = COAD_Matrix_M.mean(axis=0)
LUAD_Center_M = LUAD_Matrix_M.mean(axis=0)
PRAD_Center_M = PRAD_Matrix_M.mean(axis=0)

# Cov_BRCA = np.cov(np.array(BRCA_Matrix),rowvar=False)
# Cov_KIRC = np.cov(np.array(KIRC_Matrix),rowvar=False)
# Cov_COAD = np.cov(np.array(COAD_Matrix),rowvar=False)
# Cov_LUAD = np.cov(np.array(LUAD_Matrix),rowvar=False)
# Cov_PRAD = np.cov(np.array(PRAD_Matrix),rowvar=False)
# # print(1)
# # #######################INV COVARIANCE###########################
# Inv_cov_BRCA = np.linalg.pinv(Cov_BRCA)
# Inv_cov_KIRC = np.linalg.pinv(Cov_KIRC)
# Inv_cov_COAD = np.linalg.pinv(Cov_COAD)
# Inv_cov_LUAD = np.linalg.pinv(Cov_LUAD)
# Inv_cov_PRAD = np.linalg.pinv(Cov_PRAD)

# Covariances réduites : matrices N_GENES_M x N_GENES_M.
Cov_BRCA = np.cov(BRCA_Matrix_M, rowvar=False)
Cov_KIRC = np.cov(KIRC_Matrix_M, rowvar=False)
Cov_COAD = np.cov(COAD_Matrix_M, rowvar=False)
Cov_LUAD = np.cov(LUAD_Matrix_M, rowvar=False)
Cov_PRAD = np.cov(PRAD_Matrix_M, rowvar=False)

Inv_cov_BRCA = np.linalg.pinv(Cov_BRCA)
Inv_cov_KIRC = np.linalg.pinv(Cov_KIRC)
Inv_cov_COAD = np.linalg.pinv(Cov_COAD)
Inv_cov_LUAD = np.linalg.pinv(Cov_LUAD)
Inv_cov_PRAD = np.linalg.pinv(Cov_PRAD)

# print("BRCA complète :", BRCA_Matrix.shape)
# print("BRCA Mahalanobis :", BRCA_Matrix_M.shape)
# print("Covariance BRCA :", Cov_BRCA.shape)
# # ###################COHÉSION CLASSE BRCA#########################
# for patient in np.array(BRCA_Matrix):
#       dist = distance.mahalanobis(patient, BRCA_Center,Inv_cov_BRCA)
#       tabDist.append(dist)
#
# BRCA_cohesion_m =max(tabDist)
# tabDist.clear()
# # # ################COHÉSION CLASSE KIRC############################
# for patient in np.array(KIRC_Matrix):
#      dist = distance.mahalanobis(patient, KIRC_Center,Inv_cov_KIRC)
#      tabDist.append(dist)
# #
# KIRC_cohesion_m =max(tabDist)
# tabDist.clear()
# # # ################################################################
# # #
# # # ################COHÉSION CLASSE PRAD############################
# for patient in np.array(PRAD_Matrix):
#       dist = distance.mahalanobis(patient, PRAD_Center,Inv_cov_PRAD)
#       tabDist.append(dist)
# #
# PRAD_cohesion_m =max(tabDist)
# tabDist.clear()
# # # ################################################################
# # #
# # # ################COHÉSION CLASSE COAD############################
# for patient in np.array(COAD_Matrix):
#       dist = distance.mahalanobis(patient, COAD_Center,Inv_cov_COAD)
#       tabDist.append(dist)
# #
# COAD_cohesion_m =max(tabDist)
# tabDist.clear()
# # # ################################################################
# # #
# # #
# # # ################COHÉSION CLASSE LUAD############################
# for patient in np.array(LUAD_Matrix):
#       dist = distance.mahalanobis(patient, LUAD_Center,Inv_cov_LUAD)
#       tabDist.append(dist)
# #
# LUAD_cohesion_m =max(tabDist)
# tabDist.clear()
# # # ################################################################
# # #
# # # ################COHÉSION CLASSE PRAD############################
# for patient in np.array(PRAD_Matrix):
#       dist = distance.mahalanobis(patient, PRAD_Center,Inv_cov_PRAD)
#       tabDist.append(dist)
# #
# PRAD_cohesion_m =max(tabDist)
# tabDist.clear()

def cohesion_mahalanobis(matrice, centre, inv_cov):
    return max(
        distance.mahalanobis(patient, centre, inv_cov)
        for patient in matrice
    )

BRCA_cohesion_m = cohesion_mahalanobis(
    BRCA_Matrix_M, BRCA_Center_M, Inv_cov_BRCA
)
KIRC_cohesion_m = cohesion_mahalanobis(
    KIRC_Matrix_M, KIRC_Center_M, Inv_cov_KIRC
)
COAD_cohesion_m = cohesion_mahalanobis(
    COAD_Matrix_M, COAD_Center_M, Inv_cov_COAD
)
LUAD_cohesion_m = cohesion_mahalanobis(
    LUAD_Matrix_M, LUAD_Center_M, Inv_cov_LUAD
)
PRAD_cohesion_m = cohesion_mahalanobis(
    PRAD_Matrix_M, PRAD_Center_M, Inv_cov_PRAD
)
# # ############################################################

################################################################
###################mahalanobis Distance Fin#####################
################################################################


# ###################cosine Distance############################
################COHÉSION CLASSE BRCA############################
for patient in np.array(BRCA_Matrix):
    dist = distance.cosine(patient, BRCA_Center)
    tabDist.append(dist)

BRCA_cohesion_cosine =max(tabDist)
tabDist.clear()
###############################################################

################COHÉSION CLASSE KIRC############################
for patient in np.array(KIRC_Matrix):
    dist = distance.cosine(patient, KIRC_Center)
    tabDist.append(dist)

KIRC_cohesion_cosine =max(tabDist)
tabDist.clear()
################################################################

################COHÉSION CLASSE PRAD############################
for patient in np.array(PRAD_Matrix):
    dist = distance.cosine(patient, PRAD_Center)
    tabDist.append(dist)

PRAD_cohesion_cosine =max(tabDist)
tabDist.clear()
################################################################

################COHÉSION CLASSE COAD############################
for patient in np.array(COAD_Matrix):
    dist = distance.cosine(patient, COAD_Center)
    tabDist.append(dist)

COAD_cohesion_cosine =max(tabDist)
tabDist.clear()
################################################################


################COHÉSION CLASSE LUAD############################
for patient in np.array(LUAD_Matrix):
    dist = distance.cosine(patient, LUAD_Center)
    tabDist.append(dist)

LUAD_cohesion_cosine =max(tabDist)
tabDist.clear()
################################################################

################COHÉSION CLASSE PRAD############################
for patient in np.array(PRAD_Matrix):
    dist = distance.cosine(patient, PRAD_Center)
    tabDist.append(dist)

PRAD_cohesion_cosine =max(tabDist)
tabDist.clear()
###########################################################
###################cosine Distance Fin#####################
###########################################################



###################Distance inter classe##################
tabDistC1 = []
tabDistC2 = []
##################separation BRCA#########################
#####################KIRC#################################

for patient in np.array(BRCA_Matrix):
    dist = euclidean(patient, KIRC_Center)
    tabDistC1.append(dist)

for patient in np.array(KIRC_Matrix):
    dist = euclidean(patient, BRCA_Center)
    tabDistC2.append(dist)

res_BRCA_KIRC_euclidian = min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()

##########################################################

#####################COAD###############################
for patient in np.array(BRCA_Matrix):
    dist = euclidean(patient, COAD_Center)
    tabDistC1.append(dist)

for patient in np.array(COAD_Matrix):
    dist = euclidean(patient, BRCA_Center)
    tabDistC2.append(dist)

res_BRCA_COAD_euclidian = min(min(tabDistC1), min(tabDistC2))

tabDistC1.clear()
tabDistC2.clear()
#######################################################

######################Luad#############################
for patient in np.array(BRCA_Matrix):
    dist = euclidean(patient, LUAD_Center)
    tabDistC1.append(dist)

for patient in np.array(LUAD_Matrix):
    dist = euclidean(patient, BRCA_Center)
    tabDistC2.append(dist)

res_BRCA_Luad_euclidian = min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()
#######################################################

#######################PRAD############################
for patient in np.array(BRCA_Matrix):
    dist = euclidean(patient, PRAD_Center)
    tabDistC1.append(dist)

for patient in np.array(PRAD_Matrix):
    dist = euclidean(patient, BRCA_Center)
    tabDistC2.append(dist)

res_BRCA_Prad_euclidian = min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()
########################################################
##################Separation KIRC######################
#####################coad############################

for patient in np.array(KIRC_Matrix):
    dist = euclidean(patient, COAD_Center)
    tabDistC1.append(dist)

for patient in np.array(COAD_Matrix):
    dist = euclidean(patient, KIRC_Center)
    tabDistC2.append(dist)

res_KIRC_COAD_euclidian = min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()
#######################################################

#####################LUAD############################

for patient in np.array(KIRC_Matrix):
    dist = euclidean(patient, LUAD_Center)
    tabDistC1.append(dist)

for patient in np.array(LUAD_Matrix):
    dist = euclidean(patient, KIRC_Center)
    tabDistC2.append(dist)

res_KIRC_LUAD_euclidian = min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()
#######################################################
#####################PRAD############################

for patient in np.array(KIRC_Matrix):
    dist = euclidean(patient, PRAD_Center)
    tabDistC1.append(dist)

for patient in np.array(PRAD_Matrix):
    dist = euclidean(patient, KIRC_Center)
    tabDistC2.append(dist)

res_KIRC_PRAD_euclidian = min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()
#######################################################

######################separation COAD##################
#####################LUAD############################

for patient in np.array(COAD_Matrix):
    dist = euclidean(patient, LUAD_Center)
    tabDistC1.append(dist)

for patient in np.array(LUAD_Matrix):
    dist = euclidean(patient, COAD_Center)
    tabDistC2.append(dist)

res_COAD_LUAD_euclidian = min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()
#######################################################

#####################PRAD############################

for patient in np.array(COAD_Matrix):
    dist = euclidean(patient, PRAD_Center)
    tabDistC1.append(dist)

for patient in np.array(PRAD_Matrix):
    dist = euclidean(patient, COAD_Center)
    tabDistC2.append(dist)

res_COAD_PRAD_euclidian= min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()
#######################################################

#######################separation LUAD################
##########################Prad######################
for patient in np.array(LUAD_Matrix):
    dist = euclidean(patient, PRAD_Center)
    tabDistC1.append(dist)

for patient in np.array(PRAD_Matrix):
    dist = euclidean(patient, LUAD_Center)
    tabDistC2.append(dist)

res_LUAD_PRAD_euclidian= min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()
###################################################

#################separation mahalanobis############
###################separation BRCA##############
# for patient in np.array(BRCA_Matrix):
#     dist1 = distance.mahalanobis(patient, KIRC_Center,Inv_cov_KIRC )
#     tabDistC1.append(dist1)
#
# for patient in np.array(KIRC_Matrix):
#     dist2 = distance.mahalanobis(patient, BRCA_Center,Inv_cov_BRCA)
#     tabDistC2.append(dist2)
#
# res_BRCA_KIRC_m = min(min(tabDistC1), min(tabDistC2))
# tabDistC1.clear()
# tabDistC2.clear()
#
# ##########################################################
#
# #####################COAD###############################
# for patient in np.array(BRCA_Matrix):
#     dist = distance.mahalanobis(patient, COAD_Center,Inv_cov_COAD)
#     tabDistC1.append(dist)
#
# for patient in np.array(COAD_Matrix):
#     dist = distance.mahalanobis(patient, BRCA_Center,Inv_cov_BRCA)
#     tabDistC2.append(dist)
#
# res_BRCA_COAD_m = min(min(tabDistC1), min(tabDistC2))
#
# tabDistC1.clear()
# tabDistC2.clear()
# #######################################################
#
# ######################Luad#############################
# for patient in np.array(BRCA_Matrix):
#     dist = distance.mahalanobis(patient, LUAD_Center,Inv_cov_LUAD)
#     tabDistC1.append(dist)
#
# for patient in np.array(LUAD_Matrix):
#     dist = distance.mahalanobis(patient, BRCA_Center,Inv_cov_BRCA)
#     tabDistC2.append(dist)
#
# res_BRCA_Luad_m = min(min(tabDistC1), min(tabDistC2))
# tabDistC1.clear()
# tabDistC2.clear()
# #######################################################
#
# #######################PRAD############################
# for patient in np.array(BRCA_Matrix):
#     dist = distance.mahalanobis(patient, PRAD_Center,Inv_cov_PRAD)
#     tabDistC1.append(dist)
#
# for patient in np.array(PRAD_Matrix):
#     dist = distance.mahalanobis(patient, BRCA_Center,Inv_cov_BRCA)
#     tabDistC2.append(dist)
#
# res_BRCA_Prad_m = min(min(tabDistC1), min(tabDistC2))
# tabDistC1.clear()
# tabDistC2.clear()
# ########################################################
# ##################Separation KIRC######################
# #####################coad############################
#
# for patient in np.array(KIRC_Matrix):
#     dist = distance.mahalanobis(patient, COAD_Center,Inv_cov_COAD)
#     tabDistC1.append(dist)
#
# for patient in np.array(COAD_Matrix):
#     dist = distance.mahalanobis(patient, KIRC_Center,Inv_cov_KIRC)
#     tabDistC2.append(dist)
#
# res_KIRC_COAD_m = min(min(tabDistC1), min(tabDistC2))
# tabDistC1.clear()
# tabDistC2.clear()
# #######################################################
#
# #####################LUAD############################
#
# for patient in np.array(KIRC_Matrix):
#     dist = distance.mahalanobis(patient, LUAD_Center,Inv_cov_LUAD)
#     tabDistC1.append(dist)
#
# for patient in np.array(LUAD_Matrix):
#     dist = distance.mahalanobis(patient, KIRC_Center,Inv_cov_KIRC)
#     tabDistC2.append(dist)
#
# res_KIRC_LUAD_m = min(min(tabDistC1), min(tabDistC2))
# tabDistC1.clear()
# tabDistC2.clear()
# #######################################################
# #####################PRAD############################
#
# for patient in np.array(KIRC_Matrix):
#     dist = distance.mahalanobis(patient, PRAD_Center,Inv_cov_PRAD)
#     tabDistC1.append(dist)
#
# for patient in np.array(PRAD_Matrix):
#     dist = distance.mahalanobis(patient, KIRC_Center,Inv_cov_KIRC)
#     tabDistC2.append(dist)
#
# res_KIRC_PRAD_m = min(min(tabDistC1), min(tabDistC2))
# tabDistC1.clear()
# tabDistC2.clear()
# #######################################################
#
# ######################separation COAD##################
# #####################LUAD############################
#
# for patient in np.array(COAD_Matrix):
#     dist = distance.mahalanobis(patient, LUAD_Center,Inv_cov_LUAD)
#     tabDistC1.append(dist)
#
# for patient in np.array(LUAD_Matrix):
#     dist = distance.mahalanobis(patient, COAD_Center,Inv_cov_COAD)
#     tabDistC2.append(dist)
#
# res_COAD_LUAD_m = min(min(tabDistC1), min(tabDistC2))
# tabDistC1.clear()
# tabDistC2.clear()
# #######################################################
#
# #####################PRAD############################
#
# for patient in np.array(COAD_Matrix):
#     dist = distance.mahalanobis(patient, PRAD_Center, Inv_cov_PRAD)
#     tabDistC1.append(dist)
#
# for patient in np.array(PRAD_Matrix):
#     dist = distance.mahalanobis(patient, COAD_Center,Inv_cov_COAD)
#     tabDistC2.append(dist)
#
# res_COAD_PRAD_m= min(min(tabDistC1), min(tabDistC2))
# tabDistC1.clear()
# tabDistC2.clear()
# #######################################################
#
# #######################separation LUAD################
# ##########################Prad######################
# for patient in np.array(LUAD_Matrix):
#     dist = distance.mahalanobis(patient, PRAD_Center,Inv_cov_PRAD)
#     tabDistC1.append(dist)
#
# for patient in np.array(PRAD_Matrix):
#     dist = distance.mahalanobis(patient, LUAD_Center,Inv_cov_LUAD)
#     tabDistC2.append(dist)
#
# res_LUAD_PRAD_m= min(min(tabDistC1), min(tabDistC2))
# tabDistC1.clear()
# tabDistC2.clear()

def distance_inter_mahalanobis(
    matrice_1, centre_1, inv_cov_1,
    matrice_2, centre_2, inv_cov_2
):
    # Patients de C1 vers le centre de C2 :
    # covariance de la classe cible C2.
    minimum_1_vers_2 = min(
        distance.mahalanobis(patient, centre_2, inv_cov_2)
        for patient in matrice_1
    )

    # Patients de C2 vers le centre de C1 :
    # covariance de la classe cible C1.
    minimum_2_vers_1 = min(
        distance.mahalanobis(patient, centre_1, inv_cov_1)
        for patient in matrice_2
    )

    return min(minimum_1_vers_2, minimum_2_vers_1)


res_BRCA_KIRC_m = distance_inter_mahalanobis(
    BRCA_Matrix_M, BRCA_Center_M, Inv_cov_BRCA,
    KIRC_Matrix_M, KIRC_Center_M, Inv_cov_KIRC
)

res_BRCA_COAD_m = distance_inter_mahalanobis(
    BRCA_Matrix_M, BRCA_Center_M, Inv_cov_BRCA,
    COAD_Matrix_M, COAD_Center_M, Inv_cov_COAD
)

res_BRCA_Luad_m = distance_inter_mahalanobis(
    BRCA_Matrix_M, BRCA_Center_M, Inv_cov_BRCA,
    LUAD_Matrix_M, LUAD_Center_M, Inv_cov_LUAD
)

res_BRCA_Prad_m = distance_inter_mahalanobis(
    BRCA_Matrix_M, BRCA_Center_M, Inv_cov_BRCA,
    PRAD_Matrix_M, PRAD_Center_M, Inv_cov_PRAD
)

res_KIRC_COAD_m = distance_inter_mahalanobis(
    KIRC_Matrix_M, KIRC_Center_M, Inv_cov_KIRC,
    COAD_Matrix_M, COAD_Center_M, Inv_cov_COAD
)

res_KIRC_LUAD_m = distance_inter_mahalanobis(
    KIRC_Matrix_M, KIRC_Center_M, Inv_cov_KIRC,
    LUAD_Matrix_M, LUAD_Center_M, Inv_cov_LUAD
)

res_KIRC_PRAD_m = distance_inter_mahalanobis(
    KIRC_Matrix_M, KIRC_Center_M, Inv_cov_KIRC,
    PRAD_Matrix_M, PRAD_Center_M, Inv_cov_PRAD
)

res_COAD_LUAD_m = distance_inter_mahalanobis(
    COAD_Matrix_M, COAD_Center_M, Inv_cov_COAD,
    LUAD_Matrix_M, LUAD_Center_M, Inv_cov_LUAD
)

res_COAD_PRAD_m = distance_inter_mahalanobis(
    COAD_Matrix_M, COAD_Center_M, Inv_cov_COAD,
    PRAD_Matrix_M, PRAD_Center_M, Inv_cov_PRAD
)

res_LUAD_PRAD_m = distance_inter_mahalanobis(
    LUAD_Matrix_M, LUAD_Center_M, Inv_cov_LUAD,
    PRAD_Matrix_M, PRAD_Center_M, Inv_cov_PRAD
)
###################################################
#####################################################


#######################distance cosinus###############
##################separation BCRA#########################
#####################KIRC#################################

for patient in np.array(BRCA_Matrix):
    dist = distance.cosine(patient, KIRC_Center)
    tabDistC1.append(dist)

for patient in np.array(KIRC_Matrix):
    dist = distance.cosine(patient, BRCA_Center)
    tabDistC2.append(dist)

res_BRCA_KIRC_cosine = min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()

##########################################################

#####################COAD###############################
for patient in np.array(BRCA_Matrix):
    dist = distance.cosine(patient, COAD_Center)
    tabDistC1.append(dist)

for patient in np.array(COAD_Matrix):
    dist = distance.cosine(patient, BRCA_Center)
    tabDistC2.append(dist)

res_BRCA_COAD_cosine = min(min(tabDistC1), min(tabDistC2))

tabDistC1.clear()
tabDistC2.clear()
#######################################################

######################Luad#############################
for patient in np.array(BRCA_Matrix):
    dist = distance.cosine(patient, LUAD_Center)
    tabDistC1.append(dist)

for patient in np.array(LUAD_Matrix):
    dist = distance.cosine(patient, BRCA_Center)
    tabDistC2.append(dist)

res_BRCA_Luad_cosine = min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()
#######################################################

#######################PRAD############################
for patient in np.array(BRCA_Matrix):
    dist = distance.cosine(patient, PRAD_Center)
    tabDistC1.append(dist)

for patient in np.array(PRAD_Matrix):
    dist = distance.cosine(patient, BRCA_Center)
    tabDistC2.append(dist)

res_BRCA_Prad_cosine = min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()
########################################################
##################Separation KIRC######################
#####################coad############################

for patient in np.array(KIRC_Matrix):
    dist = distance.cosine(patient, COAD_Center)
    tabDistC1.append(dist)

for patient in np.array(COAD_Matrix):
    dist = distance.cosine(patient, KIRC_Center)
    tabDistC2.append(dist)

res_KIRC_COAD_cosine = min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()
#######################################################

#####################LUAD############################

for patient in np.array(KIRC_Matrix):
    dist = distance.cosine(patient, LUAD_Center)
    tabDistC1.append(dist)

for patient in np.array(LUAD_Matrix):
    dist = distance.cosine(patient, KIRC_Center)
    tabDistC2.append(dist)

res_KIRC_LUAD_cosine = min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()
#######################################################
#####################PRAD############################

for patient in np.array(KIRC_Matrix):
    dist = distance.cosine(patient, PRAD_Center)
    tabDistC1.append(dist)

for patient in np.array(PRAD_Matrix):
    dist = distance.cosine(patient, KIRC_Center)
    tabDistC2.append(dist)

res_KIRC_PRAD_cosine = min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()
#######################################################

######################separation COAD##################
#####################LUAD############################

for patient in np.array(COAD_Matrix):
    dist = distance.cosine(patient, LUAD_Center)
    tabDistC1.append(dist)

for patient in np.array(LUAD_Matrix):
    dist = distance.cosine(patient, COAD_Center)
    tabDistC2.append(dist)

res_COAD_LUAD_cosine = min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()
#######################################################

#####################PRAD############################

for patient in np.array(COAD_Matrix):
    dist = distance.cosine(patient, PRAD_Center)
    tabDistC1.append(dist)

for patient in np.array(PRAD_Matrix):
    dist = distance.cosine(patient, COAD_Center)
    tabDistC2.append(dist)

res_COAD_PRAD_cosine= min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()
#######################################################

#######################separation LUAD################
##########################Prad######################
for patient in np.array(LUAD_Matrix):
    dist = distance.cosine(patient, PRAD_Center)
    tabDistC1.append(dist)

for patient in np.array(PRAD_Matrix):
    dist = distance.cosine(patient, LUAD_Center)
    tabDistC2.append(dist)

res_LUAD_PRAD_cosine= min(min(tabDistC1), min(tabDistC2))
tabDistC1.clear()
tabDistC2.clear()
###################################################


#####################OVERLAP########################

###########################Euclidian###############

######################BRCA - KIRC###################
overlap_BRCA_KIRC = ( BRCA_cohesion_eucludian + KIRC_cohesion_eucludian)/ (2* res_BRCA_KIRC_euclidian)
####################################################

######################BRCA -COAD#####################
overlap_BRCA_COAD = ( BRCA_cohesion_eucludian + COAD_cohesion_eucludian)/ (2* res_BRCA_COAD_euclidian)
#####################################################

######################BRCA -LUAD#####################
overlap_BRCA_LUAD = ( BRCA_cohesion_eucludian + LUAD_cohesion_eucludian)/ (2* res_BRCA_Luad_euclidian)
#####################################################


#########################BRCA -PRAD##################
overlap_BRCA_PRAD = ( BRCA_cohesion_eucludian + PRAD_cohesion_eucludian)/ (2* res_BRCA_Prad_euclidian)
#####################################################

#########################KIRC -COAD##################
overlap_KIRC_COAD = ( KIRC_cohesion_eucludian + COAD_cohesion_eucludian)/ (2* res_KIRC_COAD_euclidian)
#####################################################

########################KIRC - LUAD##################
overlap_KIRC_LUAD = ( KIRC_cohesion_eucludian + LUAD_cohesion_eucludian)/ (2* res_KIRC_LUAD_euclidian)
#####################################################

#######################KIRC - PRAD###################
overlap_KIRC_PRAD = ( KIRC_cohesion_eucludian + PRAD_cohesion_eucludian)/ (2* res_KIRC_PRAD_euclidian)
#####################################################


####################COAD -LUAD######################
overlap_COAD_LUAD = ( COAD_cohesion_eucludian + LUAD_cohesion_eucludian)/ (2* res_COAD_LUAD_euclidian)
####################################################

##################COAD-PRAD#########################
overlap_COAD_PRAD = ( COAD_cohesion_eucludian + PRAD_cohesion_eucludian)/ (2* res_COAD_PRAD_euclidian)
###################################################

###########################LUAD-PRAD##############
overlap_LUAD_PRAD = ( PRAD_cohesion_eucludian + LUAD_cohesion_eucludian)/ (2* res_LUAD_PRAD_euclidian)
#################################################





#########################mahalanobis#################
######################BRCA - KIRC###################
overlap_BRCA_KIRC_m = ( BRCA_cohesion_m+ KIRC_cohesion_m)/ (2* res_BRCA_KIRC_m)
####################################################

######################BRCA -COAD#####################
overlap_BRCA_COAD_m = ( BRCA_cohesion_m+ COAD_cohesion_m)/ (2* res_BRCA_COAD_m)
#####################################################

######################BRCA -LUAD#####################
overlap_BRCA_LUAD_m = ( BRCA_cohesion_m + LUAD_cohesion_m)/ (2* res_BRCA_Luad_m)
#####################################################


#########################BRCA -PRAD##################
overlap_BRCA_PRAD_m = ( BRCA_cohesion_m + PRAD_cohesion_m)/ (2* res_BRCA_Prad_m)
#####################################################

#########################KIRC -COAD##################
overlap_KIRC_COAD_m = ( KIRC_cohesion_m + COAD_cohesion_m)/ (2* res_KIRC_COAD_m)
#####################################################

########################KIRC - LUAD##################
overlap_KIRC_LUAD_m = ( KIRC_cohesion_m + LUAD_cohesion_m)/ (2* res_KIRC_LUAD_m)
#####################################################

#######################KIRC - PRAD###################
overlap_KIRC_PRAD_m = ( KIRC_cohesion_m + PRAD_cohesion_m)/ (2* res_KIRC_PRAD_m)
#####################################################


####################COAD -LUAD######################
overlap_COAD_LUAD_m = ( COAD_cohesion_m + LUAD_cohesion_m)/ (2* res_COAD_LUAD_m)
####################################################

##################COAD-PRAD#########################
overlap_COAD_PRAD_m = ( COAD_cohesion_m + PRAD_cohesion_m)/ (2* res_COAD_PRAD_m)
###################################################

###########################LUAD-PRAD##############
overlap_LUAD_PRAD_m = ( PRAD_cohesion_m + LUAD_cohesion_m)/ (2* res_LUAD_PRAD_m)
#################################################
####################a faire #########################
######################################################

#######################cosinus######################
######################BRCA - KIRC###################
overlap_BRCA_KIRC_cosine = ( BRCA_cohesion_cosine + KIRC_cohesion_cosine)/ (2* res_BRCA_KIRC_cosine)
####################################################

######################BRCA -COAD#####################
overlap_BRCA_COAD_cosine = ( BRCA_cohesion_cosine+ COAD_cohesion_cosine)/ (2* res_BRCA_COAD_cosine)
#####################################################

######################BRCA -LUAD#####################
overlap_BRCA_LUAD_cosine = ( BRCA_cohesion_cosine + LUAD_cohesion_cosine)/ (2* res_BRCA_Luad_cosine)
#####################################################


#########################BRCA -PRAD##################
overlap_BRCA_PRAD_cosine = ( BRCA_cohesion_cosine + PRAD_cohesion_cosine)/ (2* res_BRCA_Prad_cosine)
#####################################################

#########################KIRC -COAD##################
overlap_KIRC_COAD_cosine = ( KIRC_cohesion_cosine + COAD_cohesion_cosine)/ (2* res_KIRC_COAD_cosine)
#####################################################

########################KIRC - LUAD##################
overlap_KIRC_LUAD_cosine = ( KIRC_cohesion_cosine + LUAD_cohesion_cosine)/ (2* res_KIRC_LUAD_cosine)
#####################################################

#######################KIRC - PRAD###################
overlap_KIRC_PRAD_cosine = ( KIRC_cohesion_cosine + PRAD_cohesion_cosine)/ (2* res_KIRC_PRAD_cosine)
#####################################################


####################COAD -LUAD######################
overlap_COAD_LUAD_cosine = ( COAD_cohesion_cosine + LUAD_cohesion_cosine)/ (2* res_COAD_LUAD_cosine)
####################################################

##################COAD-PRAD#########################
overlap_COAD_PRAD_cosine = ( COAD_cohesion_cosine + PRAD_cohesion_cosine)/ (2* res_COAD_PRAD_cosine)
###################################################

###########################LUAD-PRAD##############
overlap_LUAD_PRAD_cosine = ( PRAD_cohesion_cosine + LUAD_cohesion_cosine)/ (2* res_LUAD_PRAD_cosine)
#################################################

# print("Centre BRCA complet :", BRCA_Center.shape)
# print("Centre BRCA Mahalanobis :", BRCA_Center_M.shape)
# print("Inverse covariance BRCA :", Inv_cov_BRCA.shape)
#
# print("Cohésion BRCA Mahalanobis :", BRCA_cohesion_m)
# print("Distance BRCA–KIRC Mahalanobis :", res_BRCA_KIRC_m)
# print("Overlap BRCA–KIRC Mahalanobis :", overlap_BRCA_KIRC_m)

################################### Methode 2 ################################
#  a) et b)
###############################################################################
colonnes_unnamed = [
    colonne for colonne in data.columns
    if colonne.startswith("Unnamed")
]

data = data.drop(columns=colonnes_unnamed)

classes_attendues = ["BRCA","KIRC","COAD","LUAD","PRAD"]
classes_observees = sorted(label["Class"].unique())

#print("Classes observées :", classes_observees)
# for classe in classes_attendues:
#     nombre = (label["Class"] == classe).sum()
#     print(f"{classe} : {nombre} patients")

#
# crossedtable2 = data.merge(label[["Class"]],left_index=True,right_index=True,how="inner")
# print("Dimensions de crossedtable :", crossedtable.shape)
# print("Dimensions de crossedtable 2 :", crossedtable2.shape)

######## a) choix d'une paire globale de variables parmi les gènes selectionnés pour Mahalanobis
gene_x = "gene_439"
gene_y = "gene_9176"

donnees_numeric = crossedtable[[gene_x, gene_y, "Class"]].copy()

donnees_numeric[gene_x] = pd.to_numeric(
    donnees_numeric[gene_x],
    errors="raise"
)

donnees_numeric[gene_y] = pd.to_numeric(
    donnees_numeric[gene_y],
    errors="raise"
)
#
if donnees_numeric[[gene_x, gene_y]].isna().any().any():
    raise ValueError(
        "Variables sélectionnées contiennent des valeurs manquantes."
    )


palette_cancers = ["blue", "red", "green", "orange", "purple"]
##################  b) Nuage des points ###################################
plt.figure(figsize=(11, 8))

sns.scatterplot(
    data=donnees_numeric,
    x=gene_x,
    y=gene_y,
    hue="Class",
    hue_order=classes_attendues,
    palette=palette_cancers,
    s=45,
    alpha=0.75,
    edgecolor=None
)

plt.title(
    "Méthode 2b - Nuage de points : "
    f"{gene_x} et {gene_y}"
)

plt.xlabel(f"Expression de {gene_x}")
plt.ylabel(f"Expression de {gene_y}")
plt.legend(title="Type de cancer",loc="best")

plt.grid(alpha=0.25)
plt.tight_layout()

#plt.show()

# ===============================================================================
#   Methode 2 c) - Distributions 1D des deux variables originales en Histogramme
# ===============================================================================
fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(16, 6)
)

###### Distribution du premier gène #####

sns.histplot(
    data=donnees_numeric,
    x=gene_x,
    hue="Class",
    hue_order=classes_attendues,
    palette=palette_cancers,
    bins=20,
    stat="probability",
    common_norm=False,
    common_bins=True,
    multiple="dodge",
    element="bars",
    fill=True,
    shrink=0.85,
    alpha=0.85,
    ax=axes[0]
)

axes[0].set_title(
    f"Distribution de {gene_x} selon la classe"
)

axes[0].set_xlabel(f"Expression de {gene_x}")
axes[0].set_ylabel("Proportion de patients")
axes[0].grid(alpha=0.25)


#### Distribution du deuxième gène  #######

sns.histplot(
    data=donnees_numeric,
    x=gene_y,
    hue="Class",
    hue_order=classes_attendues,
    palette=palette_cancers,
    bins=20,
    stat="probability",
    common_norm=False,
    common_bins=True,
    multiple="dodge",
    element="bars",
    fill=True,
    shrink=0.85,
    alpha=0.85,
    ax=axes[1]
)

axes[1].set_title(
    f"Distribution de {gene_y} selon la classe"
)

axes[1].set_xlabel(f"Expression de {gene_y}")
axes[1].set_ylabel("Proportion de patients")
axes[1].grid(alpha=0.25)


fig.suptitle(
    f"Méthode 2(c) : distributions des variables "
    f"{gene_x} et {gene_y}",
    fontsize=14
)

plt.tight_layout()

#plt.show()

# ============================================================
#                    MÉTHODE 2(d) - 1) ACP
# ============================================================
colonnes_variables =[
    colonne for colonne in data.columns
    if not colonne.startswith("Unnamed")
]

# ACP est effectuée sur tous les gènes.
var_X = data[colonnes_variables].apply(pd.to_numeric,errors="raise")
if var_X.isna().any().any():
    raise ValueError( "valeurs manquantes dans la matrice!!!")

# une ligne de la matrice = 1 patient,
# et une colonne = un gène
var_X = var_X.to_numpy(dtype=float)

var_Y = label["Class"].to_numpy()

#print("Matrice meth2: ", var_X.shape)
#print("Nombre etiquettes: ", len(var_Y))

#On va appliquer un centrage-réduction pour eviter les écarts entre attriibuts
preprocess_acp = StandardScaler()

X_acp = preprocess_acp.fit_transform(var_X)

#print("Moyenne approximative de la première variable :",X_acp[:, 0].mean())
#print("Écart-type approximatif de la première variable :",X_acp[:, 0].std())

acp = PCA(n_components=2, random_state=0)
coordinates_acp = acp.fit_transform(X_acp)

#print("Dimensions des coordonnées ACP :",coordinates_acp.shape)

resultats_acp = pd.DataFrame({
    "Dimension_ACP_1": coordinates_acp[:, 0],
    "Dimension_ACP_2": coordinates_acp[:, 1],
    "Class": var_Y
})

######## ACP: Visualisation avec les 5 classes ######
plt.figure(figsize=(11, 8))

sns.scatterplot(
    data=resultats_acp,
    x="Dimension_ACP_1",
    y="Dimension_ACP_2",
    hue="Class",
    hue_order=classes_attendues,
    palette=palette_cancers,
    s=35,
    alpha=0.75,
    edgecolor=None
)

variance_cp1 = acp.explained_variance_ratio_[0] * 100
variance_cp2 = acp.explained_variance_ratio_[1] * 100

plt.title("ACP - Projection des patients selon les deux premières composantes")
plt.xlabel(f"Dimension_ACP_1 ({variance_cp1:.2f} % de la variance)")
plt.ylabel(f"Dimension_ACP_2 ({variance_cp2:.2f} % de la variance)")
plt.legend(title="Type de cancer",loc="best")
plt.grid(alpha=0.25)
plt.tight_layout()


#plt.show()
# ============================================================
#                          2) t-SNE
# ============================================================
#On va faire une réduction intermediaire par ACP à 50 composantes
#qui sera utilisé dans le calcul de t-SNE et permet de réduire le bruit (assuré suite au nombre élevé des variables)
N_COMPOSANTES_TSNE = 50

acp_tsne = PCA(n_components=N_COMPOSANTES_TSNE, random_state=0)

X_acp_tsne = acp_tsne.fit_transform(X_acp)

#------- calcul de t-SNE -------------
tsne = TSNE(
    n_components=2,
    random_state=0,
    init='pca',
    perplexity=30.0,
    max_iter=1000,
    method="barnes_hut",
    angle=0.5,
    learning_rate="auto",
    verbose=0
)

coordinates_tsne = tsne.fit_transform(X_acp_tsne)
#print("Dim des coord t-SNE: ", coordinates_tsne.shape)
#--------- Visualisation t-SNE ----------------
resultats_tsne = pd.DataFrame({
    "tSNE-1": coordinates_tsne[:, 0],
    "tSNE-2": coordinates_tsne[:, 1],
    "Class": var_Y
})

plt.figure(figsize=(11, 8))

sns.scatterplot(
    data=resultats_tsne,
    x="tSNE-1",
    y="tSNE-2",
    hue="Class",
    hue_order=classes_attendues,
    palette=palette_cancers,
    s=55,
    alpha=0.8,
    edgecolor="k"
)

plt.title(
    "t-SNE - Projection des patients dans deux dimensions"
)

plt.xlabel("tSNE-1")
plt.ylabel("tSNE-2")
plt.legend(title="Type de cancer", loc="best")
plt.grid(alpha=0.25)
plt.tight_layout()


plt.show()


