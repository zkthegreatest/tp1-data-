from operator import index

import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import distance

label  = pd.read_csv('labels.csv')
print(label)
data = pd.read_csv('data.csv')
print(data)
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
BRCA_Matrix_cut = BRCA_Matrix.iloc[:, :5]
#BRCA_Matrix = np.array(BRCA_Matrix)
#BRCA_Matrix_cut = np.array(BRCA_Matrix_cut)
#print(np.array(BRCA_Matrix))
#print(BRCA_Matrix_cut)
KIRC_Matrix =  KIRC_class.drop(columns=KIRC_class.columns[KIRC_class.columns.str.startswith('Unnamed')])
KIRC_Matrix = KIRC_Matrix.drop(columns='Class')
KIRC_Matrix_cut = KIRC_Matrix.iloc[:, :5]
#KIRC_Matrix = np.array(KIRC_Matrix)
#KIRC_Matrix_cut = np.array(KIRC_Matrix_cut)
#print(KIRC_Matrix)
COAD_Matrix = COAD_class.drop(columns=COAD_class.columns[COAD_class.columns.str.startswith('Unnamed')])
COAD_Matrix = COAD_Matrix.drop(columns='Class')
COAD_Matrix_cut = COAD_Matrix.iloc[:, :5]
#COAD_Matrix = np.array(COAD_Matrix)
#COAD_Matrix_cut = np.array(COAD_Matrix_cut)
#print(COAD_Matrix)
LUAD_Matrix = LUAD_class.drop(columns=LUAD_class.columns[LUAD_class.columns.str.startswith('Unnamed')])
LUAD_Matrix = LUAD_Matrix.drop(columns='Class')
LUAD_Matrix_cut = LUAD_Matrix.iloc[:, :5]
#LUAD_Matrix = np.array(LUAD_Matrix)
#LUAD_Matrix_cut = np.array(LUAD_Matrix_cut)
#print(LUAD_Matrix)
PRAD_Matrix = PRAD_class.drop(columns=PRAD_class.columns[PRAD_class.columns.str.startswith('Unnamed')])
PRAD_Matrix = PRAD_Matrix.drop(columns='Class')
PRAD_Matrix_cut = PRAD_Matrix.iloc[:, :5]
#PRAD_Matrix = np.array(PRAD_Matrix)
#PRAD_Matrix_cut = np.array(PRAD_Matrix_cut)
#print(PRAD_Matrix)

########################CENTRE DES CLASSES######################
BRCA_Center = np.array(BRCA_Matrix.mean())
KIRC_Center = np.array(KIRC_Matrix.mean())
COAD_Center = np.array(COAD_Matrix.mean())
LUAD_Center = np.array(LUAD_Matrix.mean())
PRAD_Center = np.array(PRAD_Matrix.mean())
BRCA_Center_cut = np.array(BRCA_Matrix_cut.mean())
KIRC_Center_cut = np.array(KIRC_Matrix_cut.mean())
COAD_Center_cut = np.array(COAD_Matrix_cut.mean())
LUAD_Center_cut = np.array(LUAD_Matrix_cut.mean())
PRAD_Center_cut = np.array(PRAD_Matrix_cut.mean())
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
print(BRCA_cohesion_eucludian , 'cohésion  eucllidian BRCA')
tabDist.clear()
###############################################################

################COHÉSION CLASSE KIRC############################
for patient in np.array(KIRC_Matrix):
    dist = euclidean(patient, KIRC_Center)
    tabDist.append(dist)

KIRC_cohesion_eucludian =max(tabDist)
print(KIRC_cohesion_eucludian, 'cohésion  eucllidian KIRC')
tabDist.clear()
################################################################

################COHÉSION CLASSE COAD############################
for patient in np.array(COAD_Matrix):
    dist = euclidean(patient, COAD_Center)
    tabDist.append(dist)

COAD_cohesion_eucludian =max(tabDist)
print(COAD_cohesion_eucludian , 'cohésion  eucllidian coad')
tabDist.clear()
################################################################


################COHÉSION CLASSE LUAD############################
for patient in np.array(LUAD_Matrix):
    dist = euclidean(patient, LUAD_Center)
    tabDist.append(dist)

LUAD_cohesion_eucludian =max(tabDist)
print(LUAD_cohesion_eucludian , 'cohésion  eucllidian luad')
tabDist.clear()
################################################################

################COHÉSION CLASSE PRAD############################
for patient in np.array(PRAD_Matrix):
    dist = euclidean(patient, PRAD_Center)
    tabDist.append(dist)

PRAD_cohesion_eucludian =max(tabDist)
print(PRAD_cohesion_eucludian , 'cohésion  eucllidian prad')
tabDist.clear()
################################################################

################################################################
###################Euclidian Distance Fin#######################
################################################################

# ###################Mahalanobis Distance#########################
#utilisation des matrice coupé car l'ordinateur est pas assez puissant
# ########################COVAVIANCE##############################
Cov_BRCA = np.cov(np.array(BRCA_Matrix_cut),rowvar=False)
Cov_KIRC = np.cov(np.array(KIRC_Matrix_cut),rowvar=False)
Cov_COAD = np.cov(np.array(COAD_Matrix_cut),rowvar=False)
Cov_LUAD = np.cov(np.array(LUAD_Matrix_cut),rowvar=False)
Cov_PRAD = np.cov(np.array(PRAD_Matrix_cut),rowvar=False)
# print(1)
# #######################INV COVARIANCE###########################
Inv_cov_BRCA = np.linalg.pinv(Cov_BRCA)
Inv_cov_KIRC = np.linalg.pinv(Cov_KIRC)
Inv_cov_COAD = np.linalg.pinv(Cov_COAD)
Inv_cov_LUAD = np.linalg.pinv(Cov_LUAD)
Inv_cov_PRAD = np.linalg.pinv(Cov_PRAD)
#
# # ###################COHÉSION CLASSE BRCA#########################
for patient in np.array(BRCA_Matrix_cut):
      dist = distance.mahalanobis(patient, BRCA_Center_cut,Inv_cov_BRCA)
      tabDist.append(dist)

BRCA_cohesion_m =max(tabDist)
print(BRCA_cohesion_m , 'cohesion mahalanobis BRCA')
tabDist.clear()
# # ################COHÉSION CLASSE KIRC############################
for patient in np.array(KIRC_Matrix_cut):
     dist = distance.mahalanobis(patient, KIRC_Center_cut,Inv_cov_KIRC)
     tabDist.append(dist)
#
KIRC_cohesion_m =max(tabDist)
print(KIRC_cohesion_m , 'cohesion mahalanobis KIRC')
tabDist.clear()
# # ################################################################

# #
# # ################COHÉSION CLASSE COAD############################
for patient in np.array(COAD_Matrix_cut):
      dist = distance.mahalanobis(patient, COAD_Center_cut,Inv_cov_COAD)
      tabDist.append(dist)
#
COAD_cohesion_m =max(tabDist)
print(COAD_cohesion_m , 'cohesion mahalanobis COAD')
tabDist.clear()
# # ################################################################
# #
# #
# # ################COHÉSION CLASSE LUAD############################
for patient in np.array(LUAD_Matrix_cut):
      dist = distance.mahalanobis(patient, LUAD_Center_cut,Inv_cov_LUAD)
      tabDist.append(dist)
#
LUAD_cohesion_m =max(tabDist)
print(LUAD_cohesion_m , 'cohesion mahalanobis LUAD')
tabDist.clear()
# # ################################################################
# #
# # ################COHÉSION CLASSE PRAD############################
for patient in np.array(PRAD_Matrix_cut):
      dist = distance.mahalanobis(patient, PRAD_Center_cut,Inv_cov_PRAD)
      tabDist.append(dist)
#
PRAD_cohesion_m =max(tabDist)
print(PRAD_cohesion_m , 'cohesion mahalanobis PRAD')
tabDist.clear()
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
print(BRCA_cohesion_cosine , 'cohesion cosine BRCA')
tabDist.clear()
###############################################################

################COHÉSION CLASSE KIRC############################
for patient in np.array(KIRC_Matrix):
    dist = distance.cosine(patient, KIRC_Center)
    tabDist.append(dist)

KIRC_cohesion_cosine =max(tabDist)
print(KIRC_cohesion_cosine , 'cohesion cosine KIRC')
tabDist.clear()
################################################################

################COHÉSION CLASSE COAD############################
for patient in np.array(COAD_Matrix):
    dist = distance.cosine(patient, COAD_Center)
    tabDist.append(dist)

COAD_cohesion_cosine =max(tabDist)
print(COAD_cohesion_cosine , 'cohesion cosine COAD')
tabDist.clear()
################################################################


################COHÉSION CLASSE LUAD############################
for patient in np.array(LUAD_Matrix):
    dist = distance.cosine(patient, LUAD_Center)
    tabDist.append(dist)

LUAD_cohesion_cosine =max(tabDist)
print(LUAD_cohesion_cosine , 'cohesion cosine LUAD')
tabDist.clear()
################################################################

################COHÉSION CLASSE PRAD############################
for patient in np.array(PRAD_Matrix):
    dist = distance.cosine(patient, PRAD_Center)
    tabDist.append(dist)

PRAD_cohesion_cosine =max(tabDist)
print(PRAD_cohesion_cosine , 'cohesion cosine PRAD')
tabDist.clear()
tabDist.clear()
###########################################################
###################cosine Distance Fin#####################
###########################################################



###################Distance intra classe##################
tabDistC1 = []
tabDistC2 = []
##################separation BCRA#########################
#####################KIRC#################################

for patient in np.array(BRCA_Matrix):
    dist = euclidean(patient, KIRC_Center)
    tabDistC1.append(dist)

for patient in np.array(KIRC_Matrix):
    dist = euclidean(patient, BRCA_Center)
    tabDistC2.append(dist)

res_BRCA_KIRC_euclidian = min(min(tabDistC1), min(tabDistC2))
print(res_BRCA_KIRC_euclidian ,'separation  BRCA-KIRC euclidian')
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
print(res_BRCA_COAD_euclidian ,'separation  BRCA-COAD euclidian')
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
print(res_BRCA_Luad_euclidian ,'separation  BRCA-LUAD euclidian')
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
print(res_BRCA_Prad_euclidian ,'separation  BRCA-PRAD euclidian' )
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
print(res_KIRC_COAD_euclidian ,'separation KIRC-COAD euclidian')
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
print(res_KIRC_LUAD_euclidian ,'separation KIRC-LUAD euclidian')
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
print(res_KIRC_PRAD_euclidian ,'separation KIRC-PRAD euclidian')
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
print(res_COAD_LUAD_euclidian ,'separation COAD-Luad euclidian')
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
print(res_COAD_PRAD_euclidian ,'separation COAD-PRAD euclidian')
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
print(res_LUAD_PRAD_euclidian ,'separation LUAD-PRAD euclidian')
tabDistC1.clear()
tabDistC2.clear()
###################################################

#################separation mahalanobis############
###################separation BRCA##############
for patient in np.array(BRCA_Matrix_cut):
    dist = distance.mahalanobis(patient, KIRC_Center_cut,Inv_cov_BRCA )
    tabDistC1.append(dist)

for patient in np.array(KIRC_Matrix_cut):
    dist = distance.mahalanobis(patient, BRCA_Center_cut,Inv_cov_KIRC)
    tabDistC2.append(dist)

res_BRCA_KIRC_m = min(min(tabDistC1), min(tabDistC2))
print(res_BRCA_KIRC_m ,'separation BRCA-Kirc mahalanobis')
tabDistC1.clear()
tabDistC2.clear()

##########################################################

#####################COAD###############################
for patient in np.array(BRCA_Matrix_cut):
    dist = distance.mahalanobis(patient, COAD_Center_cut,Inv_cov_BRCA)
    tabDistC1.append(dist)

for patient in np.array(COAD_Matrix_cut):
    dist = distance.mahalanobis(patient, BRCA_Center_cut,Inv_cov_COAD)
    tabDistC2.append(dist)

res_BRCA_COAD_m = min(min(tabDistC1), min(tabDistC2))
print(res_BRCA_COAD_m ,'separation BRCA-COAD mahalanobis')
tabDistC1.clear()
tabDistC2.clear()
#######################################################

######################Luad#############################
for patient in np.array(BRCA_Matrix_cut):
    dist = distance.mahalanobis(patient, LUAD_Center_cut,Inv_cov_BRCA)
    tabDistC1.append(dist)

for patient in np.array(LUAD_Matrix_cut):
    dist = distance.mahalanobis(patient, BRCA_Center_cut,Inv_cov_LUAD)
    tabDistC2.append(dist)

res_BRCA_Luad_m = min(min(tabDistC1), min(tabDistC2))
print(res_BRCA_Luad_m ,'separation BRCA-Luad mahalanobis')
tabDistC1.clear()
tabDistC2.clear()
#######################################################

#######################PRAD############################
for patient in np.array(BRCA_Matrix_cut):
    dist = distance.mahalanobis(patient, PRAD_Center_cut,Inv_cov_BRCA)
    tabDistC1.append(dist)

for patient in np.array(PRAD_Matrix_cut):
    dist = distance.mahalanobis(patient, BRCA_Center_cut,Inv_cov_PRAD)
    tabDistC2.append(dist)

res_BRCA_Prad_m = min(min(tabDistC1), min(tabDistC2))
print(res_BRCA_Prad_m ,'separation BRCA-PRAD mahalanobis')
tabDistC1.clear()
tabDistC2.clear()
########################################################
##################Separation KIRC######################
#####################coad############################

for patient in np.array(KIRC_Matrix_cut):
    dist = distance.mahalanobis(patient, COAD_Center_cut,Inv_cov_KIRC)
    tabDistC1.append(dist)

for patient in np.array(COAD_Matrix_cut):
    dist = distance.mahalanobis(patient, KIRC_Center_cut,Inv_cov_COAD)
    tabDistC2.append(dist)

res_KIRC_COAD_m = min(min(tabDistC1), min(tabDistC2))
print(res_KIRC_COAD_m ,'separation KIRC-COAD mahalanobis')
tabDistC1.clear()
tabDistC2.clear()
#######################################################

#####################LUAD############################

for patient in np.array(KIRC_Matrix_cut):
    dist = distance.mahalanobis(patient, LUAD_Center_cut,Inv_cov_KIRC)
    tabDistC1.append(dist)

for patient in np.array(LUAD_Matrix_cut):
    dist = distance.mahalanobis(patient, KIRC_Center_cut,Inv_cov_LUAD)
    tabDistC2.append(dist)

res_KIRC_LUAD_m = min(min(tabDistC1), min(tabDistC2))
print(res_KIRC_LUAD_m ,'separation KIRC-LUAD mahalanobis')
tabDistC1.clear()
tabDistC2.clear()
#######################################################
#####################PRAD############################

for patient in np.array(KIRC_Matrix_cut):
    dist = distance.mahalanobis(patient, PRAD_Center_cut,Inv_cov_KIRC)
    tabDistC1.append(dist)

for patient in np.array(PRAD_Matrix_cut):
    dist = distance.mahalanobis(patient, KIRC_Center_cut,Inv_cov_PRAD)
    tabDistC2.append(dist)

res_KIRC_PRAD_m = min(min(tabDistC1), min(tabDistC2))
print(res_KIRC_PRAD_m ,'separation KIRC-PRAD mahalanobis')
tabDistC1.clear()
tabDistC2.clear()
#######################################################

######################separation COAD##################
#####################LUAD############################

for patient in np.array(COAD_Matrix_cut):
    dist = distance.mahalanobis(patient, LUAD_Center_cut,Inv_cov_COAD)
    tabDistC1.append(dist)

for patient in np.array(LUAD_Matrix_cut):
    dist = distance.mahalanobis(patient, COAD_Center_cut,Inv_cov_LUAD)
    tabDistC2.append(dist)

res_COAD_LUAD_m = min(min(tabDistC1), min(tabDistC2))
print(res_COAD_LUAD_m ,'separation COAD-LUAD mahalanobis')
tabDistC1.clear()
tabDistC2.clear()
#######################################################

#####################PRAD############################

for patient in np.array(COAD_Matrix_cut):
    dist = distance.mahalanobis(patient, PRAD_Center_cut, Inv_cov_COAD)
    tabDistC1.append(dist)

for patient in np.array(PRAD_Matrix_cut):
    dist = distance.mahalanobis(patient, COAD_Center_cut,Inv_cov_PRAD)
    tabDistC2.append(dist)

res_COAD_PRAD_m= min(min(tabDistC1), min(tabDistC2))
print(res_COAD_PRAD_m ,'separation COAD-PRAD mahalanobis')
tabDistC1.clear()
tabDistC2.clear()
#######################################################

#######################separation LUAD################
##########################Prad######################
for patient in np.array(LUAD_Matrix_cut):
    dist = distance.mahalanobis(patient, PRAD_Center_cut,Inv_cov_LUAD)
    tabDistC1.append(dist)

for patient in np.array(PRAD_Matrix_cut):
    dist = distance.mahalanobis(patient, LUAD_Center_cut,Inv_cov_PRAD)
    tabDistC2.append(dist)

res_LUAD_PRAD_m= min(min(tabDistC1), min(tabDistC2))
print(res_LUAD_PRAD_m ,'separation LUAD-PRAD mahalanobis')
tabDistC1.clear()
tabDistC2.clear()
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
print(res_BRCA_KIRC_cosine ,'separation BRCA-KIRC cosine')
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
print(res_BRCA_COAD_cosine ,'separation BRCA-COAD cosine')
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
print(res_BRCA_Luad_cosine ,'separation BRCA-LUAD cosine')
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
print(res_BRCA_Prad_cosine ,'separation BRCA-PRAD cosine')
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
print(res_KIRC_COAD_cosine ,'separation KIRC-COAD cosine')
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
print(res_KIRC_LUAD_cosine ,'separation KIRC-LUAD cosine')
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
print(res_KIRC_PRAD_cosine ,'separation KIRC-PRAD cosine')
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
print(res_COAD_LUAD_cosine ,'separation COAD-LUAD cosine')
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
print(res_COAD_PRAD_cosine ,'separation COAD-PRAD cosine')
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
print(res_LUAD_PRAD_cosine ,'separation LUAD-PRAD cosine')
tabDistC1.clear()
tabDistC2.clear()
###################################################


#####################OVERLAP########################

###########################Euclidian###############

######################BRCA - KIRC###################
overlap_BRCA_KIRC = ( BRCA_cohesion_eucludian + KIRC_cohesion_eucludian)/ (2* res_BRCA_KIRC_euclidian)
print(overlap_BRCA_KIRC ,'overlap BRCA-KIRC euclidian')
####################################################

######################BRCA -COAD#####################
overlap_BRCA_COAD = ( BRCA_cohesion_eucludian + COAD_cohesion_eucludian)/ (2* res_BRCA_COAD_euclidian)
print(overlap_BRCA_COAD ,'overlap BRCA-COAD euclidian')
#####################################################

######################BRCA -LUAD#####################
overlap_BRCA_LUAD = ( BRCA_cohesion_eucludian + LUAD_cohesion_eucludian)/ (2* res_BRCA_Luad_euclidian)
print(overlap_BRCA_LUAD ,'overlap BRCA-LUAD euclidian')
#####################################################


#########################BRCA -PRAD##################
overlap_BRCA_PRAD = ( BRCA_cohesion_eucludian + PRAD_cohesion_eucludian)/ (2* res_BRCA_Prad_euclidian)
print(overlap_BRCA_PRAD ,'overlap BRCA-PRAD euclidian')
#####################################################

#########################KIRC -COAD##################
overlap_KIRC_COAD = ( KIRC_cohesion_eucludian + COAD_cohesion_eucludian)/ (2* res_KIRC_COAD_euclidian)
print(overlap_KIRC_COAD ,'overlap KIRC-COAD euclidian')
#####################################################

########################KIRC - LUAD##################
overlap_KIRC_LUAD = ( KIRC_cohesion_eucludian + LUAD_cohesion_eucludian)/ (2* res_KIRC_LUAD_euclidian)
print(overlap_KIRC_LUAD ,'overlap kirc-Luad euclidian')
#####################################################

#######################KIRC - PRAD###################
overlap_KIRC_PRAD = ( KIRC_cohesion_eucludian + PRAD_cohesion_eucludian)/ (2* res_KIRC_PRAD_euclidian)
print(overlap_KIRC_PRAD ,'overlap kirc-prad euclidian')
#####################################################


####################COAD -LUAD######################
overlap_COAD_LUAD = ( COAD_cohesion_eucludian + LUAD_cohesion_eucludian)/ (2* res_COAD_LUAD_euclidian)
print(overlap_COAD_LUAD ,'overlap coad-luad euclidian')
####################################################

##################COAD-PRAD#########################
overlap_COAD_PRAD = ( COAD_cohesion_eucludian + PRAD_cohesion_eucludian)/ (2* res_COAD_PRAD_euclidian)
print(overlap_COAD_PRAD ,'overlap coad-prad euclidian')
###################################################

###########################LUAD-PRAD##############
overlap_LUAD_PRAD = ( PRAD_cohesion_eucludian + LUAD_cohesion_eucludian)/ (2* res_LUAD_PRAD_euclidian)
print(overlap_LUAD_PRAD ,'overlap luad-prad euclidian')
#################################################





#########################mahalanobis#################
######################BRCA - KIRC###################
overlap_BRCA_KIRC_m = ( BRCA_cohesion_m+ KIRC_cohesion_m)/ (2* res_BRCA_KIRC_m)
print(overlap_BRCA_KIRC_m ,'overlap BRCA-KIRC mahalanobis')
####################################################

######################BRCA -COAD#####################
overlap_BRCA_COAD_m = ( BRCA_cohesion_m+ COAD_cohesion_m)/ (2* res_BRCA_COAD_m)
print(overlap_BRCA_COAD_m ,'overlap BRCA-COAD mahalanobis')
#####################################################

######################BRCA -LUAD#####################
overlap_BRCA_LUAD_m = ( BRCA_cohesion_m + LUAD_cohesion_m)/ (2* res_BRCA_Luad_m)
print(overlap_BRCA_LUAD_m ,'overlap BRCA-LUAD mahalanobis')
#####################################################


#########################BRCA -PRAD##################
overlap_BRCA_PRAD_m = ( BRCA_cohesion_m + PRAD_cohesion_m)/ (2* res_BRCA_Prad_m)
print(overlap_BRCA_PRAD_m ,'overlap BRCA-PRAD mahalanobis')
#####################################################

#########################KIRC -COAD##################
overlap_KIRC_COAD_m = ( KIRC_cohesion_m + COAD_cohesion_m)/ (2* res_KIRC_COAD_m)
print(overlap_KIRC_COAD_m ,'overlap KIRC-COAD mahalanobis')
#####################################################

########################KIRC - LUAD##################
overlap_KIRC_LUAD_m = ( KIRC_cohesion_m + LUAD_cohesion_m)/ (2* res_KIRC_LUAD_m)
print(overlap_KIRC_LUAD_m ,'overlap KIRC-LUAD mahalanobis')
#####################################################

#######################KIRC - PRAD###################
overlap_KIRC_PRAD_m = ( KIRC_cohesion_m + PRAD_cohesion_m)/ (2* res_KIRC_PRAD_m)
print(overlap_KIRC_PRAD_m ,'overlap KIRC-PRAD mahalanobis')
#####################################################


####################COAD -LUAD######################
overlap_COAD_LUAD_m = ( COAD_cohesion_m + LUAD_cohesion_m)/ (2* res_COAD_LUAD_m)
print(overlap_COAD_LUAD_m ,'overlap COAD-LUAD mahalanobis')
####################################################

##################COAD-PRAD#########################
overlap_COAD_PRAD_m = ( COAD_cohesion_m + PRAD_cohesion_m)/ (2* res_COAD_PRAD_m)
print(overlap_COAD_PRAD_m ,'overlap COAD-PRAD mahalanobis')
###################################################

###########################LUAD-PRAD##############
overlap_LUAD_PRAD_m = ( PRAD_cohesion_m + LUAD_cohesion_m)/ (2* res_LUAD_PRAD_m)
print(overlap_LUAD_PRAD_m ,'overlap LUAD-PRAD mahalanobis')
#################################################

######################################################

#######################cosinus######################
######################BRCA - KIRC###################
overlap_BRCA_KIRC_cosine = ( BRCA_cohesion_cosine + KIRC_cohesion_cosine)/ (2* res_BRCA_KIRC_cosine)
print(overlap_BRCA_KIRC_cosine ,'overlap BRCA-KIRC cosine')
####################################################

######################BRCA -COAD#####################
overlap_BRCA_COAD_cosine = ( BRCA_cohesion_cosine+ COAD_cohesion_cosine)/ (2* res_BRCA_COAD_cosine)
print(overlap_BRCA_COAD_cosine ,'overlap BRCA-COAD cosine')
#####################################################

######################BRCA -LUAD#####################
overlap_BRCA_LUAD_cosine = ( BRCA_cohesion_cosine + LUAD_cohesion_cosine)/ (2* res_BRCA_Luad_cosine)
print(overlap_BRCA_LUAD_cosine ,'overlap BRCA-LUAD cosine')
#####################################################


#########################BRCA -PRAD##################
overlap_BRCA_PRAD_cosine = ( BRCA_cohesion_cosine + PRAD_cohesion_cosine)/ (2* res_BRCA_Prad_cosine)
print(overlap_BRCA_PRAD_cosine ,'overlap BRCA-PRAD cosine')
#####################################################

#########################KIRC -COAD##################
overlap_KIRC_COAD_cosine = ( KIRC_cohesion_cosine + COAD_cohesion_cosine)/ (2* res_KIRC_COAD_cosine)
print(overlap_KIRC_COAD_cosine ,'overlap KIRC-COAD cosine')
#####################################################

########################KIRC - LUAD##################
overlap_KIRC_LUAD_cosine = ( KIRC_cohesion_cosine + LUAD_cohesion_cosine)/ (2* res_KIRC_LUAD_cosine)
print(overlap_KIRC_LUAD_cosine ,'overlap KIrc-LUAD cosine')
#####################################################

#######################KIRC - PRAD###################
overlap_KIRC_PRAD_cosine = ( KIRC_cohesion_cosine + PRAD_cohesion_cosine)/ (2* res_KIRC_PRAD_cosine)
print(overlap_KIRC_PRAD_cosine ,'overlap KIRC-PRAD cosine')
#####################################################


####################COAD -LUAD######################
overlap_COAD_LUAD_cosine = ( COAD_cohesion_cosine + LUAD_cohesion_cosine)/ (2* res_COAD_LUAD_cosine)
print(overlap_COAD_LUAD_cosine ,'overlap COAD -LUAD cosine')
####################################################

##################COAD-PRAD#########################
overlap_COAD_PRAD_cosine = ( COAD_cohesion_cosine + PRAD_cohesion_cosine)/ (2* res_COAD_PRAD_cosine)
print(overlap_COAD_PRAD_cosine ,'overlap prad-PRAD cosine')
###################################################

###########################LUAD-PRAD##############
overlap_LUAD_PRAD_cosine = ( PRAD_cohesion_cosine + LUAD_cohesion_cosine)/ (2* res_LUAD_PRAD_cosine)
print(overlap_LUAD_PRAD_cosine ,'overlap LUAD-PRAD cosine')
#################################################


def BienSepare(val):
    return val<1

