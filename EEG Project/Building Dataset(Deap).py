#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


import numpy as np
import os
import scipy.io
dataDir = "/content/drive/MyDrive/data_preprocessed_matlab/"
mats = []
for file in os.listdir(dataDir) :
    mats.append(scipy.io.loadmat(dataDir+file))


# In[ ]:


import pandas as pd
for i in range(len(mats)):
  mats[i]=pd.Series(mats[i])


# In[ ]:


for i in range(len(mats)):
  mats[i]=mats[i].drop(['__header__','__version__','__globals__'])


# In[ ]:


for i in range(len(mats)):
  print(i,mats[i]['data'].shape,mats[i]['labels'].shape)


# **Subject 1**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32
M


# In[ ]:


A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


# In[ ]:


B1 = []
for i in range(40):
  B1.append(A)
B1 = np.array(B1)


# In[ ]:


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []


# In[ ]:


J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[0][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


# In[ ]:


for x in range(40):
  for i in range(8064):   
    B1[x][i][0,3]=J[x].iloc[i,0]
    B1[x][i][0,5]=J[x].iloc[i,16]
    B1[x][i][1,3]=J[x].iloc[i,1]
    B1[x][i][1,5]=J[x].iloc[i,17]
    B1[x][i][2,0]=J[x].iloc[i,3]
    B1[x][i][2,2]=J[x].iloc[i,2]
    B1[x][i][2,4]=J[x].iloc[i,18]
    B1[x][i][2,6]=J[x].iloc[i,19]
    B1[x][i][2,8]=J[x].iloc[i,20]
    B1[x][i][3,1]=J[x].iloc[i,4]
    B1[x][i][3,3]=J[x].iloc[i,5]
    B1[x][i][3,5]=J[x].iloc[i,22]
    B1[x][i][3,7]=J[x].iloc[i,21]
    B1[x][i][4,0]=J[x].iloc[i,7]
    B1[x][i][4,2]=J[x].iloc[i,6]
    B1[x][i][4,4]=J[x].iloc[i,23]
    B1[x][i][4,6]=J[x].iloc[i,24]
    B1[x][i][4,8]=J[x].iloc[i,25]
    B1[x][i][5,1]=J[x].iloc[i,8]
    B1[x][i][5,3]=J[x].iloc[i,9]
    B1[x][i][5,5]=J[x].iloc[i,27]
    B1[x][i][5,7]=J[x].iloc[i,26]
    B1[x][i][6,0]=J[x].iloc[i,11]
    B1[x][i][6,2]=J[x].iloc[i,10]
    B1[x][i][6,4]=J[x].iloc[i,15]
    B1[x][i][6,6]=J[x].iloc[i,28]
    B1[x][i][6,8]=J[x].iloc[i,29]
    B1[x][i][7,3]=J[x].iloc[i,12]
    B1[x][i][7,5]=J[x].iloc[i,30]
    B1[x][i][8,3]=J[x].iloc[i,13]
    B1[x][i][8,4]=J[x].iloc[i,14]
    B1[x][i][8,5]=J[x].iloc[i,31]


# In[ ]:


B1.shape


# **Subject 2**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32
M


# In[ ]:


A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


# In[ ]:


B2 = []
for i in range(40):
  B2.append(A)
B2 = np.array(B2)


# In[ ]:


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []


# In[ ]:


J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[1][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


# In[ ]:


for x in range(40):
  for i in range(8064):   
    B2[x][i][0,3]=J[x].iloc[i,0]
    B2[x][i][0,5]=J[x].iloc[i,16]
    B2[x][i][1,3]=J[x].iloc[i,1]
    B2[x][i][1,5]=J[x].iloc[i,17]
    B2[x][i][2,0]=J[x].iloc[i,3]
    B2[x][i][2,2]=J[x].iloc[i,2]
    B2[x][i][2,4]=J[x].iloc[i,18]
    B2[x][i][2,6]=J[x].iloc[i,19]
    B2[x][i][2,8]=J[x].iloc[i,20]
    B2[x][i][3,1]=J[x].iloc[i,4]
    B2[x][i][3,3]=J[x].iloc[i,5]
    B2[x][i][3,5]=J[x].iloc[i,22]
    B2[x][i][3,7]=J[x].iloc[i,21]
    B2[x][i][4,0]=J[x].iloc[i,7]
    B2[x][i][4,2]=J[x].iloc[i,6]
    B2[x][i][4,4]=J[x].iloc[i,23]
    B2[x][i][4,6]=J[x].iloc[i,24]
    B2[x][i][4,8]=J[x].iloc[i,25]
    B2[x][i][5,1]=J[x].iloc[i,8]
    B2[x][i][5,3]=J[x].iloc[i,9]
    B2[x][i][5,5]=J[x].iloc[i,27]
    B2[x][i][5,7]=J[x].iloc[i,26]
    B2[x][i][6,0]=J[x].iloc[i,11]
    B2[x][i][6,2]=J[x].iloc[i,10]
    B2[x][i][6,4]=J[x].iloc[i,15]
    B2[x][i][6,6]=J[x].iloc[i,28]
    B2[x][i][6,8]=J[x].iloc[i,29]
    B2[x][i][7,3]=J[x].iloc[i,12]
    B2[x][i][7,5]=J[x].iloc[i,30]
    B2[x][i][8,3]=J[x].iloc[i,13]
    B2[x][i][8,4]=J[x].iloc[i,14]
    B2[x][i][8,5]=J[x].iloc[i,31]


# In[ ]:


B2.shape


# **Subject 3**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B3 = []
for i in range(40):
  B3.append(A)
B3 = np.array(B3)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[2][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B3[x][i][0,3]=J[x].iloc[i,0]
    B3[x][i][0,5]=J[x].iloc[i,16]
    B3[x][i][1,3]=J[x].iloc[i,1]
    B3[x][i][1,5]=J[x].iloc[i,17]
    B3[x][i][2,0]=J[x].iloc[i,3]
    B3[x][i][2,2]=J[x].iloc[i,2]
    B3[x][i][2,4]=J[x].iloc[i,18]
    B3[x][i][2,6]=J[x].iloc[i,19]
    B3[x][i][2,8]=J[x].iloc[i,20]
    B3[x][i][3,1]=J[x].iloc[i,4]
    B3[x][i][3,3]=J[x].iloc[i,5]
    B3[x][i][3,5]=J[x].iloc[i,22]
    B3[x][i][3,7]=J[x].iloc[i,21]
    B3[x][i][4,0]=J[x].iloc[i,7]
    B3[x][i][4,2]=J[x].iloc[i,6]
    B3[x][i][4,4]=J[x].iloc[i,23]
    B3[x][i][4,6]=J[x].iloc[i,24]
    B3[x][i][4,8]=J[x].iloc[i,25]
    B3[x][i][5,1]=J[x].iloc[i,8]
    B3[x][i][5,3]=J[x].iloc[i,9]
    B3[x][i][5,5]=J[x].iloc[i,27]
    B3[x][i][5,7]=J[x].iloc[i,26]
    B3[x][i][6,0]=J[x].iloc[i,11]
    B3[x][i][6,2]=J[x].iloc[i,10]
    B3[x][i][6,4]=J[x].iloc[i,15]
    B3[x][i][6,6]=J[x].iloc[i,28]
    B3[x][i][6,8]=J[x].iloc[i,29]
    B3[x][i][7,3]=J[x].iloc[i,12]
    B3[x][i][7,5]=J[x].iloc[i,30]
    B3[x][i][8,3]=J[x].iloc[i,13]
    B3[x][i][8,4]=J[x].iloc[i,14]
    B3[x][i][8,5]=J[x].iloc[i,31]


# In[ ]:


B3.shape


# **Subject 4**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B4 = []
for i in range(40):
  B4.append(A)
B4 = np.array(B4)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[3][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B4[x][i][0,3]=J[x].iloc[i,0]
    B4[x][i][0,5]=J[x].iloc[i,16]
    B4[x][i][1,3]=J[x].iloc[i,1]
    B4[x][i][1,5]=J[x].iloc[i,17]
    B4[x][i][2,0]=J[x].iloc[i,3]
    B4[x][i][2,2]=J[x].iloc[i,2]
    B4[x][i][2,4]=J[x].iloc[i,18]
    B4[x][i][2,6]=J[x].iloc[i,19]
    B4[x][i][2,8]=J[x].iloc[i,20]
    B4[x][i][3,1]=J[x].iloc[i,4]
    B4[x][i][3,3]=J[x].iloc[i,5]
    B4[x][i][3,5]=J[x].iloc[i,22]
    B4[x][i][3,7]=J[x].iloc[i,21]
    B4[x][i][4,0]=J[x].iloc[i,7]
    B4[x][i][4,2]=J[x].iloc[i,6]
    B4[x][i][4,4]=J[x].iloc[i,23]
    B4[x][i][4,6]=J[x].iloc[i,24]
    B4[x][i][4,8]=J[x].iloc[i,25]
    B4[x][i][5,1]=J[x].iloc[i,8]
    B4[x][i][5,3]=J[x].iloc[i,9]
    B4[x][i][5,5]=J[x].iloc[i,27]
    B4[x][i][5,7]=J[x].iloc[i,26]
    B4[x][i][6,0]=J[x].iloc[i,11]
    B4[x][i][6,2]=J[x].iloc[i,10]
    B4[x][i][6,4]=J[x].iloc[i,15]
    B4[x][i][6,6]=J[x].iloc[i,28]
    B4[x][i][6,8]=J[x].iloc[i,29]
    B4[x][i][7,3]=J[x].iloc[i,12]
    B4[x][i][7,5]=J[x].iloc[i,30]
    B4[x][i][8,3]=J[x].iloc[i,13]
    B4[x][i][8,4]=J[x].iloc[i,14]
    B4[x][i][8,5]=J[x].iloc[i,31]


# In[ ]:


B4.shape


# **Subject 5**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B5 = []
for i in range(40):
  B5.append(A)
B5 = np.array(B5)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[4][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B5[x][i][0,3]=J[x].iloc[i,0]
    B5[x][i][0,5]=J[x].iloc[i,16]
    B5[x][i][1,3]=J[x].iloc[i,1]
    B5[x][i][1,5]=J[x].iloc[i,17]
    B5[x][i][2,0]=J[x].iloc[i,3]
    B5[x][i][2,2]=J[x].iloc[i,2]
    B5[x][i][2,4]=J[x].iloc[i,18]
    B5[x][i][2,6]=J[x].iloc[i,19]
    B5[x][i][2,8]=J[x].iloc[i,20]
    B5[x][i][3,1]=J[x].iloc[i,4]
    B5[x][i][3,3]=J[x].iloc[i,5]
    B5[x][i][3,5]=J[x].iloc[i,22]
    B5[x][i][3,7]=J[x].iloc[i,21]
    B5[x][i][4,0]=J[x].iloc[i,7]
    B5[x][i][4,2]=J[x].iloc[i,6]
    B5[x][i][4,4]=J[x].iloc[i,23]
    B5[x][i][4,6]=J[x].iloc[i,24]
    B5[x][i][4,8]=J[x].iloc[i,25]
    B5[x][i][5,1]=J[x].iloc[i,8]
    B5[x][i][5,3]=J[x].iloc[i,9]
    B5[x][i][5,5]=J[x].iloc[i,27]
    B5[x][i][5,7]=J[x].iloc[i,26]
    B5[x][i][6,0]=J[x].iloc[i,11]
    B5[x][i][6,2]=J[x].iloc[i,10]
    B5[x][i][6,4]=J[x].iloc[i,15]
    B5[x][i][6,6]=J[x].iloc[i,28]
    B5[x][i][6,8]=J[x].iloc[i,29]
    B5[x][i][7,3]=J[x].iloc[i,12]
    B5[x][i][7,5]=J[x].iloc[i,30]
    B5[x][i][8,3]=J[x].iloc[i,13]
    B5[x][i][8,4]=J[x].iloc[i,14]
    B5[x][i][8,5]=J[x].iloc[i,31]


# In[ ]:


B5.shape


# **Subject 6**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B6 = []
for i in range(40):
  B6.append(A)
B6 = np.array(B6)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[5][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B6[x][i][0,3]=J[x].iloc[i,0]
    B6[x][i][0,5]=J[x].iloc[i,16]
    B6[x][i][1,3]=J[x].iloc[i,1]
    B6[x][i][1,5]=J[x].iloc[i,17]
    B6[x][i][2,0]=J[x].iloc[i,3]
    B6[x][i][2,2]=J[x].iloc[i,2]
    B6[x][i][2,4]=J[x].iloc[i,18]
    B6[x][i][2,6]=J[x].iloc[i,19]
    B6[x][i][2,8]=J[x].iloc[i,20]
    B6[x][i][3,1]=J[x].iloc[i,4]
    B6[x][i][3,3]=J[x].iloc[i,5]
    B6[x][i][3,5]=J[x].iloc[i,22]
    B6[x][i][3,7]=J[x].iloc[i,21]
    B6[x][i][4,0]=J[x].iloc[i,7]
    B6[x][i][4,2]=J[x].iloc[i,6]
    B6[x][i][4,4]=J[x].iloc[i,23]
    B6[x][i][4,6]=J[x].iloc[i,24]
    B6[x][i][4,8]=J[x].iloc[i,25]
    B6[x][i][5,1]=J[x].iloc[i,8]
    B6[x][i][5,3]=J[x].iloc[i,9]
    B6[x][i][5,5]=J[x].iloc[i,27]
    B6[x][i][5,7]=J[x].iloc[i,26]
    B6[x][i][6,0]=J[x].iloc[i,11]
    B6[x][i][6,2]=J[x].iloc[i,10]
    B6[x][i][6,4]=J[x].iloc[i,15]
    B6[x][i][6,6]=J[x].iloc[i,28]
    B6[x][i][6,8]=J[x].iloc[i,29]
    B6[x][i][7,3]=J[x].iloc[i,12]
    B6[x][i][7,5]=J[x].iloc[i,30]
    B6[x][i][8,3]=J[x].iloc[i,13]
    B6[x][i][8,4]=J[x].iloc[i,14]
    B6[x][i][8,5]=J[x].iloc[i,31]


# In[ ]:


B6.shape


# **Subject 7**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B7 = []
for i in range(40):
  B7.append(A)
B7 = np.array(B7)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[6][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B7[x][i][0,3]=J[x].iloc[i,0]
    B7[x][i][0,5]=J[x].iloc[i,16]
    B7[x][i][1,3]=J[x].iloc[i,1]
    B7[x][i][1,5]=J[x].iloc[i,17]
    B7[x][i][2,0]=J[x].iloc[i,3]
    B7[x][i][2,2]=J[x].iloc[i,2]
    B7[x][i][2,4]=J[x].iloc[i,18]
    B7[x][i][2,6]=J[x].iloc[i,19]
    B7[x][i][2,8]=J[x].iloc[i,20]
    B7[x][i][3,1]=J[x].iloc[i,4]
    B7[x][i][3,3]=J[x].iloc[i,5]
    B7[x][i][3,5]=J[x].iloc[i,22]
    B7[x][i][3,7]=J[x].iloc[i,21]
    B7[x][i][4,0]=J[x].iloc[i,7]
    B7[x][i][4,2]=J[x].iloc[i,6]
    B7[x][i][4,4]=J[x].iloc[i,23]
    B7[x][i][4,6]=J[x].iloc[i,24]
    B7[x][i][4,8]=J[x].iloc[i,25]
    B7[x][i][5,1]=J[x].iloc[i,8]
    B7[x][i][5,3]=J[x].iloc[i,9]
    B7[x][i][5,5]=J[x].iloc[i,27]
    B7[x][i][5,7]=J[x].iloc[i,26]
    B7[x][i][6,0]=J[x].iloc[i,11]
    B7[x][i][6,2]=J[x].iloc[i,10]
    B7[x][i][6,4]=J[x].iloc[i,15]
    B7[x][i][6,6]=J[x].iloc[i,28]
    B7[x][i][6,8]=J[x].iloc[i,29]
    B7[x][i][7,3]=J[x].iloc[i,12]
    B7[x][i][7,5]=J[x].iloc[i,30]
    B7[x][i][8,3]=J[x].iloc[i,13]
    B7[x][i][8,4]=J[x].iloc[i,14]
    B7[x][i][8,5]=J[x].iloc[i,31]


# **Subject 8**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B8 = []
for i in range(40):
  B8.append(A)
B8 = np.array(B8)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[7][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B8[x][i][0,3]=J[x].iloc[i,0]
    B8[x][i][0,5]=J[x].iloc[i,16]
    B8[x][i][1,3]=J[x].iloc[i,1]
    B8[x][i][1,5]=J[x].iloc[i,17]
    B8[x][i][2,0]=J[x].iloc[i,3]
    B8[x][i][2,2]=J[x].iloc[i,2]
    B8[x][i][2,4]=J[x].iloc[i,18]
    B8[x][i][2,6]=J[x].iloc[i,19]
    B8[x][i][2,8]=J[x].iloc[i,20]
    B8[x][i][3,1]=J[x].iloc[i,4]
    B8[x][i][3,3]=J[x].iloc[i,5]
    B8[x][i][3,5]=J[x].iloc[i,22]
    B8[x][i][3,7]=J[x].iloc[i,21]
    B8[x][i][4,0]=J[x].iloc[i,7]
    B8[x][i][4,2]=J[x].iloc[i,6]
    B8[x][i][4,4]=J[x].iloc[i,23]
    B8[x][i][4,6]=J[x].iloc[i,24]
    B8[x][i][4,8]=J[x].iloc[i,25]
    B8[x][i][5,1]=J[x].iloc[i,8]
    B8[x][i][5,3]=J[x].iloc[i,9]
    B8[x][i][5,5]=J[x].iloc[i,27]
    B8[x][i][5,7]=J[x].iloc[i,26]
    B8[x][i][6,0]=J[x].iloc[i,11]
    B8[x][i][6,2]=J[x].iloc[i,10]
    B8[x][i][6,4]=J[x].iloc[i,15]
    B8[x][i][6,6]=J[x].iloc[i,28]
    B8[x][i][6,8]=J[x].iloc[i,29]
    B8[x][i][7,3]=J[x].iloc[i,12]
    B8[x][i][7,5]=J[x].iloc[i,30]
    B8[x][i][8,3]=J[x].iloc[i,13]
    B8[x][i][8,4]=J[x].iloc[i,14]
    B8[x][i][8,5]=J[x].iloc[i,31]


# **Subject 9**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B9 = []
for i in range(40):
  B9.append(A)
B9 = np.array(B9)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[8][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B9[x][i][0,3]=J[x].iloc[i,0]
    B9[x][i][0,5]=J[x].iloc[i,16]
    B9[x][i][1,3]=J[x].iloc[i,1]
    B9[x][i][1,5]=J[x].iloc[i,17]
    B9[x][i][2,0]=J[x].iloc[i,3]
    B9[x][i][2,2]=J[x].iloc[i,2]
    B9[x][i][2,4]=J[x].iloc[i,18]
    B9[x][i][2,6]=J[x].iloc[i,19]
    B9[x][i][2,8]=J[x].iloc[i,20]
    B9[x][i][3,1]=J[x].iloc[i,4]
    B9[x][i][3,3]=J[x].iloc[i,5]
    B9[x][i][3,5]=J[x].iloc[i,22]
    B9[x][i][3,7]=J[x].iloc[i,21]
    B9[x][i][4,0]=J[x].iloc[i,7]
    B9[x][i][4,2]=J[x].iloc[i,6]
    B9[x][i][4,4]=J[x].iloc[i,23]
    B9[x][i][4,6]=J[x].iloc[i,24]
    B9[x][i][4,8]=J[x].iloc[i,25]
    B9[x][i][5,1]=J[x].iloc[i,8]
    B9[x][i][5,3]=J[x].iloc[i,9]
    B9[x][i][5,5]=J[x].iloc[i,27]
    B9[x][i][5,7]=J[x].iloc[i,26]
    B9[x][i][6,0]=J[x].iloc[i,11]
    B9[x][i][6,2]=J[x].iloc[i,10]
    B9[x][i][6,4]=J[x].iloc[i,15]
    B9[x][i][6,6]=J[x].iloc[i,28]
    B9[x][i][6,8]=J[x].iloc[i,29]
    B9[x][i][7,3]=J[x].iloc[i,12]
    B9[x][i][7,5]=J[x].iloc[i,30]
    B9[x][i][8,3]=J[x].iloc[i,13]
    B9[x][i][8,4]=J[x].iloc[i,14]
    B9[x][i][8,5]=J[x].iloc[i,31]


# **Subect 10**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B10 = []
for i in range(40):
  B10.append(A)
B10 = np.array(B10)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[9][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B10[x][i][0,3]=J[x].iloc[i,0]
    B10[x][i][0,5]=J[x].iloc[i,16]
    B10[x][i][1,3]=J[x].iloc[i,1]
    B10[x][i][1,5]=J[x].iloc[i,17]
    B10[x][i][2,0]=J[x].iloc[i,3]
    B10[x][i][2,2]=J[x].iloc[i,2]
    B10[x][i][2,4]=J[x].iloc[i,18]
    B10[x][i][2,6]=J[x].iloc[i,19]
    B10[x][i][2,8]=J[x].iloc[i,20]
    B10[x][i][3,1]=J[x].iloc[i,4]
    B10[x][i][3,3]=J[x].iloc[i,5]
    B10[x][i][3,5]=J[x].iloc[i,22]
    B10[x][i][3,7]=J[x].iloc[i,21]
    B10[x][i][4,0]=J[x].iloc[i,7]
    B10[x][i][4,2]=J[x].iloc[i,6]
    B10[x][i][4,4]=J[x].iloc[i,23]
    B10[x][i][4,6]=J[x].iloc[i,24]
    B10[x][i][4,8]=J[x].iloc[i,25]
    B10[x][i][5,1]=J[x].iloc[i,8]
    B10[x][i][5,3]=J[x].iloc[i,9]
    B10[x][i][5,5]=J[x].iloc[i,27]
    B10[x][i][5,7]=J[x].iloc[i,26]
    B10[x][i][6,0]=J[x].iloc[i,11]
    B10[x][i][6,2]=J[x].iloc[i,10]
    B10[x][i][6,4]=J[x].iloc[i,15]
    B10[x][i][6,6]=J[x].iloc[i,28]
    B10[x][i][6,8]=J[x].iloc[i,29]
    B10[x][i][7,3]=J[x].iloc[i,12]
    B10[x][i][7,5]=J[x].iloc[i,30]
    B10[x][i][8,3]=J[x].iloc[i,13]
    B10[x][i][8,4]=J[x].iloc[i,14]
    B10[x][i][8,5]=J[x].iloc[i,31]


# **Subject 11**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B11 = []
for i in range(40):
  B11.append(A)
B11 = np.array(B11)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[10][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B11[x][i][0,3]=J[x].iloc[i,0]
    B11[x][i][0,5]=J[x].iloc[i,16]
    B11[x][i][1,3]=J[x].iloc[i,1]
    B11[x][i][1,5]=J[x].iloc[i,17]
    B11[x][i][2,0]=J[x].iloc[i,3]
    B11[x][i][2,2]=J[x].iloc[i,2]
    B11[x][i][2,4]=J[x].iloc[i,18]
    B11[x][i][2,6]=J[x].iloc[i,19]
    B11[x][i][2,8]=J[x].iloc[i,20]
    B11[x][i][3,1]=J[x].iloc[i,4]
    B11[x][i][3,3]=J[x].iloc[i,5]
    B11[x][i][3,5]=J[x].iloc[i,22]
    B11[x][i][3,7]=J[x].iloc[i,21]
    B11[x][i][4,0]=J[x].iloc[i,7]
    B11[x][i][4,2]=J[x].iloc[i,6]
    B11[x][i][4,4]=J[x].iloc[i,23]
    B11[x][i][4,6]=J[x].iloc[i,24]
    B11[x][i][4,8]=J[x].iloc[i,25]
    B11[x][i][5,1]=J[x].iloc[i,8]
    B11[x][i][5,3]=J[x].iloc[i,9]
    B11[x][i][5,5]=J[x].iloc[i,27]
    B11[x][i][5,7]=J[x].iloc[i,26]
    B11[x][i][6,0]=J[x].iloc[i,11]
    B11[x][i][6,2]=J[x].iloc[i,10]
    B11[x][i][6,4]=J[x].iloc[i,15]
    B11[x][i][6,6]=J[x].iloc[i,28]
    B11[x][i][6,8]=J[x].iloc[i,29]
    B11[x][i][7,3]=J[x].iloc[i,12]
    B11[x][i][7,5]=J[x].iloc[i,30]
    B11[x][i][8,3]=J[x].iloc[i,13]
    B11[x][i][8,4]=J[x].iloc[i,14]
    B11[x][i][8,5]=J[x].iloc[i,31]


# **Subject 12**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B12 = []
for i in range(40):
  B12.append(A)
B12 = np.array(B12)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[11][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B12[x][i][0,3]=J[x].iloc[i,0]
    B12[x][i][0,5]=J[x].iloc[i,16]
    B12[x][i][1,3]=J[x].iloc[i,1]
    B12[x][i][1,5]=J[x].iloc[i,17]
    B12[x][i][2,0]=J[x].iloc[i,3]
    B12[x][i][2,2]=J[x].iloc[i,2]
    B12[x][i][2,4]=J[x].iloc[i,18]
    B12[x][i][2,6]=J[x].iloc[i,19]
    B12[x][i][2,8]=J[x].iloc[i,20]
    B12[x][i][3,1]=J[x].iloc[i,4]
    B12[x][i][3,3]=J[x].iloc[i,5]
    B12[x][i][3,5]=J[x].iloc[i,22]
    B12[x][i][3,7]=J[x].iloc[i,21]
    B12[x][i][4,0]=J[x].iloc[i,7]
    B12[x][i][4,2]=J[x].iloc[i,6]
    B12[x][i][4,4]=J[x].iloc[i,23]
    B12[x][i][4,6]=J[x].iloc[i,24]
    B12[x][i][4,8]=J[x].iloc[i,25]
    B12[x][i][5,1]=J[x].iloc[i,8]
    B12[x][i][5,3]=J[x].iloc[i,9]
    B12[x][i][5,5]=J[x].iloc[i,27]
    B12[x][i][5,7]=J[x].iloc[i,26]
    B12[x][i][6,0]=J[x].iloc[i,11]
    B12[x][i][6,2]=J[x].iloc[i,10]
    B12[x][i][6,4]=J[x].iloc[i,15]
    B12[x][i][6,6]=J[x].iloc[i,28]
    B12[x][i][6,8]=J[x].iloc[i,29]
    B12[x][i][7,3]=J[x].iloc[i,12]
    B12[x][i][7,5]=J[x].iloc[i,30]
    B12[x][i][8,3]=J[x].iloc[i,13]
    B12[x][i][8,4]=J[x].iloc[i,14]
    B12[x][i][8,5]=J[x].iloc[i,31]


# **Subject 13**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B13 = []
for i in range(40):
  B13.append(A)
B13 = np.array(B13)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[12][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B13[x][i][0,3]=J[x].iloc[i,0]
    B13[x][i][0,5]=J[x].iloc[i,16]
    B13[x][i][1,3]=J[x].iloc[i,1]
    B13[x][i][1,5]=J[x].iloc[i,17]
    B13[x][i][2,0]=J[x].iloc[i,3]
    B13[x][i][2,2]=J[x].iloc[i,2]
    B13[x][i][2,4]=J[x].iloc[i,18]
    B13[x][i][2,6]=J[x].iloc[i,19]
    B13[x][i][2,8]=J[x].iloc[i,20]
    B13[x][i][3,1]=J[x].iloc[i,4]
    B13[x][i][3,3]=J[x].iloc[i,5]
    B13[x][i][3,5]=J[x].iloc[i,22]
    B13[x][i][3,7]=J[x].iloc[i,21]
    B13[x][i][4,0]=J[x].iloc[i,7]
    B13[x][i][4,2]=J[x].iloc[i,6]
    B13[x][i][4,4]=J[x].iloc[i,23]
    B13[x][i][4,6]=J[x].iloc[i,24]
    B13[x][i][4,8]=J[x].iloc[i,25]
    B13[x][i][5,1]=J[x].iloc[i,8]
    B13[x][i][5,3]=J[x].iloc[i,9]
    B13[x][i][5,5]=J[x].iloc[i,27]
    B13[x][i][5,7]=J[x].iloc[i,26]
    B13[x][i][6,0]=J[x].iloc[i,11]
    B13[x][i][6,2]=J[x].iloc[i,10]
    B13[x][i][6,4]=J[x].iloc[i,15]
    B13[x][i][6,6]=J[x].iloc[i,28]
    B13[x][i][6,8]=J[x].iloc[i,29]
    B13[x][i][7,3]=J[x].iloc[i,12]
    B13[x][i][7,5]=J[x].iloc[i,30]
    B13[x][i][8,3]=J[x].iloc[i,13]
    B13[x][i][8,4]=J[x].iloc[i,14]
    B13[x][i][8,5]=J[x].iloc[i,31]


# **Subject 14**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B14 = []
for i in range(40):
  B14.append(A)
B14 = np.array(B14)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[13][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B14[x][i][0,3]=J[x].iloc[i,0]
    B14[x][i][0,5]=J[x].iloc[i,16]
    B14[x][i][1,3]=J[x].iloc[i,1]
    B14[x][i][1,5]=J[x].iloc[i,17]
    B14[x][i][2,0]=J[x].iloc[i,3]
    B14[x][i][2,2]=J[x].iloc[i,2]
    B14[x][i][2,4]=J[x].iloc[i,18]
    B14[x][i][2,6]=J[x].iloc[i,19]
    B14[x][i][2,8]=J[x].iloc[i,20]
    B14[x][i][3,1]=J[x].iloc[i,4]
    B14[x][i][3,3]=J[x].iloc[i,5]
    B14[x][i][3,5]=J[x].iloc[i,22]
    B14[x][i][3,7]=J[x].iloc[i,21]
    B14[x][i][4,0]=J[x].iloc[i,7]
    B14[x][i][4,2]=J[x].iloc[i,6]
    B14[x][i][4,4]=J[x].iloc[i,23]
    B14[x][i][4,6]=J[x].iloc[i,24]
    B14[x][i][4,8]=J[x].iloc[i,25]
    B14[x][i][5,1]=J[x].iloc[i,8]
    B14[x][i][5,3]=J[x].iloc[i,9]
    B14[x][i][5,5]=J[x].iloc[i,27]
    B14[x][i][5,7]=J[x].iloc[i,26]
    B14[x][i][6,0]=J[x].iloc[i,11]
    B14[x][i][6,2]=J[x].iloc[i,10]
    B14[x][i][6,4]=J[x].iloc[i,15]
    B14[x][i][6,6]=J[x].iloc[i,28]
    B14[x][i][6,8]=J[x].iloc[i,29]
    B14[x][i][7,3]=J[x].iloc[i,12]
    B14[x][i][7,5]=J[x].iloc[i,30]
    B14[x][i][8,3]=J[x].iloc[i,13]
    B14[x][i][8,4]=J[x].iloc[i,14]
    B14[x][i][8,5]=J[x].iloc[i,31]


# **Subject 15**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B15 = []
for i in range(40):
  B15.append(A)
B15 = np.array(B15)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[14][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B15[x][i][0,3]=J[x].iloc[i,0]
    B15[x][i][0,5]=J[x].iloc[i,16]
    B15[x][i][1,3]=J[x].iloc[i,1]
    B15[x][i][1,5]=J[x].iloc[i,17]
    B15[x][i][2,0]=J[x].iloc[i,3]
    B15[x][i][2,2]=J[x].iloc[i,2]
    B15[x][i][2,4]=J[x].iloc[i,18]
    B15[x][i][2,6]=J[x].iloc[i,19]
    B15[x][i][2,8]=J[x].iloc[i,20]
    B15[x][i][3,1]=J[x].iloc[i,4]
    B15[x][i][3,3]=J[x].iloc[i,5]
    B15[x][i][3,5]=J[x].iloc[i,22]
    B15[x][i][3,7]=J[x].iloc[i,21]
    B15[x][i][4,0]=J[x].iloc[i,7]
    B15[x][i][4,2]=J[x].iloc[i,6]
    B15[x][i][4,4]=J[x].iloc[i,23]
    B15[x][i][4,6]=J[x].iloc[i,24]
    B15[x][i][4,8]=J[x].iloc[i,25]
    B15[x][i][5,1]=J[x].iloc[i,8]
    B15[x][i][5,3]=J[x].iloc[i,9]
    B15[x][i][5,5]=J[x].iloc[i,27]
    B15[x][i][5,7]=J[x].iloc[i,26]
    B15[x][i][6,0]=J[x].iloc[i,11]
    B15[x][i][6,2]=J[x].iloc[i,10]
    B15[x][i][6,4]=J[x].iloc[i,15]
    B15[x][i][6,6]=J[x].iloc[i,28]
    B15[x][i][6,8]=J[x].iloc[i,29]
    B15[x][i][7,3]=J[x].iloc[i,12]
    B15[x][i][7,5]=J[x].iloc[i,30]
    B15[x][i][8,3]=J[x].iloc[i,13]
    B15[x][i][8,4]=J[x].iloc[i,14]
    B15[x][i][8,5]=J[x].iloc[i,31]


# **Subject 16**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B16 = []
for i in range(40):
  B16.append(A)
B16 = np.array(B16)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[15][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B16[x][i][0,3]=J[x].iloc[i,0]
    B16[x][i][0,5]=J[x].iloc[i,16]
    B16[x][i][1,3]=J[x].iloc[i,1]
    B16[x][i][1,5]=J[x].iloc[i,17]
    B16[x][i][2,0]=J[x].iloc[i,3]
    B16[x][i][2,2]=J[x].iloc[i,2]
    B16[x][i][2,4]=J[x].iloc[i,18]
    B16[x][i][2,6]=J[x].iloc[i,19]
    B16[x][i][2,8]=J[x].iloc[i,20]
    B16[x][i][3,1]=J[x].iloc[i,4]
    B16[x][i][3,3]=J[x].iloc[i,5]
    B16[x][i][3,5]=J[x].iloc[i,22]
    B16[x][i][3,7]=J[x].iloc[i,21]
    B16[x][i][4,0]=J[x].iloc[i,7]
    B16[x][i][4,2]=J[x].iloc[i,6]
    B16[x][i][4,4]=J[x].iloc[i,23]
    B16[x][i][4,6]=J[x].iloc[i,24]
    B16[x][i][4,8]=J[x].iloc[i,25]
    B16[x][i][5,1]=J[x].iloc[i,8]
    B16[x][i][5,3]=J[x].iloc[i,9]
    B16[x][i][5,5]=J[x].iloc[i,27]
    B16[x][i][5,7]=J[x].iloc[i,26]
    B16[x][i][6,0]=J[x].iloc[i,11]
    B16[x][i][6,2]=J[x].iloc[i,10]
    B16[x][i][6,4]=J[x].iloc[i,15]
    B16[x][i][6,6]=J[x].iloc[i,28]
    B16[x][i][6,8]=J[x].iloc[i,29]
    B16[x][i][7,3]=J[x].iloc[i,12]
    B16[x][i][7,5]=J[x].iloc[i,30]
    B16[x][i][8,3]=J[x].iloc[i,13]
    B16[x][i][8,4]=J[x].iloc[i,14]
    B16[x][i][8,5]=J[x].iloc[i,31]


# **Subject 17**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B17 = []
for i in range(40):
  B17.append(A)
B17 = np.array(B17)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[16][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B17[x][i][0,3]=J[x].iloc[i,0]
    B17[x][i][0,5]=J[x].iloc[i,16]
    B17[x][i][1,3]=J[x].iloc[i,1]
    B17[x][i][1,5]=J[x].iloc[i,17]
    B17[x][i][2,0]=J[x].iloc[i,3]
    B17[x][i][2,2]=J[x].iloc[i,2]
    B17[x][i][2,4]=J[x].iloc[i,18]
    B17[x][i][2,6]=J[x].iloc[i,19]
    B17[x][i][2,8]=J[x].iloc[i,20]
    B17[x][i][3,1]=J[x].iloc[i,4]
    B17[x][i][3,3]=J[x].iloc[i,5]
    B17[x][i][3,5]=J[x].iloc[i,22]
    B17[x][i][3,7]=J[x].iloc[i,21]
    B17[x][i][4,0]=J[x].iloc[i,7]
    B17[x][i][4,2]=J[x].iloc[i,6]
    B17[x][i][4,4]=J[x].iloc[i,23]
    B17[x][i][4,6]=J[x].iloc[i,24]
    B17[x][i][4,8]=J[x].iloc[i,25]
    B17[x][i][5,1]=J[x].iloc[i,8]
    B17[x][i][5,3]=J[x].iloc[i,9]
    B17[x][i][5,5]=J[x].iloc[i,27]
    B17[x][i][5,7]=J[x].iloc[i,26]
    B17[x][i][6,0]=J[x].iloc[i,11]
    B17[x][i][6,2]=J[x].iloc[i,10]
    B17[x][i][6,4]=J[x].iloc[i,15]
    B17[x][i][6,6]=J[x].iloc[i,28]
    B17[x][i][6,8]=J[x].iloc[i,29]
    B17[x][i][7,3]=J[x].iloc[i,12]
    B17[x][i][7,5]=J[x].iloc[i,30]
    B17[x][i][8,3]=J[x].iloc[i,13]
    B17[x][i][8,4]=J[x].iloc[i,14]
    B17[x][i][8,5]=J[x].iloc[i,31]


# **Subject 18**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B18 = []
for i in range(40):
  B18.append(A)
B18 = np.array(B18)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[17][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B18[x][i][0,3]=J[x].iloc[i,0]
    B18[x][i][0,5]=J[x].iloc[i,16]
    B18[x][i][1,3]=J[x].iloc[i,1]
    B18[x][i][1,5]=J[x].iloc[i,17]
    B18[x][i][2,0]=J[x].iloc[i,3]
    B18[x][i][2,2]=J[x].iloc[i,2]
    B18[x][i][2,4]=J[x].iloc[i,18]
    B18[x][i][2,6]=J[x].iloc[i,19]
    B18[x][i][2,8]=J[x].iloc[i,20]
    B18[x][i][3,1]=J[x].iloc[i,4]
    B18[x][i][3,3]=J[x].iloc[i,5]
    B18[x][i][3,5]=J[x].iloc[i,22]
    B18[x][i][3,7]=J[x].iloc[i,21]
    B18[x][i][4,0]=J[x].iloc[i,7]
    B18[x][i][4,2]=J[x].iloc[i,6]
    B18[x][i][4,4]=J[x].iloc[i,23]
    B18[x][i][4,6]=J[x].iloc[i,24]
    B18[x][i][4,8]=J[x].iloc[i,25]
    B18[x][i][5,1]=J[x].iloc[i,8]
    B18[x][i][5,3]=J[x].iloc[i,9]
    B18[x][i][5,5]=J[x].iloc[i,27]
    B18[x][i][5,7]=J[x].iloc[i,26]
    B18[x][i][6,0]=J[x].iloc[i,11]
    B18[x][i][6,2]=J[x].iloc[i,10]
    B18[x][i][6,4]=J[x].iloc[i,15]
    B18[x][i][6,6]=J[x].iloc[i,28]
    B18[x][i][6,8]=J[x].iloc[i,29]
    B18[x][i][7,3]=J[x].iloc[i,12]
    B18[x][i][7,5]=J[x].iloc[i,30]
    B18[x][i][8,3]=J[x].iloc[i,13]
    B18[x][i][8,4]=J[x].iloc[i,14]
    B18[x][i][8,5]=J[x].iloc[i,31]


# **Subject 19**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B19 = []
for i in range(40):
  B19.append(A)
B19 = np.array(B19)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[18][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B19[x][i][0,3]=J[x].iloc[i,0]
    B19[x][i][0,5]=J[x].iloc[i,16]
    B19[x][i][1,3]=J[x].iloc[i,1]
    B19[x][i][1,5]=J[x].iloc[i,17]
    B19[x][i][2,0]=J[x].iloc[i,3]
    B19[x][i][2,2]=J[x].iloc[i,2]
    B19[x][i][2,4]=J[x].iloc[i,18]
    B19[x][i][2,6]=J[x].iloc[i,19]
    B19[x][i][2,8]=J[x].iloc[i,20]
    B19[x][i][3,1]=J[x].iloc[i,4]
    B19[x][i][3,3]=J[x].iloc[i,5]
    B19[x][i][3,5]=J[x].iloc[i,22]
    B19[x][i][3,7]=J[x].iloc[i,21]
    B19[x][i][4,0]=J[x].iloc[i,7]
    B19[x][i][4,2]=J[x].iloc[i,6]
    B19[x][i][4,4]=J[x].iloc[i,23]
    B19[x][i][4,6]=J[x].iloc[i,24]
    B19[x][i][4,8]=J[x].iloc[i,25]
    B19[x][i][5,1]=J[x].iloc[i,8]
    B19[x][i][5,3]=J[x].iloc[i,9]
    B19[x][i][5,5]=J[x].iloc[i,27]
    B19[x][i][5,7]=J[x].iloc[i,26]
    B19[x][i][6,0]=J[x].iloc[i,11]
    B19[x][i][6,2]=J[x].iloc[i,10]
    B19[x][i][6,4]=J[x].iloc[i,15]
    B19[x][i][6,6]=J[x].iloc[i,28]
    B19[x][i][6,8]=J[x].iloc[i,29]
    B19[x][i][7,3]=J[x].iloc[i,12]
    B19[x][i][7,5]=J[x].iloc[i,30]
    B19[x][i][8,3]=J[x].iloc[i,13]
    B19[x][i][8,4]=J[x].iloc[i,14]
    B19[x][i][8,5]=J[x].iloc[i,31]


# **Subject 20**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B20 = []
for i in range(40):
  B20.append(A)
B20 = np.array(B20)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[19][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B20[x][i][0,3]=J[x].iloc[i,0]
    B20[x][i][0,5]=J[x].iloc[i,16]
    B20[x][i][1,3]=J[x].iloc[i,1]
    B20[x][i][1,5]=J[x].iloc[i,17]
    B20[x][i][2,0]=J[x].iloc[i,3]
    B20[x][i][2,2]=J[x].iloc[i,2]
    B20[x][i][2,4]=J[x].iloc[i,18]
    B20[x][i][2,6]=J[x].iloc[i,19]
    B20[x][i][2,8]=J[x].iloc[i,20]
    B20[x][i][3,1]=J[x].iloc[i,4]
    B20[x][i][3,3]=J[x].iloc[i,5]
    B20[x][i][3,5]=J[x].iloc[i,22]
    B20[x][i][3,7]=J[x].iloc[i,21]
    B20[x][i][4,0]=J[x].iloc[i,7]
    B20[x][i][4,2]=J[x].iloc[i,6]
    B20[x][i][4,4]=J[x].iloc[i,23]
    B20[x][i][4,6]=J[x].iloc[i,24]
    B20[x][i][4,8]=J[x].iloc[i,25]
    B20[x][i][5,1]=J[x].iloc[i,8]
    B20[x][i][5,3]=J[x].iloc[i,9]
    B20[x][i][5,5]=J[x].iloc[i,27]
    B20[x][i][5,7]=J[x].iloc[i,26]
    B20[x][i][6,0]=J[x].iloc[i,11]
    B20[x][i][6,2]=J[x].iloc[i,10]
    B20[x][i][6,4]=J[x].iloc[i,15]
    B20[x][i][6,6]=J[x].iloc[i,28]
    B20[x][i][6,8]=J[x].iloc[i,29]
    B20[x][i][7,3]=J[x].iloc[i,12]
    B20[x][i][7,5]=J[x].iloc[i,30]
    B20[x][i][8,3]=J[x].iloc[i,13]
    B20[x][i][8,4]=J[x].iloc[i,14]
    B20[x][i][8,5]=J[x].iloc[i,31]


# **Subject 21**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B21 = []
for i in range(40):
  B21.append(A)
B21 = np.array(B21)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[20][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B21[x][i][0,3]=J[x].iloc[i,0]
    B21[x][i][0,5]=J[x].iloc[i,16]
    B21[x][i][1,3]=J[x].iloc[i,1]
    B21[x][i][1,5]=J[x].iloc[i,17]
    B21[x][i][2,0]=J[x].iloc[i,3]
    B21[x][i][2,2]=J[x].iloc[i,2]
    B21[x][i][2,4]=J[x].iloc[i,18]
    B21[x][i][2,6]=J[x].iloc[i,19]
    B21[x][i][2,8]=J[x].iloc[i,20]
    B21[x][i][3,1]=J[x].iloc[i,4]
    B21[x][i][3,3]=J[x].iloc[i,5]
    B21[x][i][3,5]=J[x].iloc[i,22]
    B21[x][i][3,7]=J[x].iloc[i,21]
    B21[x][i][4,0]=J[x].iloc[i,7]
    B21[x][i][4,2]=J[x].iloc[i,6]
    B21[x][i][4,4]=J[x].iloc[i,23]
    B21[x][i][4,6]=J[x].iloc[i,24]
    B21[x][i][4,8]=J[x].iloc[i,25]
    B21[x][i][5,1]=J[x].iloc[i,8]
    B21[x][i][5,3]=J[x].iloc[i,9]
    B21[x][i][5,5]=J[x].iloc[i,27]
    B21[x][i][5,7]=J[x].iloc[i,26]
    B21[x][i][6,0]=J[x].iloc[i,11]
    B21[x][i][6,2]=J[x].iloc[i,10]
    B21[x][i][6,4]=J[x].iloc[i,15]
    B21[x][i][6,6]=J[x].iloc[i,28]
    B21[x][i][6,8]=J[x].iloc[i,29]
    B21[x][i][7,3]=J[x].iloc[i,12]
    B21[x][i][7,5]=J[x].iloc[i,30]
    B21[x][i][8,3]=J[x].iloc[i,13]
    B21[x][i][8,4]=J[x].iloc[i,14]
    B21[x][i][8,5]=J[x].iloc[i,31]


# **Subject 22**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B22 = []
for i in range(40):
  B22.append(A)
B22 = np.array(B22)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[21][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B22[x][i][0,3]=J[x].iloc[i,0]
    B22[x][i][0,5]=J[x].iloc[i,16]
    B22[x][i][1,3]=J[x].iloc[i,1]
    B22[x][i][1,5]=J[x].iloc[i,17]
    B22[x][i][2,0]=J[x].iloc[i,3]
    B22[x][i][2,2]=J[x].iloc[i,2]
    B22[x][i][2,4]=J[x].iloc[i,18]
    B22[x][i][2,6]=J[x].iloc[i,19]
    B22[x][i][2,8]=J[x].iloc[i,20]
    B22[x][i][3,1]=J[x].iloc[i,4]
    B22[x][i][3,3]=J[x].iloc[i,5]
    B22[x][i][3,5]=J[x].iloc[i,22]
    B22[x][i][3,7]=J[x].iloc[i,21]
    B22[x][i][4,0]=J[x].iloc[i,7]
    B22[x][i][4,2]=J[x].iloc[i,6]
    B22[x][i][4,4]=J[x].iloc[i,23]
    B22[x][i][4,6]=J[x].iloc[i,24]
    B22[x][i][4,8]=J[x].iloc[i,25]
    B22[x][i][5,1]=J[x].iloc[i,8]
    B22[x][i][5,3]=J[x].iloc[i,9]
    B22[x][i][5,5]=J[x].iloc[i,27]
    B22[x][i][5,7]=J[x].iloc[i,26]
    B22[x][i][6,0]=J[x].iloc[i,11]
    B22[x][i][6,2]=J[x].iloc[i,10]
    B22[x][i][6,4]=J[x].iloc[i,15]
    B22[x][i][6,6]=J[x].iloc[i,28]
    B22[x][i][6,8]=J[x].iloc[i,29]
    B22[x][i][7,3]=J[x].iloc[i,12]
    B22[x][i][7,5]=J[x].iloc[i,30]
    B22[x][i][8,3]=J[x].iloc[i,13]
    B22[x][i][8,4]=J[x].iloc[i,14]
    B22[x][i][8,5]=J[x].iloc[i,31]


# **Subject 23**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B23 = []
for i in range(40):
  B23.append(A)
B23 = np.array(B23)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[22][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B23[x][i][0,3]=J[x].iloc[i,0]
    B23[x][i][0,5]=J[x].iloc[i,16]
    B23[x][i][1,3]=J[x].iloc[i,1]
    B23[x][i][1,5]=J[x].iloc[i,17]
    B23[x][i][2,0]=J[x].iloc[i,3]
    B23[x][i][2,2]=J[x].iloc[i,2]
    B23[x][i][2,4]=J[x].iloc[i,18]
    B23[x][i][2,6]=J[x].iloc[i,19]
    B23[x][i][2,8]=J[x].iloc[i,20]
    B23[x][i][3,1]=J[x].iloc[i,4]
    B23[x][i][3,3]=J[x].iloc[i,5]
    B23[x][i][3,5]=J[x].iloc[i,22]
    B23[x][i][3,7]=J[x].iloc[i,21]
    B23[x][i][4,0]=J[x].iloc[i,7]
    B23[x][i][4,2]=J[x].iloc[i,6]
    B23[x][i][4,4]=J[x].iloc[i,23]
    B23[x][i][4,6]=J[x].iloc[i,24]
    B23[x][i][4,8]=J[x].iloc[i,25]
    B23[x][i][5,1]=J[x].iloc[i,8]
    B23[x][i][5,3]=J[x].iloc[i,9]
    B23[x][i][5,5]=J[x].iloc[i,27]
    B23[x][i][5,7]=J[x].iloc[i,26]
    B23[x][i][6,0]=J[x].iloc[i,11]
    B23[x][i][6,2]=J[x].iloc[i,10]
    B23[x][i][6,4]=J[x].iloc[i,15]
    B23[x][i][6,6]=J[x].iloc[i,28]
    B23[x][i][6,8]=J[x].iloc[i,29]
    B23[x][i][7,3]=J[x].iloc[i,12]
    B23[x][i][7,5]=J[x].iloc[i,30]
    B23[x][i][8,3]=J[x].iloc[i,13]
    B23[x][i][8,4]=J[x].iloc[i,14]
    B23[x][i][8,5]=J[x].iloc[i,31]


# **Subject 24**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B24 = []
for i in range(40):
  B24.append(A)
B24 = np.array(B24)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[23][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B24[x][i][0,3]=J[x].iloc[i,0]
    B24[x][i][0,5]=J[x].iloc[i,16]
    B24[x][i][1,3]=J[x].iloc[i,1]
    B24[x][i][1,5]=J[x].iloc[i,17]
    B24[x][i][2,0]=J[x].iloc[i,3]
    B24[x][i][2,2]=J[x].iloc[i,2]
    B24[x][i][2,4]=J[x].iloc[i,18]
    B24[x][i][2,6]=J[x].iloc[i,19]
    B24[x][i][2,8]=J[x].iloc[i,20]
    B24[x][i][3,1]=J[x].iloc[i,4]
    B24[x][i][3,3]=J[x].iloc[i,5]
    B24[x][i][3,5]=J[x].iloc[i,22]
    B24[x][i][3,7]=J[x].iloc[i,21]
    B24[x][i][4,0]=J[x].iloc[i,7]
    B24[x][i][4,2]=J[x].iloc[i,6]
    B24[x][i][4,4]=J[x].iloc[i,23]
    B24[x][i][4,6]=J[x].iloc[i,24]
    B24[x][i][4,8]=J[x].iloc[i,25]
    B24[x][i][5,1]=J[x].iloc[i,8]
    B24[x][i][5,3]=J[x].iloc[i,9]
    B24[x][i][5,5]=J[x].iloc[i,27]
    B24[x][i][5,7]=J[x].iloc[i,26]
    B24[x][i][6,0]=J[x].iloc[i,11]
    B24[x][i][6,2]=J[x].iloc[i,10]
    B24[x][i][6,4]=J[x].iloc[i,15]
    B24[x][i][6,6]=J[x].iloc[i,28]
    B24[x][i][6,8]=J[x].iloc[i,29]
    B24[x][i][7,3]=J[x].iloc[i,12]
    B24[x][i][7,5]=J[x].iloc[i,30]
    B24[x][i][8,3]=J[x].iloc[i,13]
    B24[x][i][8,4]=J[x].iloc[i,14]
    B24[x][i][8,5]=J[x].iloc[i,31]


# **Subject 25**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B25 = []
for i in range(40):
  B25.append(A)
B25 = np.array(B25)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[24][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B25[x][i][0,3]=J[x].iloc[i,0]
    B25[x][i][0,5]=J[x].iloc[i,16]
    B25[x][i][1,3]=J[x].iloc[i,1]
    B25[x][i][1,5]=J[x].iloc[i,17]
    B25[x][i][2,0]=J[x].iloc[i,3]
    B25[x][i][2,2]=J[x].iloc[i,2]
    B25[x][i][2,4]=J[x].iloc[i,18]
    B25[x][i][2,6]=J[x].iloc[i,19]
    B25[x][i][2,8]=J[x].iloc[i,20]
    B25[x][i][3,1]=J[x].iloc[i,4]
    B25[x][i][3,3]=J[x].iloc[i,5]
    B25[x][i][3,5]=J[x].iloc[i,22]
    B25[x][i][3,7]=J[x].iloc[i,21]
    B25[x][i][4,0]=J[x].iloc[i,7]
    B25[x][i][4,2]=J[x].iloc[i,6]
    B25[x][i][4,4]=J[x].iloc[i,23]
    B25[x][i][4,6]=J[x].iloc[i,24]
    B25[x][i][4,8]=J[x].iloc[i,25]
    B25[x][i][5,1]=J[x].iloc[i,8]
    B25[x][i][5,3]=J[x].iloc[i,9]
    B25[x][i][5,5]=J[x].iloc[i,27]
    B25[x][i][5,7]=J[x].iloc[i,26]
    B25[x][i][6,0]=J[x].iloc[i,11]
    B25[x][i][6,2]=J[x].iloc[i,10]
    B25[x][i][6,4]=J[x].iloc[i,15]
    B25[x][i][6,6]=J[x].iloc[i,28]
    B25[x][i][6,8]=J[x].iloc[i,29]
    B25[x][i][7,3]=J[x].iloc[i,12]
    B25[x][i][7,5]=J[x].iloc[i,30]
    B25[x][i][8,3]=J[x].iloc[i,13]
    B25[x][i][8,4]=J[x].iloc[i,14]
    B25[x][i][8,5]=J[x].iloc[i,31]


# **Subject 26**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B26 = []
for i in range(40):
  B26.append(A)
B26 = np.array(B26)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[25][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B26[x][i][0,3]=J[x].iloc[i,0]
    B26[x][i][0,5]=J[x].iloc[i,16]
    B26[x][i][1,3]=J[x].iloc[i,1]
    B26[x][i][1,5]=J[x].iloc[i,17]
    B26[x][i][2,0]=J[x].iloc[i,3]
    B26[x][i][2,2]=J[x].iloc[i,2]
    B26[x][i][2,4]=J[x].iloc[i,18]
    B26[x][i][2,6]=J[x].iloc[i,19]
    B26[x][i][2,8]=J[x].iloc[i,20]
    B26[x][i][3,1]=J[x].iloc[i,4]
    B26[x][i][3,3]=J[x].iloc[i,5]
    B26[x][i][3,5]=J[x].iloc[i,22]
    B26[x][i][3,7]=J[x].iloc[i,21]
    B26[x][i][4,0]=J[x].iloc[i,7]
    B26[x][i][4,2]=J[x].iloc[i,6]
    B26[x][i][4,4]=J[x].iloc[i,23]
    B26[x][i][4,6]=J[x].iloc[i,24]
    B26[x][i][4,8]=J[x].iloc[i,25]
    B26[x][i][5,1]=J[x].iloc[i,8]
    B26[x][i][5,3]=J[x].iloc[i,9]
    B26[x][i][5,5]=J[x].iloc[i,27]
    B26[x][i][5,7]=J[x].iloc[i,26]
    B26[x][i][6,0]=J[x].iloc[i,11]
    B26[x][i][6,2]=J[x].iloc[i,10]
    B26[x][i][6,4]=J[x].iloc[i,15]
    B26[x][i][6,6]=J[x].iloc[i,28]
    B26[x][i][6,8]=J[x].iloc[i,29]
    B26[x][i][7,3]=J[x].iloc[i,12]
    B26[x][i][7,5]=J[x].iloc[i,30]
    B26[x][i][8,3]=J[x].iloc[i,13]
    B26[x][i][8,4]=J[x].iloc[i,14]
    B26[x][i][8,5]=J[x].iloc[i,31]


# **Subject 27**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B27 = []
for i in range(40):
  B27.append(A)
B27 = np.array(B27)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[26][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B27[x][i][0,3]=J[x].iloc[i,0]
    B27[x][i][0,5]=J[x].iloc[i,16]
    B27[x][i][1,3]=J[x].iloc[i,1]
    B27[x][i][1,5]=J[x].iloc[i,17]
    B27[x][i][2,0]=J[x].iloc[i,3]
    B27[x][i][2,2]=J[x].iloc[i,2]
    B27[x][i][2,4]=J[x].iloc[i,18]
    B27[x][i][2,6]=J[x].iloc[i,19]
    B27[x][i][2,8]=J[x].iloc[i,20]
    B27[x][i][3,1]=J[x].iloc[i,4]
    B27[x][i][3,3]=J[x].iloc[i,5]
    B27[x][i][3,5]=J[x].iloc[i,22]
    B27[x][i][3,7]=J[x].iloc[i,21]
    B27[x][i][4,0]=J[x].iloc[i,7]
    B27[x][i][4,2]=J[x].iloc[i,6]
    B27[x][i][4,4]=J[x].iloc[i,23]
    B27[x][i][4,6]=J[x].iloc[i,24]
    B27[x][i][4,8]=J[x].iloc[i,25]
    B27[x][i][5,1]=J[x].iloc[i,8]
    B27[x][i][5,3]=J[x].iloc[i,9]
    B27[x][i][5,5]=J[x].iloc[i,27]
    B27[x][i][5,7]=J[x].iloc[i,26]
    B27[x][i][6,0]=J[x].iloc[i,11]
    B27[x][i][6,2]=J[x].iloc[i,10]
    B27[x][i][6,4]=J[x].iloc[i,15]
    B27[x][i][6,6]=J[x].iloc[i,28]
    B27[x][i][6,8]=J[x].iloc[i,29]
    B27[x][i][7,3]=J[x].iloc[i,12]
    B27[x][i][7,5]=J[x].iloc[i,30]
    B27[x][i][8,3]=J[x].iloc[i,13]
    B27[x][i][8,4]=J[x].iloc[i,14]
    B27[x][i][8,5]=J[x].iloc[i,31]


# **Subject 28**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B28 = []
for i in range(40):
  B28.append(A)
B28 = np.array(B28)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[27][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B28[x][i][0,3]=J[x].iloc[i,0]
    B28[x][i][0,5]=J[x].iloc[i,16]
    B28[x][i][1,3]=J[x].iloc[i,1]
    B28[x][i][1,5]=J[x].iloc[i,17]
    B28[x][i][2,0]=J[x].iloc[i,3]
    B28[x][i][2,2]=J[x].iloc[i,2]
    B28[x][i][2,4]=J[x].iloc[i,18]
    B28[x][i][2,6]=J[x].iloc[i,19]
    B28[x][i][2,8]=J[x].iloc[i,20]
    B28[x][i][3,1]=J[x].iloc[i,4]
    B28[x][i][3,3]=J[x].iloc[i,5]
    B28[x][i][3,5]=J[x].iloc[i,22]
    B28[x][i][3,7]=J[x].iloc[i,21]
    B28[x][i][4,0]=J[x].iloc[i,7]
    B28[x][i][4,2]=J[x].iloc[i,6]
    B28[x][i][4,4]=J[x].iloc[i,23]
    B28[x][i][4,6]=J[x].iloc[i,24]
    B28[x][i][4,8]=J[x].iloc[i,25]
    B28[x][i][5,1]=J[x].iloc[i,8]
    B28[x][i][5,3]=J[x].iloc[i,9]
    B28[x][i][5,5]=J[x].iloc[i,27]
    B28[x][i][5,7]=J[x].iloc[i,26]
    B28[x][i][6,0]=J[x].iloc[i,11]
    B28[x][i][6,2]=J[x].iloc[i,10]
    B28[x][i][6,4]=J[x].iloc[i,15]
    B28[x][i][6,6]=J[x].iloc[i,28]
    B28[x][i][6,8]=J[x].iloc[i,29]
    B28[x][i][7,3]=J[x].iloc[i,12]
    B28[x][i][7,5]=J[x].iloc[i,30]
    B28[x][i][8,3]=J[x].iloc[i,13]
    B28[x][i][8,4]=J[x].iloc[i,14]
    B28[x][i][8,5]=J[x].iloc[i,31]


# **Subject 29**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B29 = []
for i in range(40):
  B29.append(A)
B29 = np.array(B29)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[28][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B29[x][i][0,3]=J[x].iloc[i,0]
    B29[x][i][0,5]=J[x].iloc[i,16]
    B29[x][i][1,3]=J[x].iloc[i,1]
    B29[x][i][1,5]=J[x].iloc[i,17]
    B29[x][i][2,0]=J[x].iloc[i,3]
    B29[x][i][2,2]=J[x].iloc[i,2]
    B29[x][i][2,4]=J[x].iloc[i,18]
    B29[x][i][2,6]=J[x].iloc[i,19]
    B29[x][i][2,8]=J[x].iloc[i,20]
    B29[x][i][3,1]=J[x].iloc[i,4]
    B29[x][i][3,3]=J[x].iloc[i,5]
    B29[x][i][3,5]=J[x].iloc[i,22]
    B29[x][i][3,7]=J[x].iloc[i,21]
    B29[x][i][4,0]=J[x].iloc[i,7]
    B29[x][i][4,2]=J[x].iloc[i,6]
    B29[x][i][4,4]=J[x].iloc[i,23]
    B29[x][i][4,6]=J[x].iloc[i,24]
    B29[x][i][4,8]=J[x].iloc[i,25]
    B29[x][i][5,1]=J[x].iloc[i,8]
    B29[x][i][5,3]=J[x].iloc[i,9]
    B29[x][i][5,5]=J[x].iloc[i,27]
    B29[x][i][5,7]=J[x].iloc[i,26]
    B29[x][i][6,0]=J[x].iloc[i,11]
    B29[x][i][6,2]=J[x].iloc[i,10]
    B29[x][i][6,4]=J[x].iloc[i,15]
    B29[x][i][6,6]=J[x].iloc[i,28]
    B29[x][i][6,8]=J[x].iloc[i,29]
    B29[x][i][7,3]=J[x].iloc[i,12]
    B29[x][i][7,5]=J[x].iloc[i,30]
    B29[x][i][8,3]=J[x].iloc[i,13]
    B29[x][i][8,4]=J[x].iloc[i,14]
    B29[x][i][8,5]=J[x].iloc[i,31]


# **Subject 30**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B30 = []
for i in range(40):
  B30.append(A)
B30 = np.array(B30)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[29][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B30[x][i][0,3]=J[x].iloc[i,0]
    B30[x][i][0,5]=J[x].iloc[i,16]
    B30[x][i][1,3]=J[x].iloc[i,1]
    B30[x][i][1,5]=J[x].iloc[i,17]
    B30[x][i][2,0]=J[x].iloc[i,3]
    B30[x][i][2,2]=J[x].iloc[i,2]
    B30[x][i][2,4]=J[x].iloc[i,18]
    B30[x][i][2,6]=J[x].iloc[i,19]
    B30[x][i][2,8]=J[x].iloc[i,20]
    B30[x][i][3,1]=J[x].iloc[i,4]
    B30[x][i][3,3]=J[x].iloc[i,5]
    B30[x][i][3,5]=J[x].iloc[i,22]
    B30[x][i][3,7]=J[x].iloc[i,21]
    B30[x][i][4,0]=J[x].iloc[i,7]
    B30[x][i][4,2]=J[x].iloc[i,6]
    B30[x][i][4,4]=J[x].iloc[i,23]
    B30[x][i][4,6]=J[x].iloc[i,24]
    B30[x][i][4,8]=J[x].iloc[i,25]
    B30[x][i][5,1]=J[x].iloc[i,8]
    B30[x][i][5,3]=J[x].iloc[i,9]
    B30[x][i][5,5]=J[x].iloc[i,27]
    B30[x][i][5,7]=J[x].iloc[i,26]
    B30[x][i][6,0]=J[x].iloc[i,11]
    B30[x][i][6,2]=J[x].iloc[i,10]
    B30[x][i][6,4]=J[x].iloc[i,15]
    B30[x][i][6,6]=J[x].iloc[i,28]
    B30[x][i][6,8]=J[x].iloc[i,29]
    B30[x][i][7,3]=J[x].iloc[i,12]
    B30[x][i][7,5]=J[x].iloc[i,30]
    B30[x][i][8,3]=J[x].iloc[i,13]
    B30[x][i][8,4]=J[x].iloc[i,14]
    B30[x][i][8,5]=J[x].iloc[i,31]


# **Subject 31**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B31 = []
for i in range(40):
  B31.append(A)
B31 = np.array(B31)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[30][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B31[x][i][0,3]=J[x].iloc[i,0]
    B31[x][i][0,5]=J[x].iloc[i,16]
    B31[x][i][1,3]=J[x].iloc[i,1]
    B31[x][i][1,5]=J[x].iloc[i,17]
    B31[x][i][2,0]=J[x].iloc[i,3]
    B31[x][i][2,2]=J[x].iloc[i,2]
    B31[x][i][2,4]=J[x].iloc[i,18]
    B31[x][i][2,6]=J[x].iloc[i,19]
    B31[x][i][2,8]=J[x].iloc[i,20]
    B31[x][i][3,1]=J[x].iloc[i,4]
    B31[x][i][3,3]=J[x].iloc[i,5]
    B31[x][i][3,5]=J[x].iloc[i,22]
    B31[x][i][3,7]=J[x].iloc[i,21]
    B31[x][i][4,0]=J[x].iloc[i,7]
    B31[x][i][4,2]=J[x].iloc[i,6]
    B31[x][i][4,4]=J[x].iloc[i,23]
    B31[x][i][4,6]=J[x].iloc[i,24]
    B31[x][i][4,8]=J[x].iloc[i,25]
    B31[x][i][5,1]=J[x].iloc[i,8]
    B31[x][i][5,3]=J[x].iloc[i,9]
    B31[x][i][5,5]=J[x].iloc[i,27]
    B31[x][i][5,7]=J[x].iloc[i,26]
    B31[x][i][6,0]=J[x].iloc[i,11]
    B31[x][i][6,2]=J[x].iloc[i,10]
    B31[x][i][6,4]=J[x].iloc[i,15]
    B31[x][i][6,6]=J[x].iloc[i,28]
    B31[x][i][6,8]=J[x].iloc[i,29]
    B31[x][i][7,3]=J[x].iloc[i,12]
    B31[x][i][7,5]=J[x].iloc[i,30]
    B31[x][i][8,3]=J[x].iloc[i,13]
    B31[x][i][8,4]=J[x].iloc[i,14]
    B31[x][i][8,5]=J[x].iloc[i,31]


# **Subject 32**

# In[ ]:


M = np.zeros(81)
M.resize(9,9)
M[0,3]=1
M[0,5]=17
M[1,3]=2
M[1,5]=18
M[2,0]=4
M[2,2]=3
M[2,4]=19
M[2,6]=20
M[2,8]=21
M[3,1]=5
M[3,3]=6
M[3,5]=23
M[3,7]=22
M[4,0]=8
M[4,2]=7
M[4,4]=24
M[4,6]=25
M[4,8]=26
M[5,1]=9
M[5,3]=10
M[5,5]=28
M[5,7]=27
M[6,0]=12
M[6,2]=11
M[6,4]=16
M[6,6]=29
M[6,8]=30
M[7,3]=13
M[7,5]=31
M[8,3]=14
M[8,4]=15
M[8,5]=32

A = []
for i in range(8064):
  A.append(M)
A = np.array(A)


B32 = []
for i in range(40):
  B32.append(A)
B32 = np.array(B32)


S1M1= []
S1M2 = []
S1M3 = []
S1M4 = []
S1M5 = []
S1M6 = []
S1M7 = []
S1M8 = []
S1M9 = []
S1M10 = []
S1M11 = []
S1M12 = []
S1M13 = []
S1M14 = []
S1M15 = []
S1M16 = []
S1M17 = []
S1M18 = []
S1M19 = []
S1M20 = []
S1M21 = []
S1M22 = []
S1M23 = []
S1M24 = []
S1M25 = []
S1M26 = []
S1M27 = []
S1M28 = []
S1M29 = []
S1M30 = []
S1M31 = []
S1M32 = []
S1M33 = []
S1M34 = []
S1M35 = []
S1M36 = []
S1M37 = []
S1M38 = []
S1M39 = []
S1M40 = []

J = [S1M1 ,S1M2 ,S1M3 ,S1M4 ,S1M5 ,S1M6 ,S1M7 ,S1M8 ,S1M9 ,S1M10 ,S1M11 ,S1M12 ,S1M13 ,S1M14 ,S1M15 ,S1M16 ,S1M17 ,S1M18 ,S1M19 ,S1M20 ,S1M21 ,S1M22 ,S1M23 ,S1M24 ,S1M25 ,S1M26 ,S1M27 ,S1M28 ,S1M29 ,S1M30 ,S1M31 ,S1M32 ,S1M33 ,S1M34 ,S1M35 ,S1M36 ,S1M37 ,S1M38 ,S1M39 ,S1M40]


# In[ ]:


for j in range(len(J)): 
  for i in range(8064): 
    x = pd.DataFrame(mats[31][1][j,:,i][0:32])
    J[j].append(x)
  J[j] = np.array(J[j])
  J[j].resize(8064, 32)
  J[j] = pd.DataFrame(J[j])


for x in range(40):
  for i in range(8064):   
    B32[x][i][0,3]=J[x].iloc[i,0]
    B32[x][i][0,5]=J[x].iloc[i,16]
    B32[x][i][1,3]=J[x].iloc[i,1]
    B32[x][i][1,5]=J[x].iloc[i,17]
    B32[x][i][2,0]=J[x].iloc[i,3]
    B32[x][i][2,2]=J[x].iloc[i,2]
    B32[x][i][2,4]=J[x].iloc[i,18]
    B32[x][i][2,6]=J[x].iloc[i,19]
    B32[x][i][2,8]=J[x].iloc[i,20]
    B32[x][i][3,1]=J[x].iloc[i,4]
    B32[x][i][3,3]=J[x].iloc[i,5]
    B32[x][i][3,5]=J[x].iloc[i,22]
    B32[x][i][3,7]=J[x].iloc[i,21]
    B32[x][i][4,0]=J[x].iloc[i,7]
    B32[x][i][4,2]=J[x].iloc[i,6]
    B32[x][i][4,4]=J[x].iloc[i,23]
    B32[x][i][4,6]=J[x].iloc[i,24]
    B32[x][i][4,8]=J[x].iloc[i,25]
    B32[x][i][5,1]=J[x].iloc[i,8]
    B32[x][i][5,3]=J[x].iloc[i,9]
    B32[x][i][5,5]=J[x].iloc[i,27]
    B32[x][i][5,7]=J[x].iloc[i,26]
    B32[x][i][6,0]=J[x].iloc[i,11]
    B32[x][i][6,2]=J[x].iloc[i,10]
    B32[x][i][6,4]=J[x].iloc[i,15]
    B32[x][i][6,6]=J[x].iloc[i,28]
    B32[x][i][6,8]=J[x].iloc[i,29]
    B32[x][i][7,3]=J[x].iloc[i,12]
    B32[x][i][7,5]=J[x].iloc[i,30]
    B32[x][i][8,3]=J[x].iloc[i,13]
    B32[x][i][8,4]=J[x].iloc[i,14]
    B32[x][i][8,5]=J[x].iloc[i,31]


# In[ ]:


P = [B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,B14,B15,B16,B17,B18,B19,B20,B21,B22,B23,B24,B25,B26,B27,B28,B29,B30,B31,B32]
for i in range(32):
  print(i+1,":",P[i].shape)


# In[ ]:


np.save("/content/drive/MyDrive/B1", B1)
np.save("/content/drive/MyDrive/B2", B2)
np.save("/content/drive/MyDrive/B3", B3)
np.save("/content/drive/MyDrive/B4", B4)
np.save("/content/drive/MyDrive/B5", B5)
np.save("/content/drive/MyDrive/B6", B6)
np.save("/content/drive/MyDrive/B7", B7)
np.save("/content/drive/MyDrive/B8", B8)
np.save("/content/drive/MyDrive/B9", B9)
np.save("/content/drive/MyDrive/B10", B10)
np.save("/content/drive/MyDrive/B11", B11)
np.save("/content/drive/MyDrive/B12", B12)
np.save("/content/drive/MyDrive/B13", B13)
np.save("/content/drive/MyDrive/B14", B14)
np.save("/content/drive/MyDrive/B15", B15)
np.save("/content/drive/MyDrive/B16", B16)
np.save("/content/drive/MyDrive/B17", B17)
np.save("/content/drive/MyDrive/B18", B18)
np.save("/content/drive/MyDrive/B19", B19)
np.save("/content/drive/MyDrive/B20", B20)
np.save("/content/drive/MyDrive/B21", B21)
np.save("/content/drive/MyDrive/B22", B22)
np.save("/content/drive/MyDrive/B23", B23)
np.save("/content/drive/MyDrive/B24", B24)
np.save("/content/drive/MyDrive/B25", B25)
np.save("/content/drive/MyDrive/B26", B26)
np.save("/content/drive/MyDrive/B27", B27)
np.save("/content/drive/MyDrive/B28", B28)
np.save("/content/drive/MyDrive/B29", B29)
np.save("/content/drive/MyDrive/B30", B30)
np.save("/content/drive/MyDrive/B31", B31)
np.save("/content/drive/MyDrive/B32", B32)


# In[ ]:


label1 = mats[0][0]
label2 = mats[1][0]
label3 = mats[2][0]
label4 = mats[3][0]
label5 = mats[4][0]
label6 = mats[5][0]
label7 = mats[6][0]
label8 = mats[7][0]
label9 = mats[8][0]
label10 = mats[9][0]
label11 = mats[10][0]
label12 = mats[11][0]
label13 = mats[12][0]
label14 = mats[13][0]
label15 = mats[14][0]
label16 = mats[15][0]
label17 = mats[16][0]
label18 = mats[17][0]
label19 = mats[18][0]
label20 = mats[19][0]
label21 = mats[20][0]
label22 = mats[21][0]
label23 = mats[22][0]
label24 = mats[23][0]
label25 = mats[24][0]
label26 = mats[25][0]
label27 = mats[26][0]
label28 = mats[27][0]
label29 = mats[28][0]
label30 = mats[29][0]
label31 = mats[30][0]
label32 = mats[31][0]


# In[ ]:




