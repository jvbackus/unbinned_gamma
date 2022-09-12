#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import random
import ast
import time
import random
import statistics as st


# In[2]:


# Put all MC samples together

validPointsMinusTogether = []
validPointsPlusTogether = []
validPointsDTogether = []

for i in range(100):
    label = "high_rB/points-" + str(i + 1) + ".m"
    with open (label, "r") as f:
        text = f.read()
        validPointsMinus = text[text.find('=') + 2:text.find('}}') + 2]
        text = text.replace("= " + validPointsMinus + ";", "")
        validPointsPlus = text[text.find('=') + 2:text.find('}}') + 2]
        text = text.replace("= " + validPointsPlus + ";", "")    
        validPointsD = text[text.find('=') + 2:text.find('}}') + 2]
        for rep in (('{','['),('}',']')):
            validPointsMinus = validPointsMinus.replace(rep[0],rep[1])
            validPointsPlus = validPointsPlus.replace(rep[0], rep[1])
            validPointsD = validPointsD.replace(rep[0], rep[1])
        validPointsMinusMaster = ast.literal_eval(validPointsMinus)
        validPointsPlusMaster = ast.literal_eval(validPointsPlus)
        validPointsDMaster = ast.literal_eval(validPointsD)
        
    validPointsMinusTogether = validPointsMinusTogether + validPointsMinusMaster
    validPointsPlusTogether = validPointsPlusTogether + validPointsPlusMaster
    validPointsDTogether = validPointsDTogether + validPointsDMaster

print(len(validPointsMinusTogether))
print(len(validPointsPlusTogether))
print(len(validPointsDTogether))


# In[3]:


start = time.time()

# Insert scenario(s) here.

rBTrue = 0.9
deltaBDegTrue = 60
gammaDegTrue = 150

with open ("high_rB/points-1.m", "r") as f:
    text = f.read()
    validPointsMinus = text[text.find('=') + 2:text.find('}}') + 2]
    text = text.replace("= " + validPointsMinus + ";", "")
    validPointsPlus = text[text.find('=') + 2:text.find('}}') + 2]
    text = text.replace("= " + validPointsPlus + ";", "")    
    validPointsD = text[text.find('=') + 2:text.find('}}') + 2]
    for rep in (('{','['),('}',']')):
        validPointsMinus = validPointsMinus.replace(rep[0],rep[1])
        validPointsPlus = validPointsPlus.replace(rep[0], rep[1])
        validPointsD = validPointsD.replace(rep[0], rep[1])
    validPointsMinusMaster = ast.literal_eval(validPointsMinus)
    validPointsPlusMaster = ast.literal_eval(validPointsPlus)
    validPointsDMaster = ast.literal_eval(validPointsD)

spacing = 0.01
countTot = 1000
typ = 'tilde' # can be either 'none', 'tilde', or 'bar'

sList = []
s = 0.35
while s <= 3.0:
    sList.append([s])
    s = s + spacing
sLength = len(sList)
sList = np.array(sList)

sizePlus = 6466
sizeMinus  = 6313
sizeD = 5360

validPointsMinusMaster = random.sample(validPointsMinusMaster, sizeMinus)
validPointsPlusMaster = random.sample(validPointsPlusMaster, sizePlus)
validPointsDMaster  = random.sample(validPointsDMaster, sizeD)

validPointsMinus = np.array(validPointsMinusMaster)
validPointsPlus = np.array(validPointsPlusMaster)
validPointsD = np.array(validPointsDMaster)


# In[4]:


NPlusTot = len(validPointsPlus)
NMinusTot = len(validPointsMinus)
NDTot = len(validPointsD)

print("Number of B^+ decays: " + str(NPlusTot))
print("Number of B^- decays: " + str(NMinusTot))
print("Number of D decays: " + str(NDTot))


# In[5]:


# Computes N(s12, s13) or N(s13, s12) 
#(with a tilde too) for either B^+, B^-, or D.
# Possibilities for var: 'none', 'tilde', 'bar'.
def count_events(events, flip=False, var='none'):
    N = []
    eventsXLong = np.tile(events[:,0],(sLength,1,))
    eventsYLong = np.tile(events[:,1],(sLength,1,))
    
    if (var == 'none'):
        eventsX = eventsXLong < sList
        eventsY = eventsYLong < sList
        
        if not flip:
            for i in range(sLength):
                for j in range(sLength):
                    count = np.count_nonzero(np.logical_and(eventsX[i], eventsY[j]))
                    N.append(count)
        else:
            for i in range(sLength):
                for j in range(sLength):
                    count = np.count_nonzero(np.logical_and(eventsX[j], eventsY[i]))
                    N.append(count)
    
    elif (var == 'tilde'):
        eventsX = eventsXLong > sList
        eventsY = eventsYLong > sList   
        
        if not flip:
            for i in range(sLength):
                for j in range(sLength):
                    count = np.count_nonzero(np.logical_and(eventsX[i], eventsY[j]))
                    N.append(count)
        else:
            for i in range(sLength):
                for j in range(sLength):
                    count = np.count_nonzero(np.logical_and(eventsX[j], eventsY[i]))
                    N.append(count)
                    
    elif (var == 'bar'):
        eventsX = eventsXLong > sList
        eventsY = eventsYLong < sList   
        
        if not flip:
            for i in range(sLength):
                for j in range(sLength):
                    count = np.count_nonzero(np.logical_and(eventsX[i], eventsY[j]))
                    N.append(count)
        else:
            for i in range(sLength):
                for j in range(sLength):
                    count = np.count_nonzero(np.logical_and(eventsX[j], eventsY[i]))
                    N.append(count)
    
    return N


# Computes R_sigmaS, R_sigmaA, R_deltaS, R_deltaA for each [s12, s13]
# (tilde included also).
# Possibilities for var: 'none', 'tilde', 'bar'.
def compute_R(validPointsPlusInt, validPointsMinusInt, var='none'):
    
    NPlusNoFlip = np.array(count_events(validPointsPlusInt, var=var))
    NPlusFlip = np.array(count_events(validPointsPlusInt, flip=True, var=var))
    NMinusNoFlip = np.array(count_events(validPointsMinusInt, var=var))
    NMinusFlip = np.array(count_events(validPointsMinusInt, flip=True, var=var))
    
    sigmaPlus = 1/2 * (NPlusNoFlip + NPlusFlip)
    sigmaMinus  = 1/2 * (NMinusNoFlip + NMinusFlip)
    deltaPlus = 1/2 * (NPlusNoFlip - NPlusFlip)
    deltaMinus = 1/2 * (NMinusNoFlip - NMinusFlip)
    
    RSigmaS = 1/2 * (sigmaPlus + sigmaMinus)
    RSigmaA = 1/2 * (sigmaPlus - sigmaMinus)  
    RDeltaS = 1/2 * (deltaPlus + deltaMinus)
    RDeltaA = 1/2 * (deltaPlus - deltaMinus)
        
    return [RSigmaS, RSigmaA, RDeltaS, RDeltaA, NPlusNoFlip, NPlusFlip, NMinusNoFlip, NMinusFlip]


# In[6]:


Rs = compute_R(validPointsPlus, validPointsMinus, var=typ)

NDNoFlip = np.array(count_events(validPointsD, var=typ))
NDFlip = np.array(count_events(validPointsD, var=typ, flip=True))

RDeltaS = Rs[2]
RSigmaA = Rs[1]

NPlusNoFlip = Rs[4]
NPlusFlip = Rs[5]
NMinusNoFlip = Rs[6]
NMinusFlip = Rs[7]


# In[7]:


minGammas = []
minDeltaBs = []
minDs = []
minRbs = []
count = 1

while count <= countTot:
    
    if (count > 1 and count <= 100):    
        newName = 'high_rB/points-' + str(count) + '.m'
        with open (newName, "r") as f:
            text = f.read()
            validPointsMinus = text[text.find('=') + 2:text.find('}}') + 2]
            text = text.replace("= " + validPointsMinus + ";", "")
            validPointsPlus = text[text.find('=') + 2:text.find('}}') + 2]
            text = text.replace("= " + validPointsPlus + ";", "")    
            validPointsD = text[text.find('=') + 2:text.find('}}') + 2]
            for rep in (('{','['),('}',']')):
                validPointsMinus = validPointsMinus.replace(rep[0],rep[1])
                validPointsPlus = validPointsPlus.replace(rep[0], rep[1])
                validPointsD = validPointsD.replace(rep[0], rep[1])
            validPointsMinusMaster = ast.literal_eval(validPointsMinus)
            validPointsPlusMaster = ast.literal_eval(validPointsPlus)
            validPointsDMaster = ast.literal_eval(validPointsD)
        
        validPointsMinusMaster = random.sample(validPointsMinusMaster, sizeMinus)
        validPointsPlusMaster = random.sample(validPointsPlusMaster, sizePlus)
        validPointsDMaster = random.sample(validPointsDMaster, sizeD)

        validPointsMinus = np.array(validPointsMinusMaster)
        validPointsPlus = np.array(validPointsPlusMaster)
        validPointsD = np.array(validPointsDMaster)
        
        NPlusTot = len(validPointsPlus)
        NMinusTot = len(validPointsMinus)
        NDTot = len(validPointsD)
        
        Rs = compute_R(validPointsPlus, validPointsMinus, var=typ)

        NDNoFlip = np.array(count_events(validPointsD, var=typ))
        NDFlip = np.array(count_events(validPointsD, var=typ, flip=True))

        RDeltaS = Rs[2]
        RSigmaA = Rs[1]

        NPlusNoFlip = Rs[4]
        NPlusFlip = Rs[5]
        NMinusNoFlip = Rs[6]
        NMinusFlip = Rs[7]
    
    elif (count > 100):
        
        # random sample from all MCs
        validPointsMinusMaster = random.sample(validPointsMinusTogether, sizeMinus)
        validPointsPlusMaster = random.sample(validPointsPlusTogether, sizePlus)
        validPointsDMaster = random.sample(validPointsDTogether, sizeD)
        
        validPointsMinus = np.array(validPointsMinusMaster)
        validPointsPlus = np.array(validPointsPlusMaster)
        validPointsD = np.array(validPointsDMaster)
        
        NPlusTot = len(validPointsPlus)
        NMinusTot = len(validPointsMinus)
        NDTot = len(validPointsD)
        
        Rs = compute_R(validPointsPlus, validPointsMinus, var=typ)

        NDNoFlip = np.array(count_events(validPointsD, var=typ))
        NDFlip = np.array(count_events(validPointsD, var=typ, flip=True))

        RDeltaS = Rs[2]
        RSigmaA = Rs[1]

        NPlusNoFlip = Rs[4]
        NPlusFlip = Rs[5]
        NMinusNoFlip = Rs[6]
        NMinusFlip = Rs[7]
    
    # Lattice spacings
    deltaBDegSpacing = 2
    gammaDegSpacing = 2
    rBSpacing = 0.05
    
    minGammasTemp = []
    minDeltaBsTemp = []
    minDsTemp = []

    rB = 0.50
    while rB <= 1.30:
        minGammasTempTemp = []
        minDsTempTemp = []
        
        deltaBDeg = 1
        while deltaBDeg < 90:

            deltaBRad = math.radians(deltaBDeg)

            RDeltaSWGamma = []
            RSigmaAWGamma = []

            gammaDeg = 91
            while gammaDeg < 180:

                gammaRad = math.radians(gammaDeg)
                RDeltaSWGamma.append(RDeltaS)
                a = -(math.cos(deltaBRad) / math.sin(deltaBRad)) * (math.cos(gammaRad) / math.sin(gammaRad))
                RSigmaAWGamma.append(a * RSigmaA)

                gammaDeg = gammaDeg + gammaDegSpacing

            RDeltaSWGamma = np.array(RDeltaSWGamma)
            RSigmaAWGamma = np.array(RSigmaAWGamma)

            abssq1213integral = 452.49440811079535
            abssq1312integral = 452.4832645076384
            angleintegral1 = -11.35064189321533
            angleintegral2 = -0.0002800600245513131

            denomPlus = []
            denomMinus = []

            gammaDeg = 91
            while gammaDeg < 180:
                gammaRad = math.radians(gammaDeg)

                denomMinusTemp = abssq1213integral + (rB**2)*abssq1312integral + 2*rB*((math.cos(deltaBRad)*math.cos(gammaRad) + math.sin(deltaBRad)*math.sin(gammaRad))*angleintegral1 + (math.cos(gammaRad)*math.sin(deltaBRad) - math.cos(deltaBRad)*math.sin(gammaRad))*angleintegral2)
                denomPlusTemp = abssq1312integral + (rB**2)*abssq1213integral + 2*rB*((math.cos(deltaBRad)*math.cos(gammaRad) - math.sin(deltaBRad)*math.sin(gammaRad))*angleintegral1 - (math.cos(gammaRad)*math.sin(deltaBRad) + math.cos(deltaBRad)*math.sin(gammaRad))*angleintegral2)

                denomMinus.append(denomMinusTemp)
                denomPlus.append(denomPlusTemp)

                gammaDeg = gammaDeg + gammaDegSpacing

            denomMinus = np.array(denomMinus)
            denomPlus = np.array(denomPlus)

            rPlus = abssq1213integral / denomPlus
            rMinus = abssq1213integral / denomMinus

            RSigmaSSubWGamma = []
            RDeltaASubWGamma = []
            
            i = 0
            while i < 45:

                NPlusSubNoFlip = NPlusNoFlip - rPlus[i]*(NPlusTot / NDTot)*(NDFlip + (rB**2)*NDNoFlip)
                NMinusSubNoFlip = NMinusNoFlip - rMinus[i]*(NMinusTot / NDTot)*(NDNoFlip + (rB**2)*NDFlip)
                NPlusSubFlip = NPlusFlip - rPlus[i]*(NPlusTot / NDTot)*(NDNoFlip + (rB**2)*NDFlip)
                NMinusSubFlip = NMinusFlip - rMinus[i]*(NMinusTot / NDTot)*(NDFlip + (rB**2)*NDNoFlip)

                SigmaPlusSub = 1/2 * (NPlusSubNoFlip + NPlusSubFlip)
                SigmaMinusSub = 1/2 * (NMinusSubNoFlip + NMinusSubFlip)
                DeltaPlusSub = 1/2 * (NPlusSubNoFlip - NPlusSubFlip)
                DeltaMinusSub = 1/2 * (NMinusSubNoFlip - NMinusSubFlip)

                RSigmaSSub = 1/2 * (SigmaPlusSub + SigmaMinusSub)
                RDeltaASub = 1/2 * (DeltaPlusSub - DeltaMinusSub)

                RSigmaSSubWGamma.append(RSigmaSSub)

                gammaRad = math.radians(int(2*i + 91))
                b = (math.cos(deltaBRad) / math.sin(deltaBRad)) * (math.sin(gammaRad) / math.cos(gammaRad))

                RDeltaASubWGamma.append(b * RDeltaASub)

                i = i + 1

            RSigmaSSubWGamma = np.array(RSigmaSSubWGamma)
            RDeltaASubWGamma = np.array(RDeltaASubWGamma)

            DSigma = np.absolute(RSigmaSSubWGamma - RSigmaAWGamma)
            DDelta = np.absolute(RDeltaSWGamma - RDeltaASubWGamma)

            DSigmaMax = []
            DDeltaMax = []

            i = 0
            while i < 45:
                DSigmaMax.append(max(DSigma[i]))
                DDeltaMax.append(max(DDelta[i]))

                i = i + 1

            minSigmaPred = int(2*DSigmaMax.index(min(DSigmaMax)) + 91)
            minDeltaPred = int(2*DDeltaMax.index(min(DDeltaMax)) + 91)

            DSigmaMax = np.array(DSigmaMax)
            DDeltaMax = np.array(DDeltaMax)

            D = DSigmaMax + DDeltaMax
            D = D.tolist()

            DSigmaMax = DSigmaMax.tolist()
            DDeltaMax = DDeltaMax.tolist()

            minDPred = int(2*D.index(min(D)) + 91)
            minimumD = min(D)
            
            minGammasTempTemp.append(minDPred)
            minDsTempTemp.append(minimumD)
            
            deltaBDeg = deltaBDeg + deltaBDegSpacing
            
        minimumDFixedRB = min(minDsTempTemp)                
        indexOfMinDFixedRB = minDsTempTemp.index(min(minDsTempTemp))
        
        minDsTemp.append(minimumDFixedRB)
        minGammasTemp.append(minGammasTempTemp[indexOfMinDFixedRB])
        minDeltaBsTemp.append(int(2*indexOfMinDFixedRB + 1))
        
        rB = round(rB + rBSpacing, 2)
    
    minimumDEverythingVaried = min(minDsTemp)
    indexOfMinDEverythingVaried = minDsTemp.index(min(minDsTemp))
    
    minDs.append(minimumDEverythingVaried)
    minGammas.append(minGammasTemp[indexOfMinDEverythingVaried])
    minDeltaBs.append(minDeltaBsTemp[indexOfMinDEverythingVaried])
    minRbs.append(round(indexOfMinDEverythingVaried*0.05 + 0.50, 2))
        
    print("D Minimum Value: " + str(minDs[-1]))
    print("Gamma: " + str(minGammas[-1]))
    print("DeltaB: " + str(minDeltaBs[-1]))
    print("rB: " + str(minRbs[-1]))
    
    print('Sample number ' + str(count))
    print('List of minimum test statistics thus far: ' + str(minDs))
    print('List of gamma predictions thus far: ' + str(minGammas))
    print('List of deltaB predictions thus far: ' + str(minDeltaBs))
    print('List of rB predictions thus far: ' + str(minRbs))
            
    end1 = time.time()
    timeElapsedMins1 = (end1 - start) / 60
    print(timeElapsedMins1)
    
    count = count + 1
        
    print('-------------------------------------------------')    

print('-------------------------------------------------')
end = time.time()
timeElapsedMins = (end - start) / 60
print('Time elapsed: ' + str(timeElapsedMins) + ' mins')


# In[8]:


# Error computations for 1000 samples.

minGammas.sort()
minDeltaBs.sort()
minRbs.sort()

print(minGammas)

centerGamma = st.mean(minGammas)
leftErrorGamma = centerGamma - minGammas[160]
rightErrorGamma = minGammas[-161] - centerGamma

print(centerGamma)
print(leftErrorGamma)
print(rightErrorGamma)

print('-------------------------------------------------')

print(minDeltaBs)

centerDeltaB = st.mean(minDeltaBs)
leftErrorDeltaB = centerDeltaB - minDeltaBs[160]
rightErrorDeltaB = minDeltaBs[-161] - centerDeltaB

print(centerDeltaB)
print(leftErrorDeltaB)
print(rightErrorDeltaB)

print('-------------------------------------------------')

print(minRbs)

centerRb = st.mean(minRbs)
leftErrorRb = centerRb - minRbs[160]
rightErrorRb = minRbs[-161] - centerRb

print(centerRb)
print(leftErrorRb)
print(rightErrorRb)


# In[ ]:




