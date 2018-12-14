def getFrequency_and_augment(X, Y, num_classes, dsrd_count):
    """
    Function to generate new samples (augmentation)
    Still under developement
    """
    listCount = [0]*num_classes
    
    #first combine the matrix X and Y 
    Y = Y.reshape((len(Y), 1))
    XX = np.append(X, Y,1)

    
    for i in range(len(XX)):
        listCount[XX[i, -1]] +=1
        
    
    newSample =[]
    for i in range (len(XX)):
        
        freq = listCount[XX[i, -1]]

        nn = dsrd_count//freq
        
        if nn>0:
            for j in range(nn):
                newSample.append(XX[i, :])
        else:
            newSample.append(XX[i, :])
           
    return np.array(newSample)
