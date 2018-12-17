""""
Helper functions for analysis purposes
""""


from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(class_pred, class_true, figsize0, figsize1):
    """
    Helper function to plot confusion matrix
      
    """
     
    cm = confusion_matrix(y_true=class_true,
                          y_pred=class_pred)
    
    fig = plt.figure(figsize=(figsize0,figsize1))
    
    plt.matshow(cm, cmap=plt.cm.plasma)

    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()
    
    