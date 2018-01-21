#!/usr/bin/python
def getPrediction(elem):
    return elem[2]

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    
    # Calculate error (networth - prediction) for each data point and push to array
    for i in range(len(predictions)):
        cleaned_data.append([int(ages[i]), float(net_worths[i]), float((net_worths[i] - predictions[i])**2)])
        
    # Sort data in descending order of error.
    cleaned_data.sort(key = getPrediction, reverse = True)
    
    # Delete first 10% elements in list (10% elements with highest error)
    n = int(len(predictions)*0.1)
    del cleaned_data[:n]
    
    return cleaned_data


