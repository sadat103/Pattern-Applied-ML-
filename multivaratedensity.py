import numpy as np
from math import exp, sqrt


file = open("Train.txt")

numOfClass, numOfFeatures, datasetLength = 0,0,0

lines = file.readlines()

dataset = []

count = 0

for line in lines:
    if(count == 0):
        var = line.split()
        numOfFeatures = int(var[0])
        numOfClass = int(var[1])
        datasetLength = int(var[2])
    else:
        var = line.split()
        size = len(var)
        data = []
        index = 0
        for i in var:
            if(index == size - 1):
                data.append(int(i))
            else:
                data.append(float(i))
            index = index + 1
        dataset.append(data)
    count = count + 1
    
print(numOfClass, numOfFeatures, datasetLength)
print(dataset)

file1 = open("Test.txt")

test_dataset = []
test_dataset1 = []

line1 = file1.readlines()

for line in line1:
    var = line.split()
    size = len(var)
    data = []
    idx = 0
    for i in var:
        if(idx == size - 1):
            data.append(int(i))
        else:
            data.append(float(i))
        idx = idx + 1
    test_dataset.append(data)
    test_dataset1.append(data)
    
print("test dataset")
print(test_dataset)

class_wise_dataset = []

classes = set()

for data in dataset:
    classes.add(data[numOfFeatures])
    
print(classes)

prior_probability = []
#printing probability
for c in classes:
    d = []
    iit = 0
    for data in dataset:
        if data[numOfFeatures] == c:
            d.append(data)
            iit = iit + 1
    class_wise_dataset.append(d)
    prior_probability.append(float(iit/datasetLength))
    
print("prior probability")
print(prior_probability)

class_wise_avg = []
#printing average 
for c in classes:
    avg = []
    for i in range(numOfFeatures):
        sum = 0
        for data in class_wise_dataset[c-1]:
            sum = sum + data[i]
        average = sum/len(class_wise_dataset[c-1])
        avg.append(average)
    class_wise_avg.append(avg)

    
#co var matrix
class_wise_co_var = []

for i in range(numOfFeatures):
    row = []
    for j in range(numOfFeatures):
        sum = 0
        for data in dataset:
            sum = sum + (data[i] - avg[i]) * (data[j] - avg[j])
        row.append(float(sum/datasetLength))
    class_wise_co_var.append(row)



for c in classes:
    print(class_wise_co_var[c-1])

co_variance_matrix = np.matrix(class_wise_co_var)

print(co_variance_matrix)

inv_co_variance_matrix = np.linalg.inv(co_variance_matrix)
print("Co variance matrix")   
print(inv_co_variance_matrix)

det = np.linalg.det(co_variance_matrix)
print("Co variance matrx determination")
print(det)

feature_vectors = []

for data in test_dataset:
    s = data.pop()
    feature_vector = np.matrix(data)
    feature_vector = np.transpose(feature_vector)
    feature_vectors.append(feature_vector)
    data.append(s)

#print(feature_vectors)
print("test dataset")
print(test_dataset)



output = []

def mul_var(num,dt,x,y):
    cc = float(1/((2*3.1416)**float(num/2))*sqrt(dt))
    m = cc*exp(-0.5*x.transpose()*(y*x))
    return m

for test_data in feature_vectors:
    posterior = []
    for c in range(numOfClass):
        #mul = 1.0
        x = np.matrix(test_data - np.matrix(class_wise_avg[c]).transpose())
        mul = mul_var(numOfFeatures,det,x,inv_co_variance_matrix)
        mul = mul*prior_probability[c]
        posterior.append(mul)
    output.append(posterior)
print("Posterior output")
print(posterior)
print(output)

accuracy = 0
print(test_dataset)
count = 0
print(len(test_dataset))
print("sample no , feature values , actual class , estimated class")
for out in output:
    if test_dataset[count][numOfFeatures] == out.index(max(out))+1:
        accuracy = accuracy + 1
        
    else:
        print(count+1 , test_dataset[count], out.index(max(out))+1)
    count = count + 1
print(count)
print(accuracy)
print("Accuracy :" , float((accuracy/len(test_dataset)))*100)
