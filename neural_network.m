trainingSet = dlmread('trainNN1.txt');

[Samples, Features] = size(trainingSet);

labels = trainingSet(:, Features);

Classes = length(unique(labels));

temp = zeros(Samples,Classes);

for l=1:Samples
    
    line = temp(l, :);
    
    line(labels(l)) = 1;
    
    temp(l, :) = line;
    
    
end

labels = temp;

trainingSet(:, Features) = [];
Features = Features - 1;


inputLayer = Features;
display(inputLayer);
if (Classes==2)
    outputLayer = 1;
else
    outputLayer = Classes;
end


hiddenLayer = dlmread('layer_configuration.txt');
disp("nodes per layer---------output layer");
totLayer = [hiddenLayer outputLayer];
disp(totLayer)
inputMatrix = cell(length(totLayer),1);
outputMatrix = cell(length(totLayer),1);
weightMatrix = cell(sum(totLayer),1);

alpha = 0.3;
mu = 0.5;
thres = 0.5;
maxIter = 200;


layer = inputLayer;

w = 1;

for r=1:length(totLayer)
    newLayer = totLayer(r);
    
    for j=1:newLayer
        weightMatrix{w} = rand(1,layer+1);
        w = w + 1;
    end
    
    layer = newLayer;
    
end

missClassified = Samples;

iter = 0;
while ((missClassified > 0) || (iter <= maxIter))
    iter = iter + 1; 
    
    missClassified = 0;
    
    if (iter > maxIter)
        break;
    end
    
    for n=1:Samples
        
        ouOfLayer = trainingSet(n, :);
        layer = inputLayer;
        
        ouOfLayer(layer+1) = 1;
        outputMatrix{1} = ouOfLayer;
        
        w = 1;
        %forward propagation
        for r=1:length(totLayer)
            newLayer = totLayer(r);
            ouOfNewLayer = ones(1,newLayer) * -1;
            inputOfNewLayer = ones(1,newLayer) * -1;
            
            for j=1:newLayer
                %for calculating v and y 
                weightVector = weightMatrix{w};
                w = w + 1;
                %disp(weightVector);
                inputOfNeuron = ouOfLayer * weightVector';
                inputOfNewLayer(j) = inputOfNeuron;
                ouOfNewLayer(j) = 1 / (1 + exp( - alpha * inputOfNeuron));
                
            end
            inputMatrix{r} = inputOfNewLayer;
            ouOfLayer = ouOfNewLayer;
            layer = newLayer;
            ouOfLayer(layer+1) = 1;
            outputMatrix{r+1} = ouOfLayer;
            
        end
       
        
        ouOfNeuron = outputMatrix{length(outputMatrix)};
        ouOfNeuron(length(ouOfNeuron)) = [];
        
        for o=1:length(ouOfNeuron)
            if (ouOfNeuron(o) > thres)
                ouOfNeuron(o) = 1;
            else
                ouOfNeuron(o) = 0;
            end
        end
        
        
        if (~isequal(labels(n, :), ouOfNeuron))
            missClassified = missClassified + 1;
        end
        
        deltaMatrix = cell(length(totLayer),1);
        
        lastLayer = inputMatrix{length(inputMatrix)};
        deltaOfLastLayer = ones(1,1);
        
        for j=1:length(lastLayer)
            %for calculating delta
            inputOfNeuron = lastLayer(j);
            activationFunction = 1 / (1 + exp( - alpha * inputOfNeuron));
            derivative = alpha * activationFunction * (1 - activationFunction);
            
            if (activationFunction > thres)
                activationFunction = 1;
            else
                activationFunction = 0;
            end
            
            individualError = activationFunction - labels(n, j);
            deltaOfLastLayer(j) = individualError * derivative;
            
        end
        
        deltaMatrix{1} = deltaOfLastLayer;
        index = 2;
        
        w = length(weightMatrix);
        deltaOfLayer = deltaOfLastLayer;
        %backpropagation
        for r=length(inputMatrix)-1:-1:1
            
            layer = inputMatrix{r};
            deltaOfPreLayer = ones(1,1);
            
            w = w - length(deltaOfLayer);
            
            for j=1:length(layer)
                
                inputOfNeuron = layer(j);
                activationFunction = 1 / (1 + exp( - alpha * inputOfNeuron));
                derivative = alpha * activationFunction * (1 - activationFunction);
                
                b = w + 1;
                
                for k=1:length(deltaOfLayer)
                    
                    delta = deltaOfLayer(k);
                    weightVector = weightMatrix{b};
                    b = b + 1;
                    individualError = individualError + delta * weightVector(j);
                    
                end
                
                deltaOfPreLayer(j) = individualError * derivative;
                
            end
            
            
            deltaMatrix{index} = deltaOfPreLayer;
            index = index + 1;
            deltaOfLayer = deltaOfPreLayer;
            
        end
        
        deltaMatrix = flipud(deltaMatrix);
        gradient = cell(length(totLayer),1);
         %updating weight
        
        index = 1;
        
        for r=1:length(deltaMatrix)
            
            delta = deltaMatrix{r};
            output = outputMatrix{r};
            
            for j=1:length(delta)
                
                gradient{index} = delta(j) * outputMatrix{r};
                index = index + 1;
                
            end
            
        end
        

        
        newWeightMatrix = cell(length(totLayer),1);
        
        for r=1:length(weightMatrix)
            
            newWeightMatrix{r} = weightMatrix{r} - mu * gradient{r};
            
        end
        
    
        weightMatrix = newWeightMatrix;
        
    end
    
  
    
end


testSet = dlmread('testNN1.txt');
[row, col] = size(testSet);

actualLabels = testSet(:, Features + 1);

temp = zeros(row,Classes);

for l=1:row
    
    line = temp(l, :);
    
    line(actualLabels(l)) = 1;
    
    temp(l, :) = line;
    
    
end

actualLabels = temp;
predictedLabels = actualLabels;

testSet(:, Features+1) = [];

for t=1:col-1
    testSet(:, t) = testSet(:, t) / max(testSet(:, t));
end

wrong = 0;

for n=1:row
    
    ouOfLayer = testSet(n, :);
    layer = inputLayer;
    
    ouOfLayer(layer+1) = 1;
    outputMatrix{1} = ouOfLayer;
    
    w = 1;
    
    for r=1:length(totLayer)
        newLayer = totLayer(r);
        ouOfNewLayer = ones(1,newLayer) * -1;
        inputOfNewLayer = ones(1,newLayer) * -1;
        
        for j=1:newLayer
            
            weightVector = weightMatrix{w};
            w = w + 1;
            inputOfNeuron = ouOfLayer * weightVector';
            inputOfNewLayer(j) = inputOfNeuron;
            ouOfNewLayer(j) = 1 / (1 + exp( - alpha * inputOfNeuron));
            
        end
        
        inputMatrix{r} = inputOfNewLayer;
        ouOfLayer = ouOfNewLayer;
        layer = newLayer;
        ouOfLayer(layer+1) = 1;
        outputMatrix{r+1} = ouOfLayer;
        
    end
    
    
    ouOfNeuron = outputMatrix{length(outputMatrix)};
    ouOfNeuron(length(ouOfNeuron)) = [];
    
    for o=1:length(ouOfNeuron)
        if (ouOfNeuron(o) > thres)
            ouOfNeuron(o) = 1;
        else
            ouOfNeuron(o) = 0;
        end
    end
    
    predictedLabels(n, :) = ouOfNeuron;
    
    if (~isequal(actualLabels(n, :), predictedLabels(n, :)))
        wrong = wrong + 1;
    end
    
end

result = zeros(length(actualLabels), 2);

for p=1:length(actualLabels)
    actual = actualLabels(p,:);
    predicted = predictedLabels(p,:);
    
    result(p, 1) = find(actual==1);
    value = find(predicted==1);
    if (isempty(value))
        value = 1;
    end
    result(p, 2) = value(1);
    
end



disp('actual-----predicted');
disp(result);
cor = length(result) - wrong;
accuracy = cor/length(result);
fprintf('Total Datapoints = %d\n',length(result));
fprintf('correct = %d\n',cor);
fprintf('missclassified = %d, accuracy = %d\n', wrong, accuracy*100);
