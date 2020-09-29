fileName = 'trainLinearlyNonSeparable.txt';
fid = fopen(fileName);
input = fscanf(fid, '%d %d %d', [1 3]);
fclose(fid);

Features = input(1);
Class = input(2);
Samples = input(3);

trainingSet = dlmread(fileName);
trainingSet(1, :) = [];

labels = trainingSet(:, Features+1);

trainingSet(:, Features+1) = 1;
labels(labels==2) = -1;


ro = 0.01;
weightVector = rand(1,Features+1);
storedWeightVector = weightVector;


missClassified = Samples;
score = missClassified;

iter=0;
max_iter = 10000;

while (missClassified > 0) && (iter < max_iter)
    
    iter = iter + 1;
    missClassified = 0;
    
    gradient = zeros(1, Features+1);
    
    
    for i = 1:Samples
        
        if (trainingSet(i, :) * weightVector' * labels(i) < 0)
            missClassified = missClassified + 1;
            gradient = gradient - labels(i) * trainingSet(i,:);
        end
        
    end
    
    weightVector = weightVector - ro * gradient;
    
    if (score > missClassified)
        score = missClassified;
        storedWeightVector = weightVector;
        %disp(storedWeightVector);
    end
    
end

disp('Stored Weight Vector');
disp(storedWeightVector);


testSet = dlmread('testLinearlyNonSeparable.txt');
[row, col] = size(testSet);

actualclass = testSet(:, Features + 1);
predictedclass = zeros(row,1);
dataSet = testSet;

testSet(:, Features+1) = 1;

wrong = 0;

for i = 1:row
    
    result = testSet(i, :) * storedWeightVector';
    resultlabels(i) = result ;
    if (result < 0)
        predictedclass(i) = 2;
    else
        predictedclass(i) = 1;
    end
    
    if (predictedclass(i) ~= actualclass(i))
        wrong = wrong + 1;
    end
    
end
cor = row - wrong ;
accuracy = cor / row;

disp('actual     predicted   resultedweight');
disp([actualclass, predictedclass , resultlabels]);
fprintf('Total samples = %d\n',row);
fprintf('correct = %d\n',cor);
fprintf('wrong = %d\n',wrong);
fprintf('accuracy = %d\n', accuracy*100);
