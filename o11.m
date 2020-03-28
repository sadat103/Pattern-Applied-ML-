fileName = 'Train.txt';
fid = fopen(fileName);
header = fscanf(fid, '%d %d %d', [1 3]);
fclose(fid);

numOfFeatures = header(1);
numOfClasses = header(2);
numOfSamples = header(3);

trainingSet = dlmread(fileName);
trainingSet(1, :) = [];

labels = trainingSet(:, numOfFeatures+1);

trainingSet(:, numOfFeatures+1) = 1;
labels(labels==2) = -1;


rho = 0.01;
weightVector = rand(1,numOfFeatures+1);

missClassified = numOfSamples;

iter=0;
max_iter = 10000;

while (missClassified > 0) && (iter < max_iter)
    
    iter = iter + 1;
    missClassified = 0;
    
    gradient = zeros(1, numOfFeatures+1);
    
    
    for i = 1:numOfSamples
        
        if (trainingSet(i, :) * weightVector' * labels(i) < 0)
            missClassified = missClassified + 1;
            gradient = gradient - labels(i) * trainingSet(i,:);
        end
        
    end
    
    weightVector = weightVector - rho * gradient;
    
end


disp(weightVector);


testSet = dlmread('Test.txt');
[row, col] = size(testSet);

actualLabels = testSet(:, numOfFeatures + 1);
predictedLabels = zeros(row,1);
dataSet = testSet;

testSet(:, numOfFeatures+1) = 1;

wrong = 0;

for i = 1:row
    
    result = testSet(i, :) * weightVector';
    
    if (result < 0)
        predictedLabels(i) = 2;
    else
        predictedLabels(i) = 1;
    end
    
    if (predictedLabels(i) ~= actualLabels(i))
        wrong = wrong + 1;
    end
    
end


data = confusionmat(actualLabels, predictedLabels);

act = sum(data,1);
recall1 = data(1,1)/act(1);
recall2 = data(2,2)/act(2);

act = sum(data);
precision1 = data(1,1)/act(1);
precision2 = data(2,2)/act(2);

accuracy = 1 - wrong / row;

disp('actual    predicted');
disp([actualLabels, predictedLabels]);

fprintf('accuracy = %d\trecall1 = %d\tprecision1 = %d\n', accuracy*100, recall1*100, precision1*100);
fprintf('accuracy = %d\trecall2 = %d\tprecision2 = %d\n', accuracy*100, recall2*100, precision2*100);

class1 = dataSet(dataSet(:, numOfFeatures+1) == 1, :);
class1(:, numOfFeatures+1) = [];
class2 = dataSet(dataSet(:, numOfFeatures+1) == 2, :);
class2(:, numOfFeatures+1) = [];


figure;
scatter3(class1(:,1), class1(:,2), class1(:,3));

hold on;
scatter3(class2(:,1), class2(:,2), class2(:,3));

hold on;

[xx,yy,zz] = meshgrid(-10:0.1:20, -10:0.1:20, -10:0.1:20);
isosurface(xx, yy, zz, weightVector(1)*xx+weightVector(2)*yy+weightVector(3)*zz+weightVector(4), 0)