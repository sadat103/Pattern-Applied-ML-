fileName = 'trainLinearlySeparable.txt';
fid = fopen(fileName);
input = fscanf(fid, '%d %d %d', [1 3]);
fclose(fid);

Features = input(1);
Classes = input(2);
Samples = input(3);

trainingSet = dlmread(fileName);
trainingSet(1, :) = [];

labels = trainingSet(:, Features+1);

trainingSet(:, Features+1) = 1;
labels(labels==2) = -1;

%disp(trainingSet);

rho = 0.01;
weightVector = rand(1,Features+1);

missClassified = Samples;

iter=0;
max_iter = 10000;

while (missClassified > 0) && (iter < max_iter)
    
    missClassified = 0;
    
    for i = 1:Samples
        iter = iter + 1;
        
        if (trainingSet(i, :) * weightVector' * labels(i) < 0)
            missClassified = missClassified + 1;
            weightVector = weightVector + rho * labels(i) * trainingSet(i,:);
        end
        
    end
    
end

disp('Weight Vector');
disp(weightVector);


testSet = dlmread('testLinearlySeparable.txt');
[row, col] = size(testSet);

actualclass = testSet(:, Features + 1);
predictedclass = zeros(row,1);
dataSet = testSet;

testSet(:, Features+1) = 1;

wrong = 0;

for i = 1:row
    
    result = testSet(i, :) * weightVector';
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

disp('actual    predicted  resultedweight');
disp([actualclass, predictedclass , resultlabels ]);
fprintf('Total samples = %d\n',row);
fprintf('correct = %d\n',cor);
fprintf('wrong = %d\n',wrong);

fprintf('accuracy = %d\n', accuracy*100);

