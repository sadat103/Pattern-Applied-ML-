loc = dlmread('bisecting.txt');
[m, col] = size(loc);
dist = pdist2(loc,loc);
sortedDist = sort(dist,2);

minPoints = 4;

% knn part
sortedDist = sort(sortedDist(:,minPoints+1));
%disp(sortedDist)
curve = sortedDist';

nPoints = length(curve);
allCoord = [1:nPoints;curve]';

firstPoint = allCoord(1,:);
lineVec = allCoord(end,:) - firstPoint;
lineVecN = lineVec / sqrt(sum(lineVec.^2));
vecFromFirst = bsxfun(@minus, allCoord, firstPoint);

scalarProduct = dot(vecFromFirst, repmat(lineVecN,nPoints,1), 2);
vecFromFirstParallel = scalarProduct * lineVecN;
vecToLine = vecFromFirst - vecFromFirstParallel;
distToLine = sqrt(sum(vecToLine.^2,2));

[maxDist,idxOfBestPoint] = max(distToLine);

figure, plot(curve)
hold on
plot(allCoord(idxOfBestPoint,1), allCoord(idxOfBestPoint,2), 'or')


eps = allCoord(idxOfBestPoint,2);
disp("minpoints and eps is")
disp([minPoints eps]);

clust = 0;

%dbscan part
clusters = zeros(m,1);
isEdge = false(m,1);
noisePt = false(m,1);

for i=1:m
    
    if ~isEdge(i)
        isEdge(i) = true;
        neigh = find(dist(i,:) <= eps);
        
        if length(neigh) < minPoints
            noisePt(i) = true;
        else
            clust = clust + 1;
            clusters(i) = clust;
            j = 1;
            while j <= length(neigh)
                t = neigh(j);
                j = j + 1;
                
                if ~isEdge(t)
                    isEdge(t) = true;
                    
                    neigh2 = find(dist(t,:) <= eps);
                    
                    if length(neigh2) >= minPoints
                        n = [neigh neigh2];
                        neigh = n;
                    end
                end
                
                if clusters(t) == 0
                    clusters(t) = clust;
                end
                
            end
        end
    end
end


CM = jet(max(clusters));

noise = find(noisePt == true);
n = length(noise);

figure;
hold on;
for i=1:n
    plot(loc(noise(i),1),loc(noise(i),2),'color','RED','marker','o');
end

for i=1:m
    n = clusters(i);
    if (n == 0)
        continue;
    end
    plot(loc(i,1),loc(i,2),'color',CM(n,:),'marker','o');
end

%kmeans part

clust = max(clusters);
disp("Cluster count")
disp(clust);
maxIter = 200;

assign = zeros(1);
centroid = zeros(1,col);
clusters = cell(1);

for k=1:clust
    centroid(k, :) = loc(randi(m), :);
end

%disp(centroid);
i = 1;
while true
    %disp(i);
    oldCentroid = centroid;
    
    dist = pdist2(loc, centroid);
    
    for d=1:m
        c = find(dist(d, :) == min(dist(d, :)));
        assign(d) = c(1);
    end
    
    for k=1:clust
        clusters{k} = loc(assign == k, :);
        centroid(k, :) = mean(clusters{k});
    end
    
    if centroid == oldCentroid
        break;
    end
    i = i + 1;
    
end

CM = jet(clust);

figure;
hold on;
for k=1:clust
    t = clusters{k};
    for i=1:length(t(:,1))
        plot(t(i,1),t(i,2),'color',CM(k,:),'marker','o');
    end
end

