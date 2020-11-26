format long g;
hold on;
legendInfo = cell(1);

input = VideoReader('input.mov');

testImage = cell(1);

k = 0;
while hasFrame(input)
    k = k+1;
    testImage{k} = rgb2gray(readFrame(input));
end

[testHeight, testWidth] = size(testImage{1});

reff = rgb2gray(imread('reference.jpg'));
[reffHeight, reffWidth] = size(reff);

depth = 2;
window  = 8;
avg = cell(1);

for d = 1:length(depth)
    for w = 1:length(window)
        
        tic;
        output = QTWriter('out3.MOV');
        output.FrameRate = input.FrameRate;
        
        p = window(w);
        
        bestMatch = inf;
        bestI = 1;
        bestJ = 1;
        
        t = 1;
        
        exIter = 0;
        
        for i = 1:testHeight - reffHeight
            for j = 1:testWidth - reffWidth
                
                exIter = exIter + 1;
                
                block = testImage{t}(i:reffHeight+i-1, j:reffWidth+j-1) - reff;
                block = block.^2;
                
                match = sum(block(:));
                
                if (bestMatch > match)
                    bestMatch = match;
                    bestI = i;
                    bestJ = j;
                end
                
            end
            
        end
        
        im = insertShape(testImage{t}, 'rectangle', [bestJ bestI reffWidth reffHeight],'Color','r','LineWidth',2);
        writeMovie(output,im);
        
        disp(t);
        
        level = depth(d);
        
        versionReff = cell(1);
        versionReff{1} = reff;
        
        for l = 2:level+1
            versionReff{l} = imresize(versionReff{l-1}, 0.5);
        end
        
        bestIJ = cell(1);
        bestIJ{1} = [bestI bestJ];
        
        totalIter = 0;
        for t = 2:k
            
            versionTest = cell(1);
            versionTest{1} = testImage{t};
            
            newP = p;
            
            
            for l = 2:level+1
                versionTest{l} = imresize(versionTest{l-1}, 0.5);
                newP = newP/2;
                bestIJ{l} = bestIJ{l-1}/2;
            end
            
            newP = round(newP);
            
            l = level + 1;
            
            bestI = round(bestIJ{l}(1));
            bestJ = round(bestIJ{l}(2));
            test = versionTest{l};
            [testHeight, testWidth] = size(test);
            reff = versionReff{l};
            [reffHeight, reffWidth] = size(reff);
            
            
            flag = false;
            
            %newD = power(2, ceil(log2(newP))-1);
            
            bestMatch = inf;
            
            iter = 0;
            bestIter = iter;
            
            while (newP ~= 1)
                iter = iter + 1;
                
                newP = round(newP/2);
                %newD = power(2, ceil(log2(newP))-1);
                
                points = cell(1);
                i = 0;
                for x = -newP:newP:newP
                    for y = -newP:newP:newP
                        if (flag && x == 0 && y == 0)
                            continue;
                        end
                        
                        newX = x + bestI;
                        newY = y + bestJ;
                        
                        if (newX > testHeight - reffHeight || newX <= 0)
                            continue;
                        elseif (newY > testWidth - reffWidth || newY <= 0)
                            continue;
                        end
                        
                        i = i + 1;
                        points{i} = [newX newY];
                        
                    end
                end
                
                flag = true;
                
                for c = 1:length(points)
                    
                    point = points{c};
                    i = point(1);
                    j = point(2);
                    
                    block = test(i:reffHeight+i-1, j:reffWidth+j-1) - reff;
                    block = block.^2;
                    
                    match = sum(block(:));
                    
                    if (bestMatch > match)
                        bestMatch = match;
                        bestI = i;
                        bestJ = j;
                        bestIter = iter;
                    end
                    
                end
                
            end
            
            bestIJ{l} = [bestI bestJ];
            
            bestIJ{l-1} = floor((bestIJ{l-1}+ 2 * bestIJ{l})/2);
            
            for l = level:-1:1
                
                bestI = bestIJ{l}(1);
                bestJ = bestIJ{l}(2);
                test = versionTest{l};
                [testHeight, testWidth] = size(test);
                reff = versionReff{l};
                [reffHeight, reffWidth] = size(reff);
                
                
                newP = 1;
                bestMatch = inf;
                
                points = cell(1);
                i = 0;
                for x = -newP:newP:newP
                    for y = -newP:newP:newP
                        
                        newX = x + bestI;
                        newY = y + bestJ;
                        
                        if (newX > testHeight - reffHeight || newX <= 0)
                            continue;
                        elseif (newY > testWidth - reffWidth || newY <= 0)
                            continue;
                        end
                        i = i + 1;
                        points{i} = [newX newY];
                        
                    end
                end
                
                for c = 1:length(points)
                    
                    point = points{c};
                    i = point(1);
                    j = point(2);
                    
                    
                    block = test(i:reffHeight+i-1, j:reffWidth+j-1) - reff;
                    block = block.^2;
                    
                    match = sum(block(:));
                    
                    if (bestMatch > match)
                        bestMatch = match;
                        bestI = i;
                        bestJ = j;
                    end
                    
                end
                
                
                bestIJ{l} = [bestI bestJ];
                
                %disp(l);
                
                if (l==1)
                    break;
                end
                
                bestIJ{l-1} = floor((bestIJ{l-1}+ 2 * bestIJ{l})/2);
                
            end
            
            %disp(bestIter);
            totalIter = totalIter + bestIter;
            
            im = insertShape(test, 'rectangle', [bestJ bestI reffWidth reffHeight],'Color','r','LineWidth',2);
            writeMovie(output,im);
            
            disp(t);
            
        end
        
        save(output);
            
    end
    
    
end
