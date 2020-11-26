format long g;

input = VideoReader('input.mov');

test = cell(1);

k = 0;
while hasFrame(input)
    k = k+1;
    test{k} = rgb2gray(readFrame(input));
end

[testHeight, testWidth] = size(test{1});

reff = rgb2gray(imread('reference.jpg'));
[reffHeight, reffWidth] = size(reff);

%window  = [1 2 4 8 16 32 64 128 256 512];
window  = 8; 
avg = cell(1);

for w = 1:length(window)
    
    tic;
    
    output = QTWriter('outtest2.MOV');
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
            
            block = test{t}(i:reffHeight+i-1, j:reffWidth+j-1) - reff;
            block = block.^2;
            
            match = sum(block(:));
            
            
            if (bestMatch > match)
                bestMatch = match;
                bestI = i;
                bestJ = j;
            end
            
        end
        
    end
    
    im = insertShape(test{t}, 'rectangle', [bestJ bestI reffWidth reffHeight],'Color','r','LineWidth',2);
    writeMovie(output,im);
    
    disp(t);
    
    totalIter = 0;
    
    for t = 2:k
        
        newP = p;
        
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
            %disp(i);
            
            
            for c = 1:length(points)
                
                point = points{c};
                i = point(1);
                j = point(2);
                
                %disp([i reffHeight+i-1 j reffWidth+j-1]);
                
                block = test{t}(i:reffHeight+i-1, j:reffWidth+j-1) - reff;
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
        
        %disp(bestIter);
        totalIter = totalIter + bestIter;
        
        im = insertShape(test{t}, 'rectangle', [bestJ bestI reffWidth reffHeight],'Color','r','LineWidth',2);
        writeMovie(output,im);
        
        disp(t);
    end
    
    save(output);
    

    
end

