input = VideoReader('/home/sadat/Desktop/TemplateMatching/input.mov');

test = cell(1);

k = 0;
while hasFrame(input)
    k = k+1;
    test{k} = rgb2gray(readFrame(input));
end

[testHeight, testWidth] = size(test{1});

reff = rgb2gray(imread('/home/sadat/Desktop/TemplateMatching/reference.jpg'));
[reffHeight, reffWidth] = size(reff);

output = QTWriter('/home/sadat/Desktop/TemplateMatching/out1.MOV');
output.FrameRate = input.FrameRate;

for t = 1:1
    
    bestMatch = inf;
    bestI = 1;
    bestJ = 1;
    
    for i = 1:testHeight - reffHeight
        for j = 1:testWidth - reffWidth
            
            
            block = test{t}(i:reffHeight+i-1, j:reffWidth+j-1) - reff;
            block = block.^2;
            
            match = sum(block(:));
            
            %disp(match);
            
            if (bestMatch > match)
                bestMatch = match;
                bestI = i;
                bestJ = j;
            end
            
        end
        
    end
    
    %disp(bestMatch);
    %disp([bestI bestJ]);
    
    im = insertShape(test{t}, 'rectangle', [bestJ bestI reffWidth reffHeight],'Color','r','LineWidth',2);
    writeMovie(output,im);
    
    disp(t);
    
end

save(output);

