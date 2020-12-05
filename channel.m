fp = fopen('train.txt');
trainingData = fread(fp)-'0';
fclose(fp);
trainingData = trainingData(:,1);

info = dlmread('parameter.txt');

coeff = info(1:length(info)-1);
sigma = info(length(info));

rl = length(coeff);

states = 0:2^rl-1;
states = rem(floor(states'*pow2(-(rl-1):0)),2);

maxI = 100000;
received = Inf(1);
gap = rl-1;

for s=1:maxI
    sub = trainingData(s:s+gap);
    noise = normrnd(0, sigma);
    received(s) = coeff * sub + noise;
end
obs = Inf(1,rl);

for s=1:maxI-gap
    obs(s,:) = received(s:s+gap);
end

st = Inf(1);

for s=1:maxI-gap
    st(s) = 1+bin2dec(num2str(trainingData(s:s+rl)'));
end

maxclust = 2^(rl+1);
clust = Inf(1,rl);

prior = zeros(maxclust,1);

for s=1:maxclust
    ind = find(st == s);
    prior(s) = length(ind);
    clust(s, :) = mean(obs(ind,:));
end

scatter(clust(:, 1), clust(:, 2));

prior = prior ./ sum(prior);
disp("prior prob");
disp(prior);
transProb = zeros(maxclust);

for s=1:maxI-gap-1
    t = st(s);
    u = st(s+1);
    transProb(t,u) = transProb(t,u) + 1;
end

transProb = transProb ./ sum(transProb, 2);
disp("Transition Prob");
disp(transProb);

fp = fopen('test.txt');
test = fread(fp)-'0';
fclose(fp);
test = test(:,1);

testData = test;

l = length(test);
testData = [testData; zeros(gap,1);];

received = Inf(1);

for s=1:l
    sub = testData(s:s+gap);
    noise = normrnd(0, sigma);
    received(s) = coeff * sub + noise;
end



norm_noise = zeros(1);
distance = Inf(maxclust, l);
parent = Inf(maxclust, l);
%disp("clust");
%disp(clust);

distanceO = Inf(1,l);
parentO = Inf(1,l);

for g=1:maxclust
    norm_noise(g) = log(normpdf(received(1), clust(g), sigma));
end

distance(:, 1) = log(prior) + norm_noise';

distanceO(1) = max(distance(:, 1));
f = find(distance(:, 1) == max(distance(:, 1)));
parentO(1) = f(1);

%method 1
dist1 = zeros(1);
a = length(test);
for k=2:l
    
    for c=1:maxclust
        
        for g=1:maxclust
            norm_noise = log(normpdf(received(k), clust(g), sigma));
            dist1(g) = distance(g,k-1) + log(transProb(g,c)) + norm_noise;
        end
        
        distance(c, k) = max(dist1);
        dis = find(dist1 == max(dist1));
        parent(c, k) = dis(1);
        
    end
   
    maxPar = find(distance(:, k) == max(distance(:, k)));
    parentO(k) = parent(maxPar(1), k);
    
end

out1 = zeros(1);
corr1 = 0;
for p=1:l
    str = dec2bin(parentO(p)-1, rl+1);
    out1(p) = str(1);
    if (out1(p)-'0' == test(p))
        corr1 = corr1 + 1;
    end
end
disp("--------Method1-----");
disp("correct1");
disp(corr1);

disp("accuracy 1");
accu1 = (corr1/a)*100 ;
disp(accu1);
fp = fopen('out1.txt', 'w');
fwrite(fp, out1);
fclose(fp);


dist2 = zeros(1);

%method 2
for k=2:l
    
    for c=1:maxclust
        
        for g=1:maxclust
            norm_noise = log(normpdf(received(k), clust(g), sigma));
            dist2(g) = abs(received(k)-clust(g)) + norm_noise;
        end
        
        distance(c, k) = max(dist2);
        dis = find(dist2 == max(dist2));
        parent(c, k) = dis(1);
        
    end
    maxPar = find(distance(:, k) == max(distance(:, k)));
    parentO(k) = parent(maxPar(1), k);
    
end

out2 = zeros(1);
corr2 = 0;
for p=1:l
    str = dec2bin(parentO(p)-1, rl+1);
    out2(p) = str(1);
    if (out2(p)-'0' == test(p))
        corr2 = corr2 + 1;
    end
end

disp("--------Method2-----");
disp("correct2");
disp(corr2);

disp("accuracy 2");
accu2 = (corr2/a)*100 ;
disp(accu2);
fp = fopen('out2.txt', 'w');
fwrite(fp, out2);
fclose(fp);
