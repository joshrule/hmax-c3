function models = trainC3(c2, labels,rounds)
    if (nargin < 3) rounds = 100; end;
    nImgs = size(c2,2);
    nClasses = size(labels,1);
    parfor iClass = 1:nClasses
        fprintf('%d\n',iClass);
        training = equalRep(labels(iClass,:));
        trainX = c2(:,training);
        trainY = c2(iClass,training) .* 2 - 1;
        models{iClass} = gentleBoost(trainX,trainY,rounds);
    end
end
