function models = trainC3(c2,labels,method,options)
% models = trainC3(c2,labels,rounds)
%
% This function generates classifiers for HMAX C3 activations.
%
% c2: an nPatches x nImgs array, C2 responses for an image set
% labels: an nClasses x nImgs logical array, labels(i,j) = 1 if image j is a
%     member of class i, and is 0 otherwise.
% rounds: a scalar, the number of rounds of gentleBoost training to use
%
% models: the gentleBoost classifiers on which the activations are based
    [nClasses, nImgs] = size(labels);
    for iClass = 1:nClasses
        a = tic;
        fprintf('%d/%d\n',iClass,nClasses);
        training = equalRep(labels(iClass,:));
        trainX = c2(:,training)';
        trainY = labels(iClass,training)';
        switch lower(method)
          case 'gb'
            trainY = double(trainY).*2 - 1; % [0,1] -> [-1,1]
            models{iClass} = gentleBoost(trainX',trainY',options);
          case {'svm','libsvm'}
            detector = svmtrain(trainY,trainX,options.svmTrainFlags);
            positives = c2(:, logical(labels(iClass,:)));
            negatives = c2(:,~logical(labels(iClass,:)));
            shuffledPossibleNegs = randperm(size(negatives,2));
            negsInUse = negatives(:,shuffledPossibleNegs(1:floor(size(negatives,2)/10)));
            models{iClass} = hardNegativeMining(positives,negsInUse, ...
              detector,options.startPerIter,options.alpha,options.threshold, ...
              options.svmTrainFlags, options.svmTestFlags);
        end
        fprintf('%d: %.3fs to train class\n',iClass,toc(a));
    end
end
