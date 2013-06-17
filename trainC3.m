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
    if (nargin < 3) rounds = 100; end;
    [nClasses, nImgs] = size(labels);
    for iClass = 1:nClasses
        fprintf('%d/%d\n',iClass,nClasses);
        training = equalRep(labels(iClass,:));
        trainX = c2(:,training)';
        trainY = labels(iClass,training)';
        switch lower(method)
          case 'gb'
            trainY = double(trainY).*2 - 1; % [0,1] -> [-1,1]
            models{iClass} = gentleBoost(trainX',trainY',options);
          case {'svm','libsvm'}
            models{iClass} = svmtrain(trainY,trainX,options);
        end
    end
end
