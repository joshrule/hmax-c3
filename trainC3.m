function models = trainC3(c2,labels,method,options)
% models = trainC3(c2,labels,method,options)
%
% generate classifiers to act as HMAX C3 units
%
% c2: an nPatches x nImgs array, C2 responses for an image set
% labels: an nClasses x nImgs logical array, labels(i,j) = 1 if image j is a
%     member of class i, and is 0 otherwise.
% method: string, 'gb' or 'svm', the classifier to use for C3 units
% options: depends on the classifier:
%   'gb': scalar, the number of boosting rounds to complete
%   'svm': struct, governs classifier and hard negative mining: 
%       svmTrainFlags: options for training C3 classifiers with hard mining
%       svmTestFlags: options for testing C3 classifiers with hard mining
%       alpha: scalar, governs the growth of the hard negative mining
%       startPerIter: scalar, number of images in the first mining iteration
%       threshold: scalar, probability above which a negative is "hard"
%
% models: the binary classifiers on which C3 activations are based
    [nClasses, nImgs] = size(labels);
    for iClass = 1:nClasses
        a = tic;
        fprintf('%d/%d\n',iClass,nClasses);
        training = equalRep(labels(iClass,:),inf,0.5);
        trainX = c2(:,training)';
        trainY = labels(iClass,training)';
        switch lower(method)
          case 'gb'
            trainY = double(trainY).*2 - 1; % [0,1] -> [-1,1]
            models{iClass} = gentleBoost(trainX',trainY',options);
          case {'svm','libsvm'}
            models{iClass} = svmtrain(trainY,trainX,options.svmTrainFlags);
%           detector = svmtrain(trainY,trainX,options.svmTrainFlags);
%           positives = c2(:, logical(labels(iClass,:)));
%           negatives = c2(:,~logical(labels(iClass,:)));
%           shuffledPossibleNegs = randperm(size(negatives,2));
%           negsInUse = negatives(:,shuffledPossibleNegs(1:floor(size(negatives,2)/10)));
%           models{iClass} = hardNegativeMining(positives,negsInUse, ...
%             detector,options.startPerIter,options.alpha,options.threshold, ...
%             options.svmTrainFlags, options.svmTestFlags);
        end
        fprintf('%d: %.3fs to train class\n',iClass,toc(a));
    end
end
