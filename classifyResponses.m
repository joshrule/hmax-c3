function [fi, probabilities] = classifyResponses(x,y,class,featureGroups,cv,percentages,classifier)
% [fi, probabilities] = classifyResponses(x,y,class,featureGroups,cv,percentages,classifier)
%
% a set of svmprobs for binary classification over a given image set
%
% x: an [nFeatures nExamples] array holding feature values
% y: a vector of length nExamples, the class labels of examples in x
% featureGroups: an array of indices, breaking features into groups. If the
%     array is n elements long, n-1 classes will be used when calculating the
%     fisher information numbers. For length(featureGroups) < 2, random features
%     will be used. Use featureGroups = [1 nFeatures+1] to calculate FI over a
%     single class.
% cv: a cross-validation object
% percentages: a vector of percentages of features to use in classification
%
% fi: the fisher information calculated for selecting patches
% probabilities: the probability estimates from the classifier

    nFolds = cv.NumTestSets;
    fi = struct(); % default return value
    targY = y(class,:);

    for iPercent = 1:length(percentages)
        for iFold = 1:nFolds
            testExamples = cv.test(iFold);
            trainExamples = cv.training(iFold);

            if length(featureGroups) < 2 % random
                indices = randperm(size(x,1));
            else % FI
                fish = [];
                for iClass = class:length(featureGroups)-1
                    fiY = y(iClass,:);
                    trainTargs = find(fiY' == trainExamples);
                    trainDtors = find((-1*fiY + 1)' == trainExamples);
                    featureStart = featureGroups(iClass);
                    featureStop = featureGroups(iClass+1)-1;
                    newFI = fisher(x(featureStart:featureStop,trainTargs)',...
                                   x(featureStart:featureStop,trainDtors)');
                    fish = [fish newFI];
                end
                [fi(iFold).sortedFI fi(iFold).indices] = sort(fish,'descend');
                indices = fi(iFold).indices;
            end
            featuresSelected = indices(1:floor(percentages(iPercent)/100*size(x,1)));

            trainX = x(featuresSelected,trainExamples)';
            testX  = x(featuresSelected,testExamples)';
            trainY = targY(trainExamples)';
            testY  = targY(testExamples)';
            if strcmp(classifier,'svm')
                model = svmtrain(trainY,trainX, '-q -t 0 -b 1');
                [~,~,probs] = svmpredict(testY,testX, model, '-b 1');
                probabilities{iPercent,iFold} = probs(:,2-trainY(1));
            else % gentleBoost
                nRounds = 100;
                trainY = double(trainY);
                trainY(trainY == 0) = -1;
                model = gentleBoost(trainX',trainY',nRounds);
                [classes, probs] = strongGentleClassifier(testX',model);
                probabilities{iPercent,iFold} = (probs+nRounds)/(2*nRounds);
            end
        end
    end
end
