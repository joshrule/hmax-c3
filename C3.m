function [c3,models] = C3(c2, labels)
    nImgs = size(c2,2);
    nClasses = size(labels,1);
    parfor iClass = 1:nClasses
        fprintf('%d\n',iClass);
        training = equalRep(labels(iClass,:));
        trainX = c2(:,training);
        trainY = c2(iClass,training) .* 2 - 1;
        models{iClass} = gentleBoost(trainX,trainY,100); % 100 rounds
        [~,c3(iClass,:)] = strongGentleClassifier(c2,models{iClass});
    end
end

%function c3 = C3(c2,labels)
%    nImgs = size(c2,2);
%    nClasses = size(labels,1);
%    for iImg = 1:nImgs
%        validIndices = [1:iImg-1 iImg+1:nImgs];
%        for iClass = 1:nClasses
%            training(validIndices) = equalRep(labels(iClass,validIndices));
%            training(iImg) = 0;
%            preds = classifyResponses(c2,labels(iClass,:),training,'gb');
%            % fprintf('%d %d - iImg: %d, first neg: %d\n',iImg, iClass, iImg, find(find(~training) == iImg));
%            c3(iClass,iImg) = preds(find(find(~training) == iImg));
%        end
%    end
%end
