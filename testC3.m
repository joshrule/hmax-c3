function c3 = testC3(c2,models,method,firstLabel)
% c3 = testC3(c2,models,method)
%
% This function generates HMAX C3 activations.
% 
% c2: an nPatches x nImgs array, C2 responses for an image set
% models: the gentleBoost classifiers on which the activations are based
%
% c3: an nClasses x nImgs array, the C3 activations
    if (nargin < 4) firstLabel = NaN; end;
    for iClass = 1:length(models)
        fprintf('%d/%d\n',iClass,length(models));    
        switch lower(method)
          case 'svm'
            testY = ones(size(c2,2),1);
            [~,~,allVals] = svmpredict(testY,c2', models{iClass},'-b 1');
            c3(iClass,:) = allVals(:,2-firstLabel);
          case 'gb'
            [~,c3(iClass,:)] = strongGentleClassifier(c2,models{iClass});
        end
    end
end
