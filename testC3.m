function c3 = testC3(c2,models,method)
% c3 = testC3(c2,models,method)
%
% generate HMAX C3 responses
% 
% c2: nPatches x nImgs array, C2 responses for an image set
% models: cell array of classifiers, the classifiers giving the C3 responses
% method: string, 'gb' or 'svm', the type of classifier the C3 units are
%
% c3: nModels x nImgs array, the C3 activations
    for iClass = 1:length(models)
        fprintf('%d/%d\n',iClass,length(models));    
        switch lower(method)
          case 'svm'
            testY = ones(size(c2,2),1);
            [~,~,allVals] = svmpredict(testY,c2', models{iClass},'-b 1');
            c3(iClass,:) = allVals(:,2-models{iClass}.Label(1));
          case 'gb'
            [~,c3(iClass,:)] = strongGentleClassifier(c2,models{iClass});
        end
    end
end
