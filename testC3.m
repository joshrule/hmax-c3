function c3 = testC3(c2,models)
% c3 = testC3(c2,models)
%
% generate HMAX C3 responses
% 
% c2: nPatches x nImgs array, C2 responses for an image set
% models: cell array of classifiers, the classifiers giving the C3 responses
%
% c3: nModels x nImgs array, the C3 activations
    for iClass = 1:length(models)
        testY = ones(size(c2,2),1);
        [~,~,~,allVals] = evalc('svmpredict(testY,c2'', models{iClass},''-b 1'')');
        c3(iClass,:) = allVals(:,2-models{iClass}.Label(1));
    end
end
