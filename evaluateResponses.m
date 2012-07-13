function aucs = evaluateResponses(x,y,cv)
% aucs = evaluateResponses(x,y,cv)

    % here, we have no guarantee of equally many positive and negative examples,
    % which we want to add in the multiclass case.
    [~,probabilities] = classifyResponses(x,y,1,[1 size(x,1)+1],cv,[100],'svm');
    for iPercent = 1:size(probabilities,1)
        for iFold = 1:cv.NumTestSets
            aucs(iPercent,iFold) = auc(probabilities{iPercent,iFold},...
                                       y(cv.test(iFold)));
        end
    end
end
