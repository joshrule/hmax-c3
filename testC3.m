function c3 = testC3(c2,models)
    parfor iClass = 1:length(models)
        [~,c3(iClass,:)] = strongGentleClassifier(c2,models{iClass});
    end
end
