function p = randomPatches(sizes)
% patches = randomPatches(imgNames, patchSpecs,method)
%
% generate a set of random patches
%
% sizes:  a 4 x m array, 
% m = nPatchSizes and 4 rows = [rows; columns; depth; patchesPerSize]
%
% p: a cell array of length nPatchSizes, the patches

    p = arrayfun(@(x) rand(prod(sizes(1:3,x)),sizes(4,x)),...
                 1:size(sizes,2),...
                 'UniformOutput',0);
end
