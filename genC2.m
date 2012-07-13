function c2 = genC2(gaborSpecs,imgNames,c1bands,linPatches,patchSpecs,USEMATLAB)
% c2 = genC2(gaborSpecs,imgNames,c1bands,linPatches,patchSpecs,USEMATLAB)
% 
% give C2 responses for an image set
%
% gaborSpecs: a struct, holds information needed for creating S1 filters
% imgNames: a cell array, a list of filenames in the image set
% c1bands: a struct, holds information about bands and scales in C1
% linPatches: a cell array, the patches, before being reshaped
% patchSpecs:  a 4 x m array, 
% m = nPatchSizes and 4 rows = [rows; columns; depth; patchesPerSize]
% USEMATLAB: a boolean, if true, use HMAX-MATLAB, else use HMAX-CUDA
%
% c2, an nPatches x nImgs array, C2 responses for an image set

    if (nargin < 6) USEMATLAB = 1; end;

    [filterSizes,linFilters,c1OL,~] = initGabor(gaborSpecs.orientations,...
                                              gaborSpecs.receptiveFieldSizes,...
                                              gaborSpecs.div);

    c2 = zeros(sum(patchSpecs(4,:)),length(imgNames));
    if USEMATLAB
        parfor iImg = 1:length(imgNames)
            c2(:,iImg) = extractC2FromCell(linFilters,filterSizes,...
                                           c1bands.c1Space,c1bands.c1Scale,...
                                           c1OL,linPatches,...
                                           readImages(imgNames(iImg)),...
                                           size(patchSpecs,2),...
                                           patchSpecs(1:3,:),0,[],0,0);
        end
    else

        % create square filters (S1)
        nFilters = length(filterSizes);
        for i = 1:nFilters
            iSize = filterSizes(i);
            filters{i} = reshape(linFilters(1:(iSize^2),i),iSize,iSize);
        end

        % flip patches and reshape into squares
        patchSizes = patchSpecs(1:3,:);
        nPatchSizes = size(patchSizes,2);
        iPatch = 1;
        for iSize = 1:nPatchSizes
            nOrientations = patchSizes(3,iSize);
            dimPatch = [patchSizes(1,iSize) patchSizes(2,iSize)];
            nElements = prod(dimPatch);
            for iLinearPatch = 1:size(linPatches{iSize},2)
                for iOrient = 1:nOrientations
                    patchStart = 1+(iOrient-1)*nElements;
                    patchStop = iOrient*nElements;
                    patches{iPatch} = reshape(linPatches{iSize}(patchStart:patchStop,iLinearPatch),dimPatch);
                    iPatch = iPatch+1;
                end
            end
        end

        for iImg = 1:length(imgNames)
            img = readImages(imgNames(iImg));
            for i = 1:ceil(length(patches)/2048);
                startIn = 2048*i-2047;
                stopIn = min(length(patches),2048*i);
                startOut = (2048/nOrientations)*i-(2048/nOrientations-1);
                stopOut = min(length(patches)/nOrientations,2048/nOrientations*i);
                c2(startOut:stopOut,iImg) = hmaxcudanew(img,filters,patches(startIn:stopIn));
            end
            clear img;
        end
    end
end
