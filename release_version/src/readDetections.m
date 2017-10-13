function [bbox, conf, recnum] = readDetections(detfn, rec)
% [bbox, conf, recnum] = readDetections(detfn, rec)
%
% Reads detection results in format (file_id conf x1 y1 x2 y2) and outputs
% the bounding box, confidence, and corresponding record index

[ids,conf,x1,y1,x2,y2]=textread(detfn,'%s %f %f %f %f %f');

bbox = [x1 y1 x2 y2];

% constrain the bounding box to lie within the image and get the record
% number
bbox = max(bbox, 1);
if exist('rec', 'var')
  recnum = zeros(size(conf));
  for r = 1:numel(rec)
    ind = strcmp(ids, strtok(rec(r).filename, '.'));
    recnum(ind) = r;    
    bbox(ind, 3) = min(bbox(ind, 3), rec(r).imgsize(1));
    bbox(ind, 4) = min(bbox(ind, 4), rec(r).imgsize(2));
  end
end

