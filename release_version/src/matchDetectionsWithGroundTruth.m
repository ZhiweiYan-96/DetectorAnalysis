function [det, gt] = matchDetectionsWithGroundTruth(det, gt, ovthresh)
% [det, gt] = matchDetectionsWithGroundTruth(det, gt, ovthresh)
%
% Returns the intersection over union, intersection/object, and
% intersection/gt areas and index of closest ground truth for each
% detection
%
% Input:
%   Ground truth anntoations: 
%       gt.(bbox, rnum, isdiff)
%   Object detection results:
%       det.(bbox, conf, rnum)
%
% Output
%   Adds to the input det struct:
%     the index of best-matching ground truth (gtnum), the corresponding
%     overlap measures, the label (1=true, 0=difficult, -1=false), and whether
%     the detection is a duplicate (label=-1 in this case) 
%       det.(bbox, conf, rnum, gtnum, isdiff, ov, ov_obj, ov_gt, label, isduplicate)
%   Adds to the input gt struct:
%     (1) index (detnum) of highest scoring detection with ov>ovthresh;
%     (2) index (detnum_ov) of maximum overlap detection
%     (3) the overlaps with the detection of maximum overlap (ov, ov_obj,
%     ov_gt), which may not be the same detection
%       gt.(bbox, rnum, isdiff, detnum): 


Ngt = size(gt.bbox, 1);
gt.detnum = zeros(Ngt, 1);
gt.detnum_ov = zeros(Ngt, 1);
gt.ov = zeros(Ngt, 1);
gt.ov_obj = zeros(Ngt, 1);
gt.ov_gt = zeros(Ngt, 1);


Nd = size(det.bbox, 1);
det.gtnum = zeros(Nd, 1);
det.ov = zeros(Nd, 1);
det.ov_obj = zeros(Nd, 1);
det.ov_gt = zeros(Nd, 1);
det.isdiff = zeros(Nd, 1);
det.label = -ones(Nd, 1);
det.isduplicate = false(Nd, 1);

isdetected = zeros(Ngt, 1);
[sv, si] = sort(det.conf, 'descend');

for dtmp = 1:Nd
  
  d = si(dtmp);
     
  indgt = find(gt.rnum == det.rnum(d));
 
  if isempty(indgt), continue; end
  
  bbgt = gt.bbox(indgt, [1 3 2 4]);  % ground truth in same image
  box = det.bbox(d, [1 3 2 4]);    % detection window
  
  bi=[max(box(1),bbgt(:, 1))  max(box(3),bbgt(:, 3))  ...
    min(box(2),bbgt(:, 2))  min(box(4),bbgt(:, 4))];

  iw=bi(:, 3)-bi(:, 1)+1;
  ih=bi(:, 4)-bi(:, 2)+1;

  ind = find(iw >0 & ih > 0); % others have no intersection
  if ~isempty(ind)
    a1 = (bbgt(ind, 2)-bbgt(ind, 1)+1).*(bbgt(ind, 4)-bbgt(ind, 3)+1);
    a2 = (box(2)-box(1)+1)*(box(4)-box(3)+1);    
    intersectArea = iw(ind).*ih(ind);
    unionArea = a1 + a2 - intersectArea;
    
    i = find((intersectArea ./ unionArea) >= ovthresh & (~isdetected(indgt(ind))));
    if ~isempty(i) % correct detection
      [det.ov(d), i] = max((intersectArea ./ unionArea) .* (~isdetected(indgt(ind))));  
      gti = indgt(ind(i));
      if gt.isdiff(gti)
        det.label(d) = 0;
        det.isdiff(d) = 1;
      else
        det.label(d) = 1;        
      end      
      gt.detnum(gti) = d;
      isdetected(gti) = 1;
    else % no correct detection, or extra detection
      [det.ov(d), i] = max(intersectArea ./ unionArea);
      gti = indgt(ind(i));
      if det.ov(d)>=ovthresh
        det.isduplicate(d) = true;
      end
    end
        
    det.ov_obj(d) = intersectArea(i) ./ a2;
    det.ov_gt(d) = intersectArea(i) ./ a1(i);
    det.gtnum(d) = gti;       
    
    if det.ov(d) > gt.ov(gti)
      gt.ov(gti) = det.ov(d);    
      gt.ov_obj(gti) = intersectArea(i) ./ a2;
      gt.ov_gt(gti) = intersectArea(i) ./ a1(i);
      gt.detnum_ov(gti) = d;
    end
      
  end
end