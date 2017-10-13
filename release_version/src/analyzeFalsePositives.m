function result = analyzeFalsePositives(gtall, objind, similar_ind, det, normalizedCount, topN)
% result = analyzeFalsePositives(gt, gt_similar, det, normalizedCount)

LOW_OV = 0.1;
HIGH_OV = 0.5;

gt = gtall(objind);
gt_similar = gtall(similar_ind);

[sv, si] = sort(det.conf, 'descend');
det.bbox = det.bbox(si, :);
det.conf = det.conf(si);
det.rnum = det.rnum(si);

% Regular
overlap_thresh = HIGH_OV;
[det, gt] = matchDetectionsWithGroundTruth(det, gt, overlap_thresh);
npos = sum(~[gt.isdiff]);
result.all = averagePrecisionNormalized(det.conf, det.label, npos, normalizedCount);
result.iscorrect = (det.label>=0);

% Ignore localization error: remove detections that are duplicates or have
% poor localization
overlap_thresh = LOW_OV;
det2 = matchDetectionsWithGroundTruth(det, gt, overlap_thresh);
result.isloc = (det.label==-1) & (det2.label>=0 | det2.isduplicate);

% Code below sets confidence of localization errors to -Inf
conf = det.conf;
conf(result.isloc) = -Inf;
result.ignoreloc = averagePrecisionNormalized(conf, det.label, npos, normalizedCount);

% Code below reassigns poor localizations to correct detections
det2.conf(det2.isduplicate) = -Inf;
result.fixloc = averagePrecisionNormalized(det2.conf, det2.label, npos, normalizedCount);

conf = det.conf;
conf(~result.isloc & (det.label==-1)) = -Inf;
result.onlyloc = averagePrecisionNormalized(conf, det.label, npos, normalizedCount);

% Ignore similar objects
overlap_thresh = LOW_OV;
confuse_sim = false(det.N, 1);
if ~isempty(gt_similar)
  for o2 = 1:numel(gt_similar)
    det2_s(o2) = matchDetectionsWithGroundTruth(det, gt_similar(o2), overlap_thresh);
  end
  conf = det.conf;
  confuse_sim = (any(cat(2, det2_s.label)>=0, 2) | any(cat(2, det2_s.isduplicate), 2)) & (det.label==-1) & (~result.isloc);  
  conf(confuse_sim) = -Inf;
  result.ignoresimilar = averagePrecisionNormalized(conf, det.label, npos, normalizedCount);
  conf = det.conf;
  conf(~confuse_sim & (det.label==-1)) = -Inf;
  result.onlysimilar = averagePrecisionNormalized(conf, det.label, npos, normalizedCount);
end
result.issim = confuse_sim;

% Ignore background detections (all other false positives)
bg_error = (~result.isloc) & (det.label==-1) & (~result.issim);
result.isbg = bg_error;
conf = det.conf;
conf(bg_error) = -Inf;
result.ignorebg = averagePrecisionNormalized(conf, det.label, npos, normalizedCount);
conf = det.conf;
conf(~bg_error & (det.label==-1)) = -Inf;
result.onlybg = averagePrecisionNormalized(conf, det.label, npos, normalizedCount);

% Record false positives with other (non-similar) objects
isother = zeros(size(result.isbg));
for k = setdiff(1:numel(gtall), [similar_ind objind])
  detk = matchDetectionsWithGroundTruth(det, gtall(k), LOW_OV);
  isother = isother | (~result.iscorrect & ~result.issim & ~result.isloc & detk.label>=0);
end
result.isother = isother;
result.isbg_notobj = result.isbg & ~result.isother;

% Ignore localization error and similar objects
conf = det2.conf;
conf(confuse_sim) = -Inf;
result.ignorelocsim = averagePrecisionNormalized(conf, det2.label, npos, normalizedCount);

% Get counts of types of false positives for topN detections
result.confuse_count.object = zeros(numel(gtall), numel(topN));
result.confuse_count.correct = zeros(1, numel(topN));
result.confuse_count.loc = zeros(1, numel(topN));
result.confuse_count.bg = zeros(1, numel(topN));
for n = 1:numel(topN)
  if topN(n)<0
    topN(n) = find(cumsum(det.label==-1)==-topN(n), 1, 'first');
  end
  detn.bbox = det.bbox(1:topN(n), :);
  detn.conf = det.conf(1:topN(n));
  detn.rnum = det.rnum(1:topN(n));  
  det2 = matchDetectionsWithGroundTruth(detn, gt, HIGH_OV);
  iscorrect = det2.label>=0;
  det2 = matchDetectionsWithGroundTruth(detn, gt, LOW_OV);
  isloc = (det2.label>=0 | det2.isduplicate) & ~iscorrect;    
  isobj = false(topN(n), numel(gtall));
  objov = zeros(topN(n), numel(gtall));
  for k = setdiff(1:numel(gtall), objind)
    det2 = matchDetectionsWithGroundTruth(detn, gtall(k), LOW_OV);
    isobj(:, k) = ((det2.label>=0) | det2.isduplicate) & ~isloc & ~iscorrect;
    objov(isobj(:, k), k) = det2.ov(isobj(:, k));
  end
  [mv, mi] = max(objov, [], 2);
  mi(mv==0) = 0;  
  isobj = false(topN(n), 1);
  isobj(mi>0) = true;
  isbg = ~iscorrect & ~isloc & ~isobj;
  for k = setdiff(1:numel(gtall), objind)
    result.confuse_count.object(k, n) = sum(mi==k);
  end  
  result.confuse_count.total(n) = topN(n);
  result.confuse_count.correct(n) = sum(iscorrect);
  result.confuse_count.loc(n) = sum(isloc);
  result.confuse_count.bg(n) = sum(isbg);  
  result.confuse_count.similarobj(n) = sum(result.confuse_count.object(similar_ind, n));
  result.confuse_count.otherobj(n) = topN(n)-sum(iscorrect)-sum(isloc)-sum(isbg)-result.confuse_count.similarobj(n);
end 
    
        
    
