% detectionAnalysisScript (main script)

DO_TP_ANALYSIS = 1;
DO_FP_ANALYSIS = 1;
DO_TP_DISPLAY = 1;
DO_FP_DISPLAY = 1;
DO_TEX = 1;
DO_SHOW_SURPRISING_MISSES = 0;

NORM_FRACT = 0.15; % parameter for setting normalized precision (default = 0.15)

% objects with extra annotation
objnames_extra = {'aeroplane', 'bicycle', 'bird', 'boat', 'cat', ...
  'chair', 'diningtable'}; 

% all 20 VOC object names
objnames_all = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', ...
       'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', ...
       'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'}; 

% objects to analyze (could be a subset)    
objnames_selected  = objnames_all; %{'aeroplane'}; 
     
% set to use for analysis     
imset = 'test';

% needs to be set for your computer
imdir = '/home/dhoiem/data/pascal07/VOCdevkit/VOC2007/JPEGImages/';
VOCsourcepath = './VOCcode';  % change this for later VOC versions
VOCset = 'VOC2007';
addpath(VOCsourcepath);

annotationdir = '../annotations';

% set of detectors to evaluate
full_set = {'felzenszwalb_v2', 'felzenszwalb_v3', 'felzenszwalb_v4', 'vedaldi2009'};
detectors = {'felzenszwalb_v3', 'felzenszwalb_v4', 'vedaldi2009'};  %{'felzenszwalb_v4', 'vedaldi2009'};
for d = 1:numel(detectors);
  detector = detectors{d};
  fprintf('\nevaluating detector %s\n\n', detector);
  
  switch detector
    case 'felzenszwalb_v2'
      detpath = '../detections/felzenszwalb_v2/fel_v2_VOC2007_%s_det.txt';
      resultdir = '../results/felzenszwalb_v2';
      detname = 'FGMR (v2)';  
    case 'felzenszwalb_v3'
      detpath = '../detections/felzenszwalb_v3/fel_v3_VOC2007_%s_det.txt';
      resultdir = '../results/felzenszwalb_v3';
      detname = 'FGMR (v3)';  
    case 'felzenszwalb_v4'
      detpath = '../detections/felzenszwalb_v4/fel_v4_VOC2007_%s_det.txt';
      resultdir = '../results/felzenszwalb_v4';  
      detname = 'FGMR (v4)';    
    case 'vedaldi2009'
      detpath = '../detections/vedaldi2009/ts07full-%s-st0.txt';
      resultdir = '../results/vedaldi2009';
      detname = 'VGVZ 2009';  
    otherwise
      error('unknown detector')
  end
  if ~exist(resultdir, 'file')
    try, mkdir(resultdir); catch; end;
  end

  %% Read the records, attach annotations, and save
  outfn = fullfile(annotationdir, sprintf('%s_records_%s.mat', VOCset, imset));
  if ~exist(outfn, 'file')
    rec = PASreadAllRecords(imset, 'main');
    for o = 1:numel(objnames_extra)    
      annotationpath = fullfile(annotationdir, 'gt_ved_%s.txt');
      rec = updateRecordAnnotations(rec, annotationpath, objnames_extra{o});
    end
    for r = 1:numel(rec)
      for n = 1:numel(rec(r).objects)
        if isfield(rec(r).objects(n), 'details') && ~isempty(rec(r).objects(n).details)
          rec(r).objects(n).detailedannotation = 1;
        else
          rec(r).objects(n).details = [];
          rec(r).objects(n).detailedannotation = 0;        
        end
      end
    end
    save(outfn, 'rec'); 
  else
    load(outfn, 'rec');
  end

  %% Analyze true positives and their detection confidences
  % Get overall and individual AP and PR curves for: occlusion, part visible,
  % side visible, aspect ratio, and size.  For aspect ratio, split into
  % bottom 10%, 10-30%, middle 40%, 70-90%, and top 10%.  Same for size
  % (measured in terms of height).  For AP, compute confidence interval (it
  % is a mean of confidence).
  if DO_TP_ANALYSIS
    for o = 1:numel(objnames_selected)
      objname = objnames_selected{o}; 
      disp(objname)
      
      % Read ground truth and detections
      usediff = true;
      clear gt;
      [gt.ids, gt.bbox, gt.isdiff, gt.istrunc, gt.isocc, gt.details, gt.rnum, gt.onum] = ...
        PASgetObjects(rec, objname, usediff);
      [det.bbox, det.conf, det.rnum] = readDetections(sprintf(detpath, objname), rec);
      det.N  =size(det.bbox, 1);
      gt.N = size(gt.bbox, 1);
      det.nimages = numel(rec);

      outdir = resultdir;
      if ~exist(outdir, 'file'), mkdir(outdir); end;  

      % Compute normalized PR/AP for various subsets of objects
      nposNorm = NORM_FRACT*det.nimages;
      result = analyzeTrueDetections(gt, det, rec, nposNorm, 0.1, 0.01);
      result.name = objname;
      save(fullfile(outdir, sprintf('results_%s_ov10.mat', objname)), 'result');  
      result = analyzeTrueDetections(gt, det, rec, nposNorm, 0.5, 0.01);
      result.name = objname;
      save(fullfile(outdir, sprintf('results_%s_ov50.mat', objname)), 'result');

    end
  end  

  %% False positive analysis
  if DO_FP_ANALYSIS
        
    % load ground truth
    clear gt;
    usediff = true;
    for o = 1:numel(objnames_all)
      objname = objnames_all{o}; 
      [gt(o).ids, gt(o).bbox, gt(o).isdiff, gt(o).istrunc, gt(o).isocc, ...
        gt(o).details, gt(o).rnum, gt(o).onum] = PASgetObjects(rec, objname, usediff);
      gt(o).N = size(gt(o).bbox, 1);
    end
    
    % specify sets of similar objects
    animals = [3 8 10 12 13 15 17];    % animals + person (15)
    vehicles1 = [1 4 6 7 19 2 14]; % exclude bicycle motorcycle
    vehicles2 = [2 14]; % bicycle motorcycle
    furniture = [9 11 18]; % chair, table, sofa    
    airobjects = [1 3]; % bird, airplane
    allsimilar = {animals, vehicles1, vehicles2, furniture, airobjects};        
    
    for o = 1:numel(objnames_selected)
      objname = objnames_selected{o}; 
      o_all = find(strcmp(objnames_all, objname));
      
      % Read detections
      [det.bbox, det.conf, det.rnum] = readDetections(sprintf(detpath, objname), rec);
      det.N  =size(det.bbox, 1);
      det.nimages = numel(rec);
      nposNorm = NORM_FRACT*det.nimages;          

      outdir = resultdir;
      if ~exist(outdir, 'file'), mkdir(outdir); end;  

      % Get indices of similar objects
      similar_ind = [];
      for k = 1:numel(allsimilar), 
        if any(allsimilar{k}==o_all)
          similar_ind = union(similar_ind, setdiff(allsimilar{k}, o_all));
        end
      end      
      
      % Analyze FP
      % topN: -X means top X false positives; +X means top X of all detections
      topN = [-100 -sum(~[gt(o_all).isdiff]) sum(~[gt(o_all).isdiff])];                  
      result_fp = analyzeFalsePositives(gt, o_all, similar_ind, det, nposNorm, topN);
      result_fp.name = objname;
      result_fp.o = o_all;
      cc = result_fp.confuse_count;
      fprintf('%s:\tloc=%d  bg=%d  similar=%d  other=%d\n', objname, cc.loc(1), cc.bg(1), cc.similarobj(1), cc.otherobj(1));
      save(fullfile(outdir, sprintf('results_fp_%s.mat', objname)), 'result_fp');  

    end
  end

  %% Create plots and .txt files for true positive analysis
  if DO_TP_DISPLAY
    
    detail_subset = [1 4 5 6]; % objects for which to create per-object detailed plots 
    plotnames = {'occlusion', 'area', 'height', 'aspect', 'truncation', 'parts', 'view'}; 
    
    ovstr = 'ov50'; % set to ov10 to do analysis ignoring localization error
    clear result;
    for o = 1:numel(objnames_extra)
      tmp = load(fullfile(resultdir, sprintf('results_%s_%s.mat', objnames_extra{o}, ovstr)));
      result(o) = tmp.result;
    end
    
    % Create plots for all objects and write out the first five plots
    displayPerCharacteristicPlots(result, detname)
    for f = 1:5
      set(f, 'PaperUnits', 'inches'); set(f, 'PaperSize', [8.5 11]); set(f, 'PaperPosition', [0 11-3 8.5 2.5]);
      print('-dpdf', ['-f' num2str(f)], fullfile(resultdir, sprintf('plots_%s_%s.pdf', plotnames{f}, ovstr)));
    end
    
    % Create plots for a selection of objects and write out parts/view (6/7)
    if ~isempty(detail_subset)
      displayPerCharacteristicPlots(result(detail_subset), detname);
      for f = 6:7
        set(f, 'PaperUnits', 'inches'); set(f, 'PaperSize', [8.5 11]); set(f, 'PaperPosition', [0 11-3 8.5 2.5]);
        print('-dpdf', ['-f' num2str(f)], fullfile(resultdir, sprintf('plots_%s_%s.pdf', plotnames{f}, ovstr)));
      end    
    end
    
    % Create plots for all characteristics for each object
    displayCharacteristicPerClassPlots(result, detname);
    for f = 1:numel(objnames_extra)      
      set(f, 'PaperUnits', 'inches'); set(f, 'PaperSize', [8.5 11]); set(f, 'PaperPosition', [0 11-3 8 2.5]);
      print('-dpdf', ['-f' num2str(f)], fullfile(resultdir, sprintf('plots_%s_%s.pdf', objnames_extra{f}, ovstr)));      
    end
    
    % Display summary of sensitivity and impact
    displayAverageSensitivityImpactPlot(result, detname);
    f=1;set(f, 'PaperUnits', 'inches'); set(f, 'PaperSize', [8.5 11]); set(f, 'PaperPosition', [0 11-3 4 2.75]);
    print('-dpdf', ['-f' num2str(f)], fullfile(resultdir, sprintf('plots_%s_%s.pdf', 'impact', ovstr)));    
    
    
    % Write text file of missed object characteristics
    writeMissedObjectCharacteristics(result, detector, ...
      fullfile(resultdir, sprintf('missed_object_characteristics_%s_%s.txt', detector, ovstr)));
    
    % Save the object examples that are classified less well than predicted
    nimages = 15;
    try; mkdir(fullfile(resultdir, 'tp')); catch; end;
    displayGtConfidencePredictions(imdir, rec, result, fullfile(resultdir, 'tp'), nimages);
    
  end
  
  if DO_SHOW_SURPRISING_MISSES
    ovstr = 'ov50';
    for o = 1:numel(objnames_extra)
      load(fullfile(resultdir, sprintf('results_%s_%s.mat', objnames_selected{o}, ovstr)), 'result');                  
      showSurprisingMisses(imdir, rec, result); % for displaying unlikely misses
    end
  end
  
  %% Write summaries of false positive displays
  if DO_FP_DISPLAY
    clear result_fp det gt;
    for o = 1:numel(objnames_selected)
      objname = objnames_selected{o};
      tmp = load(fullfile(resultdir, sprintf('results_fp_%s.mat', objname)));
      if ~isfield(tmp.result_fp, 'ignoresimilar')
        tmp.result_fp.ignoresimilar.ap = -1; tmp.result_fp.ignoresimilar.apn = -1; 
        tmp.result_fp.onlysimilar.ap = -1;  tmp.result_fp.onlysimilar.apn = -1;
      end            
      result_fp(o) = orderfields(tmp.result_fp);              
      usediff = true;
      [gt.ids, gt.bbox, gt.isdiff, gt.istrunc, gt.isocc, gt.details, gt.rnum, gt.onum] = ...
        PASgetObjects(rec, objname, usediff);            
      [det.bbox, det.conf, det.rnum] = readDetections(sprintf(detpath, objname), rec);            
      det.N  =size(det.bbox, 1);
      det.nimages = numel(rec);      
      
      % display top false positives
      nimages = 20;
      try; mkdir(fullfile(resultdir, 'fp')); catch; end;
      displayTopFP(imdir, rec, result_fp(o), det, gt, fullfile(resultdir, 'fp'), nimages);          
      
    end    
    
    % plot impact summary
    if numel(objnames_selected)==numel(objnames_all)
      % sets are grouped into all, animals, vehicles, furniture    
      sets = cat(2, num2cell(1:20), {[3 8 10 12 13 17], [1 4 6 7 19 2 14], [9 11 18]});        
      setnames = cat(2, objnames_all, {'animals', 'vehicles', 'furniture'});
    else
      sets = num2cell(1:numel(objnames_selected));
      setnames = objnames_selected;
    end        
    for f = 1:numel(sets)
      displayFalsePositiveImpactPlot(result_fp(sets{f}), '', setnames{f});
      set(1, 'PaperUnits', 'inches'); set(1, 'PaperSize', [8.5 11]); set(1, 'PaperPosition', [0 11-1.5 3 1.5]);   
      print('-dpdf', '-f1', fullfile(resultdir, sprintf('plots_fp_%s.pdf', setnames{f})));
      set(2, 'PaperUnits', 'inches'); set(2, 'PaperSize', [8.5 11]); set(2, 'PaperPosition', [0 11-3 3 3]);   
      print('-dpdf', '-f2', fullfile(resultdir, sprintf('plots_fp_pie_%s.pdf', setnames{f})));            
      
      nfp = [25 50 100 200 400 800 1600 3200];
      displayFPTrend(result_fp(sets{f}), nfp, setnames{f});
      set(3, 'PaperUnits', 'inches'); set(3, 'PaperSize', [8.5 11]); set(3, 'PaperPosition', [0 11-6 6 6]);   
      print('-dpdf', '-f3', fullfile(resultdir, sprintf('plots_fp_trendarea_%s.pdf', setnames{f})));            
      set(4, 'PaperUnits', 'inches'); set(4, 'PaperSize', [8.5 11]); set(4, 'PaperPosition', [0 11-6 6 6]);   
      print('-dpdf', '-f4', fullfile(resultdir, sprintf('plots_fp_trendline_%s.pdf', setnames{f})));                   
      set(5, 'PaperUnits', 'inches'); set(5, 'PaperSize', [8.5 11]); set(5, 'PaperPosition', [0 11-6 6 6]);   
      print('-dpdf', '-f5', fullfile(resultdir, sprintf('plots_fp_trendline_nl_%s.pdf', setnames{f})));                   
      
    end   
    
    % text summary
    writeFPAnalysisSummary(result_fp, objnames_selected, detname, ...
      fullfile(resultdir, sprintf('false_positive_analysis_%s.txt', detector)));
       
  end
  
  if DO_TEX
    if ~exist(fullfile(resultdir, 'tex'), 'file'), mkdir(fullfile(resultdir, 'tex')); end;
    system(sprintf('cp ../results/*.tex %s', fullfile(resultdir, 'tex')));
    for o = 1:numel(objnames_selected)
      writeTexHeader(fullfile(resultdir, 'tex'), detname)
      usediff = false;
      [gt.ids, gt.bbox, gt.isdiff, gt.istrunc, gt.isocc, gt.details, gt.rnum, gt.onum] = ...
          PASgetObjects(rec, objnames_selected{o}, usediff);
      writeTexObject(objnames_selected{o}, fullfile(resultdir, 'tex'), gt); 
    end
  end
  
end

  
