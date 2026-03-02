clc
clear all
close all

addpath('C:\Users\wang6\Vaziri Dropbox\Yao Wang\Project_Maximum-HR\Measurements\Yao LBM\202602_StanfordImaging\32622_1_00003_00063\save_as')


% Run this code on timelapse image of a single Z

%%  1. Load Image and set important parameters.
OrigImageName = 'tp00001-00693_zplane01_stack.tif';
OrigImage = tiffreadVolume(OrigImageName); 
[imgHeight, imgWidth, totalFrames] = size(OrigImage);

minObjectSize = 4;               % Minimum pixel size to keep after segmentation
fiberThickness = 3; % Approximate neurite thickness in pixels
roiSize = fiberThickness;
activityThresholdRatio = 0.2;     % Keep ROIs with max signal >= 50% above mean
frameRate = 17.58; % in Hz
frameDuration = 1/frameRate;

tracesPerFig = 10;                % Number of stacked traces per plot window
%%  2. Calculate the Mean Image & Segment (from previous steps)
OrigImage = OrigImage - min(OrigImage(:));
meanImage = double(mean(OrigImage, 3));
meanImage = meanImage / max(meanImage(:));

%  Enhance and Binarize
Gx = [-ones(fiberThickness) 2*ones(fiberThickness) -ones(fiberThickness)]; % the defacult fiber thickness is 3
Gy = Gx';
ImageX = imfilter(meanImage,Gx,'replicate');
ImageY = imfilter(meanImage,Gy,'replicate');
ImageX(ImageX<0) = 0; % rectify
ImageY(ImageY<0) = 0;
ImageMagnitude = sqrt(ImageX.^2+ImageY.^2); % this is the enhanced image
enhancedImage = ImageMagnitude / max(ImageMagnitude(:));
% enhancedImage = fibermetric(meanImage, fiberThickness, 'StructureSensitivity', 1);
% enhancedImage = enhancedImage / max(enhancedImage(:));

figure
subplot(1,2,1)
imshow(meanImage,[])
title('mean image of timelapse')
subplot(1,2,2)
imshow(enhancedImage,[0.1,0.9])
title('Enhanced image for segmentation')
% Synchronize zoom between the two subplots
linkaxes(findobj(gcf, 'Type', 'axes'), 'xy');

% Apply global thresholding to create a binary mask
% find the top 5% pixels in the enhancedImage
% Calculate the global threshold value based on the enhanced image
globalThres = prctile(enhancedImage(:), 95);
binaryMask = enhancedImage > globalThres; 
binaryMask = bwareaopen(binaryMask, 4); % Clean up small noise

figure
subplot(1,2,1)
imshow(meanImage,[])
title('mean image of timelapse')
subplot(1,2,2)
imshow(binaryMask)
title('image segmentation with global thresholding')
linkaxes(findobj(gcf, 'Type', 'axes'), 'xy');

% % Apply local thresholding to create a binary mask
% sensitivity = 0.08; 
% T = adaptthresh(enhancedImage, sensitivity, 'ForegroundPolarity', 'bright');
% binaryMask = imbinarize(enhancedImage, T);
% binaryMask = bwareaopen(binaryMask, 4); % Clean up small noise
% 
% figure
% subplot(1,2,1)
% imshow(meanImage,[])
% title('mean image of timelapse')
% subplot(1,2,2)
% imshow(binaryMask)
% title('image segmentation with local thresholding')
% linkaxes(findobj(gcf, 'Type', 'axes'), 'xy');




%% 3. Extract Activity and Filter ROIs
disp('Extracting activity from segmented regions...');
halfRoi = floor(roiSize/2);

roiTraces = [];     % Storage: Rows = Frames, Columns = ROIs
roiLocations = [];  % Storage: Rows = ROIs, Columns = [X, Y]

% Iterate with stride to prevent overlapping analysis regions
count_checked = 0;
for y = (1+halfRoi) : roiSize : (imgHeight-halfRoi)
    for x = (1+halfRoi) : roiSize : (imgWidth-halfRoi)
        
        % Only analyze if the center pixel falls on the segmented mask
        if binaryMask(y, x) == 1
            count_checked = count_checked + 1;
            
            % Define 3x3 block indices
            y_idx = y-halfRoi : y+halfRoi;
            x_idx = x-halfRoi : x+halfRoi;
            
            % Extract tiny 3D volume and cast to double for calculation
            roiVolume = double(OrigImage(y_idx, x_idx, :));
            
            % Spatial mean to get 1D temporal trace
            trace = squeeze(mean(roiVolume, [1, 2])); 
            
            % --- Activity Filtering Logic ---
            meanF = mean(trace);
            % Smooth slightly to avoid picking up single-frame noise spikes
            smoothedTrace = movmean(trace, 3);
            maxF = max(smoothedTrace);
            
            % Check if peak activity passes threshold defined above mean
            if (maxF - meanF) / meanF >= activityThresholdRatio
                % Keep this ROI
                roiTraces = [roiTraces, trace];       % Append trace as new column
                roiLocations = [roiLocations; x, y];  % Append coords as new row
            end
        end
    end
end

numROIs = size(roiLocations, 1);
disp(['Checked ', num2str(count_checked), ' segmented regions.']);
disp(['Found ', num2str(numROIs), ' ROIs matching activity threshold.']);


%% 4. Save Data to File
if numROIs > 0
    disp('Saving data...');
    % --- Save Time Series Data to CSV ---
    framesCol = (1:totalFrames)';
    timeSeriesData = [framesCol, roiTraces];
    
    % Create headers: Frame, ROI_1, ROI_2, ...
    tsHeaders = {'FrameNumber'};
    for i = 1:numROIs
        tsHeaders{end+1} = sprintf('ROI_%d', i);
    end
    
    tsTable = array2table(timeSeriesData, 'VariableNames', tsHeaders);
    writetable(tsTable, 'SpineActivity_TimeSeries.csv');
    disp('-> Saved SpineActivity_TimeSeries.csv');
    
    % --- Save XY Locations to Excel ---
    locTable = array2table(roiLocations, 'VariableNames', {'X_Center', 'Y_Center'});
    % Adding ROI ID column for easier matching in Excel
    locTable = [array2table((1:numROIs)', 'VariableNames',{'ROI_ID'}), locTable];
    writetable(locTable, 'SpineActivity_Locations.xlsx');
    disp('-> Saved SpineActivity_Locations.xlsx');
else
    warning('No ROIs met the activity threshold. No files saved.');
end


%% 5. Visualization 1: Stacked Activity Plots
if numROIs > 0
    disp('Generating stacked activity plots...');
    timeVector = (0:totalFrames-1) * frameDuration;
    
    % Calculate vertical spacing based on global max to prevent overlap
    % Using a robust max (ignoring top 1% outliers) helps if one trace is huge
    robustMax = prctile(roiTraces(:), 99); 
    spacing = robustMax * 0.8; 
    if spacing == 0; spacing = 10; end % fallback if signal is flat
    
    numFigs = ceil(numROIs / tracesPerFig);
    
    for f = 1:numFigs
        idxStart = (f-1) * tracesPerFig + 1;
        idxEnd = min(f * tracesPerFig, numROIs);
        currentIndices = idxStart:idxEnd;
        
        figure('Color', 'w', 'Name', sprintf('Activity Traces %d-%d', idxStart, idxEnd));
        hold on;
        
        plotCount = 0;
        for i = currentIndices 
            trace = roiTraces(:, i);
            % Stack upwards: trace + (position * spacing)
            verticalOffset = plotCount * spacing;
            
            plot(timeVector, trace + verticalOffset, 'LineWidth', 1.5);
            
            % Label Y-axis near the start of the trace
            text(timeVector(1), verticalOffset + mean(trace), ...
                 sprintf('ROI %d', i), 'HorizontalAlignment', 'right', ...
                 'FontSize', 8, 'FontWeight', 'bold', 'Color', [0.2 0.2 0.2]);
             
            plotCount = plotCount + 1;
        end
        
        title(sprintf('Neurite Activity (ROIs %d - %d)', idxStart, idxEnd));
        xlabel('Time (s)');
        ylabel('Fluorescence (Arbitrary Units - Stacked)');
        yticks([]); box off; set(gca, 'FontSize', 10);
        % Optionally set consistent X-limits depending on your experiment length
        % xlim([0, timeVector(end)]); 
        hold off;
    end
end


%% 6. Visualization 2: Spatial Location Map
% This draws the ROIs on top of the mean neuron image
if numROIs > 0
    disp('Generating spatial location map...');
    figure('Name', 'Active ROI Locations Map', 'Color', 'w');
    
    % Display the mean projection image scaled appropriately
    imshow(meanImage, []); 
    hold on;
    title(['Spatial Map of ', num2str(numROIs), ' Active ROIs']);
    
    % 1. Plot markers for all ROI centers
    % Red circles ('or') with yellow edges make them pop against dark or bright backgrounds
    plot(roiLocations(:,1), roiLocations(:,2), 'or', ...
         'MarkerSize', 8, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'y', 'LineWidth', 1.5);
     
    % 2. Add numerical labels next to each marker
    for i = 1:numROIs
        x_coord = roiLocations(i, 1);
        y_coord = roiLocations(i, 2);
        
        % Place text slightly offset (+4 pixels) for readability so it doesn't cover the marker
        text(x_coord + 4, y_coord + 4, num2str(i), ...
             'Color', 'y', 'FontSize', 10, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
    end
    
    hold off;
    % Ensure axes match image dimensions exactly
    axis image; 
end

disp('Done.');