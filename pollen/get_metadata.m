function [metadata_out] = get_metadata(filename)
% Extract metadata from a ScanImage TIFF file.
%
% Read and parse Tiff metadata stored in the .tiff header
% and ScanImage metadata stored in the 'Artist' tag which contains roi sizes/locations and scanning configuration
% details in a JSON format.
%
% Parameters
% ----------
% filename : char
%     The full path to the TIFF file from which metadata will be extracted.
%
% Returns
% -------
% metadata_out : struct
%     A struct containing metadata such as center and size of the scan field,
%     pixel resolution, image dimensions, number of frames, frame rate, and
%     additional roi data extracted from the TIFF file.
%
% Examples
% --------
% metadata = get_metadata("path/to/file.tif");
%

hTiff = Tiff(filename);
[fpath, fname, ~] = fileparts(filename);

% Metadata in JSON format stored by ScanImage in the 'Artist' tag
roistr = hTiff.getTag('Artist');
roistr(roistr == 0) = []; % Remove null termination from string
mdata = jsondecode(roistr); % Decode JSON string to structure
mdata = mdata.RoiGroups.imagingRoiGroup.rois; % Pull out a single roi, assumes they will always be the same
num_rois = length(mdata); % only accurate way to determine the number of ROI's
scanfields = mdata.scanfields;

% roi (scanfield) metadata, gives us pixel sizes
center_xy = scanfields.centerXY;
size_xy = scanfields.sizeXY;
num_pixel_xy = scanfields.pixelResolutionXY; % misleading name
tic;

% TIFF header data for additional metadata
% getHeaderData() is a ScanImage utility that iterates through every

[header, desc] = scanimage.util.private.getHeaderData(hTiff);
toc
sample_format = hTiff.getTag('SampleFormat'); % raw data type, scanimage uses int16

switch sample_format
    case 1
        sample_format = 'uint16';
    case 2
        sample_format = 'int16';
otherwise
    error('Invalid image datatype')
end

% Needed to preallocate the raw images
tiff_length = hTiff.getTag("ImageLength");
tiff_width = hTiff.getTag("ImageWidth");

% .. deprecated:: v1.8.0
%
%   hStackManager.framesPerSlice - only works for slow-stack aquisition
%   hScan2D.logFramesPerFile - this only logs multi-file recordings,
%   otherwise is set to 'Inf', which isn't useful for the primary use
%   case of this variable that is preallocating an array to fill this image
%   data
%
%   num_frames_total = header.SI.hStackManager.framesPerSlice; % the total number of frames for this imaging session
%   num_frames_file = header.SI.hScan2D.logFramesPerFile; % integer, for split files only: how many images per file to capture before rolling over a new file.

num_planes = length(header.SI.hChannels.channelSave); % an array of active channels: channels are where information from each light bead is stored
num_frames = numel(desc) / num_planes;

% .. deprecated:: v1.3.x
%
% hRoiManager.linesPerFrame - not captured for multi-roi recordings
% lines_per_frame = header.SI.hRoiManager.linesPerFrame; % essentially gives our "raw roi width"

num_lines_between_scanfields = round(header.SI.hScan2D.flytoTimePerScanfield / header.SI.hRoiManager.linePeriod);
% uniform_sampling = header.SI.hScan2D.uniformSampling;

% Calculate using frame rate and field-of-view
line_period = header.SI.hRoiManager.linePeriod;
scan_frame_period = header.SI.hRoiManager.scanFramePeriod;
frame_rate = header.SI.hRoiManager.scanVolumeRate;
objective_resolution = header.SI.objectiveResolution;

fovx = round(objective_resolution * size_xy(1) * num_rois); % account for the x extent being a single roi
fovy = round(objective_resolution * size_xy(2));
fov_xy = [fovx fovy];

fov_roi = round(objective_resolution * size_xy); % account for the x extent being a single roi
pixel_resolution = mean(fov_roi ./ num_pixel_xy);

% Number of pixels in X and Y
roi_width_px = num_pixel_xy(1);
roi_height_px = num_pixel_xy(2);

metadata_out = struct( ...
    'num_planes', num_planes, ...
    'num_rois', num_rois, ...
    'num_frames', num_frames, ...
    'frame_rate', frame_rate, ...
    'fov', fov_xy, ...  % in micron
    'pixel_resolution', pixel_resolution, ...
    'sample_format', sample_format, ...
    'roi_width_px', roi_width_px, ...
    'roi_height_px', roi_height_px,  ...
    'tiff_length', tiff_length, ...
    'tiff_width', tiff_width, ...
    'raw_filename', fname, ...
    'raw_filepath', fpath, ...
    'raw_fullfile', filename, ...
    ... %% used internally 
    'num_lines_between_scanfields', num_lines_between_scanfields, ...
    'center_xy', center_xy, ...
    'line_period', line_period, ...
    'scan_frame_period', scan_frame_period, ...
    'size_xy', size_xy, ...
    'objective_resolution', objective_resolution ...
    );

end
