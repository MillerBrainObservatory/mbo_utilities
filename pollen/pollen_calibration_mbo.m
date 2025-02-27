function pollen_calibration_mbo(filepath, dual_cavity,z_step_um,order)
% filepath: file dialog will open to this location
% dual_cavity: 0 if single cavity, 1 if dual cavity
% z_step_um: distance between zplanes (um)
% order: order of zplanes, leave empty for original order
% pollen_calibration(filepath,0,5); single cavity, 5 um step size,
% original order (1:num_planes)

[filename, filepath] = uigetfile('*.tif', 'Select file:', filepath, 'MultiSelect', 'off');
if isequal(filename, 0)
    disp('User canceled file selection.');
    return;
end

metadata = get_metadata([filepath filename]);
fov_um_x = metadata.fov(1);
fov_um_y = metadata.fov(2);

nx = metadata.tiff_length;
ny = metadata.tiff_width;
nz = metadata.num_frames;
nt = 1;
nc = metadata.num_planes;

if nargin < 4 || isempty(order)
    order = 1:nc;
end

dx = fov_um_x/nx;
dy = fov_um_y/ny;

vol = load_or_read_data(filepath, filename, ny, nx, nc, nt, nz);

% 1. scan offset correction
vol = correct_scan_phase(vol);

% 2. user marked pollen
[xs, ys, Iz, III] = user_pollen_selection(vol);

% 3. power vs z
[ZZ, zoi, pp] = analyze_power_vs_z(Iz, filepath, z_step_um, order);

% 4. analyze z
analyze_z_positions(ZZ, zoi, order, filepath, dual_cavity)

% 5. exponential decay
fit_exp_decay(ZZ, zoi, order, filepath, dual_cavity, pp, z_step_um)

% X, Y calibration
calibrate_xy(xs, ys, III, filepath, dual_cavity,nx,ny,dx,dy);
end

%% input data handling
function vol = load_or_read_data(filepath, filename, ny, nx, nc, nt, nz)
import ScanImageTiffReader.*

fname = filename(1:end-4);
if exist([filepath fname '.mat'], 'file') < 2
    disp('Loading TIFF Data...');
    vol = ScanImageTiffReader([filepath fname '.tif']).data();
    vol = reshape(vol, ny, nx, nc, nt, nz);
    vol = vol - mean(vol(:));  % Normalize
    vol = mean(vol, 4);
    vol = reshape(vol, ny, nx, nc, nz);
    save([filepath fname '.mat'], 'vol', '-v7.3');
else
    disp('Loading Preprocessed Data...');
    load([filepath fname '.mat'], 'vol');
end
end

%% scan offset correction
function vol = correct_scan_phase(vol)
dim = 2;
disp('Planes shifted. Detecting scan offset...');

Iinit = max(vol, [], 4);
scan_corrections = zeros(1, size(vol, 3));

for ijk = 1:size(vol, 3)
    scan_corrections(ijk) = returnScanOffset2(Iinit(:,:,ijk), dim);
end

disp('Offsets returned. Correcting scan phase...');
for ijk = 1:size(vol, 3)
    disp(['Correcting plane ' num2str(ijk) '...']);
    POI = vol(:,:,ijk,:);
    POI = fixScanPhase(POI, scan_corrections(ijk), dim);
    vol(:,:,ijk,:) = POI;
end
end

%% user select beads
function [xs, ys, Iz, III] = user_pollen_selection(vol)
nc = size(vol, 3);
nz = size(vol, 4);
num = 10;

xs = zeros(1, nc);
ys = zeros(1, nc);
Iz = zeros(nc, nz);
III = zeros(2*num+1, 2*num+1, nc);

disp('Select pollen beads...');
for kk = 1:nc
    figure(901);
    imagesc(max(vol(:,:,kk,:), [], 4));
    axis image; colormap(gray);
    set(gca, 'xtick', [], 'ytick', []);
    title(['Select pollen bead for beamlet ' num2str(kk)]);
    drawnow;

    [x, y] = ginput(1);
    indx = round(x); indy = round(y);

    Iz(kk, :) = reshape(max(max(movmean(movmean(vol(indy-num:indy+num,indx-num:indx+num,kk,:), 3, 1), 3, 2))), 1, []);
    [~, zoi] = max(movmean(Iz(kk,:), 10./5, 2));

    xs(kk) = x;
    ys(kk) = y;
    III(:,:,kk) = reshape(vol(indy-num:indy+num, indx-num:indx+num, kk, zoi), num*2+1, num*2+1);
end
close("gcf");
end

%% power vs z
function [ZZ, zoi, pp] = analyze_power_vs_z(Iz, filepath, DZ, order)
disp('Running power vs z')
amt = 10./DZ;
nz = size(Iz, 2);
ZZ = fliplr((0:(nz-1)) * DZ);

f99 = figure(77);
f99.OuterPosition = [670,800,570,510];

plot(ZZ, sqrt(movmean(Iz(order, :), amt, 2)), 'LineWidth', 1.5);
grid(gca, 'on');
xlabel('Piezo Z (\mum)');
ylabel('2p signal (a.u.)');
title('Power vs. Z-depth');

zoi = zeros(1, length(order));
for ii = 1:length(order)
    [~, zoi(ii)] = max(movmean(Iz(order(ii),:), amt, 2), [], 2);
end

pp = max(movmean(Iz, amt, 2), [], 2);

hold on;
plot(ZZ(zoi), sqrt(pp), 'k.', 'MarkerSize', 10);

for i = 1:numel(zoi)
    text(ZZ(zoi(i)), sqrt(pp(i)) + range(ylim) * 0.02, num2str(find(order == i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end

saveas(f99, [filepath 'pollen_calibration_pollen_signal_vs_z.fig']);
hold off;
end

%% z-positions
function analyze_z_positions(ZZ, zoi, order, filepath, dual_cavity)
disp('anayzing z-positions')
Z0 = ZZ(zoi(order(1)));
figure;
hold on;

if dual_cavity
    nc_half = length(order) / 2;

    plot(1:nc_half, ZZ(zoi(order(1:nc_half))) - Z0, 'bo', 'MarkerSize', 6);

    plot(nc_half+1:length(order), ZZ(zoi(order(nc_half+1:end))) - Z0, ...
        'gsquare', 'Color', [0 0.5 0], 'MarkerSize', 6);

    [ft2, goodness, ~] = fit((1:length(order))', ZZ(zoi(order))' - Z0, 'poly1');
    bb = linspace(0, length(order) + 1, 101);
    plot(bb, ft2(bb), 'k-');

    legend('Data Cavity A', 'Data Cavity B', ...
        ['Linear fit (r² = ' num2str(goodness.rsquare, 3) ')'], ...
        'Location', 'NorthEast');

else
    % single cavity - all beamlets in one plot
    plot(1:length(order), ZZ(zoi(order)) - Z0, 'bo', 'MarkerSize', 6);

    [ft2, goodness, ~] = fit((1:length(order))', ZZ(zoi(order))' - Z0, 'poly1');
    bb = linspace(0, length(order) + 1, 101);
    plot(bb, ft2(bb), 'k-');

    legend('Beamlet Z positions', ...
        ['Linear fit (r² = ' num2str(goodness.rsquare, 3) ')'], ...
        'Location', 'NorthEast');
end

% Formatting
xlabel('Beam number');
% ylabel('Z position (/mum)');
ylabel('Z position (\mum)');
grid(gca, 'on');

% Save figure
saveas(gcf, [filepath 'pollen_calibration_z_vs_N.fig']);

hold off;
end

%%
function fit_exp_decay(ZZ, zoi, order, filepath, dual_cavity, pp, DZ)

if ~dual_cavity
    z1 = ZZ(zoi(order(1:length(order))));
    p1 = sqrt(pp(order(1:length(order))));
    figure;
    plot(z1, p1, 'bo', 'MarkerSize', 6);
    xlabel('Z (\mum)');
    ylabel('Power (a.u.)');
    grid(gca, 'on');

    [ft1, g1] = fit(z1', p1, 'exp1');

    % Plot the fit
    hold on;
    plot(DZ * linspace(0, length(ZZ)-1, 1001), ft1(DZ * linspace(0, length(ZZ)-1, 1001)), 'r-');
    legend(['Fit (l_s = ' num2str(1/ft1.b, 3) ' \mum)'], 'Location', 'NorthWest');

    saveas(gcf, [filepath 'pollen_calibration_power_linear.fig']);
    hold off;

    % Dual cavity case (separate fits)
else
    z1 = ZZ(zoi(order(1:length(order)/2)));
    p1 = sqrt(pp(order(1:length(order)/2)));
    z2 = ZZ(zoi(order(length(order)/2+1:end)));
    p2 = sqrt(pp(order(length(order)/2+1:end)));

    figure;
    plot(z1, p1, 'bo', 'MarkerSize', 6);
    hold on;
    plot(z2, p2, 'bsquare', 'Color', [0 0.5 0], 'MarkerSize', 6);
    xlabel('Z (\mum)');
    ylabel('Power (a.u.)');
    grid(gca, 'on');

    [ft1, g1] = fit(z1', p1, 'exp1');
    [ft2, g2] = fit(z2', p2, 'exp1');

    plot(DZ * linspace(0, length(ZZ)-1, 1001), ft1(DZ * linspace(0, length(ZZ)-1, 1001)), 'r-');
    plot(DZ * linspace(0, length(ZZ)-1, 1001), ft2(DZ * linspace(0, length(ZZ)-1, 1001)), 'k-');

    legend('Data Cavity A', 'Data Cavity B', ...
        ['Fit C1 (l_s = ' num2str(1/ft1.b, 3) ' \mum)'], ...
        ['Fit C2 (l_s = ' num2str(1/ft2.b, 3) ' \mum)'], 'Location', 'NorthWest');

    saveas(gcf, [filepath 'pollen_calibration_power_linear.fig']);
    hold off;
end
end

%% x y offsets
function calibrate_xy(xs, ys, III, filepath, dual_cavity, nx, ny, dx, dy)

nc_total = size(III, 3);  % Total number of beamlets
if dual_cavity
    nc = nc_total / 2;
else
    nc = nc_total;
end

vx = (-floor(nx/2):floor(nx/2))*dx;
vy = (-floor(ny/2):floor(ny/2))*dy;

figure;
plot(vx(round(xs(1:nc))), vy(round(ys(1:nc))), 'bo', 'MarkerSize', 6);
hold on;

if dual_cavity && length(xs) >= nc+1
    plot(vx(round(xs(nc+1:end))), vy(round(ys(nc+1:end))), 'bsquare', 'Color', [0 0.5 0], 'MarkerSize', 6);
end

xlabel('X (\mum)');
ylabel('Y (\mum)');
grid(gca, 'on');
axis equal;
if dual_cavity
    legend('Beamlets Cavity A', 'Beamlets Cavity B', 'Location', 'NorthEast');
else
    legend('Beamlets (single cavity)', 'Location', 'NorthEast');
end
saveas(gcf, [filepath 'pollen_calibration_x_y_offsets.fig']);
end

%% find scan offset
function correction = returnScanOffset2(Iin, dim)
if numel(size(Iin)) > 2
    Iin = mean(Iin, 3);
end

n = 8;
if dim == 2
    Iv1 = Iin(:, 1:2:end);
    Iv2 = Iin(:, 2:2:end);
    Iv1 = Iv1(:, 1:min(size(Iv1,2), size(Iv2,2)));
    Iv2 = Iv2(:, 1:min(size(Iv1,2), size(Iv2,2)));

    buffers = zeros(n, size(Iv1,2));
    Iv1 = cat(1, buffers, Iv1, buffers);
    Iv2 = cat(1, buffers, Iv2, buffers);

    Iv1 = Iv1(:) - mean(Iv1(:));
    Iv2 = Iv2(:) - mean(Iv2(:));
    Iv1(Iv1 < 0) = 0;
    Iv2(Iv2 < 0) = 0;

    [r, lag] = xcorr(Iv1, Iv2, n, 'unbiased');
    [~, ind] = max(r);
    correction = lag(ind);
else
    correction = 0;
end
end

%% fix scan offset
function dataOut = fixScanPhase(dataIn, offset, dim)
[sy, sx, sc, sz] = size(dataIn);
dataOut = zeros(sy, sx, sc, sz);

if dim == 2
    if offset > 0
        dataOut(1+offset:end, :, :, :) = dataIn(1:end-offset, :, :, :);
    elseif offset < 0
        offset = abs(offset);
        dataOut(1:end-offset, :, :, :) = dataIn(1+offset:end, :, :, :);
    else
        dataOut = dataIn;
    end
end
end

%% metadata from file
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

% TIFF header data for additional metadata
% getHeaderData() is a ScanImage utility that iterates through every

[header, desc] = scanimage.util.private.getHeaderData(hTiff);
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
