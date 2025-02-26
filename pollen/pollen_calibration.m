function pollen_calibration(filepath, dual_cavity, nx, ny, nc, nt, nz, fov_um, zoom, order, DZ)
    % filepath = 'D:\W2_DATA\kbarber\2025-02-17\';
    % dual_cavity = 0;
    % fov_um = 1418;
    % ny = 224;
    % nx = 224;
    % nc = 14;  
    % nt = 1;
    % nz = 81;
    % zoom = 2;
    % z_step_um = 5;
    % order = (1:nc);
    % pollen_calibration(filepath,dual_cavity,nx,ny,nc,nt,nz,fov_um,zoom,order,z_step_um);
    clc;
    
    [filename, filepath] = uigetfile('*.tif', 'Select file:', filepath, 'MultiSelect', 'off');
    if isequal(filename, 0)
        disp('User canceled file selection.');
        return;
    end
    filename = filename(1:end-4);

    dx = fov_um/zoom/nx;
    dy = fov_um/zoom/ny;
    
    vol = load_or_read_data(filepath, filename, ny, nx, nc, nt, nz);
    
    % 1. scan offset correction
    vol = correct_scan_phase(vol);
    
    % 2. user marked pollen
    [xs, ys, Iz, III] = user_pollen_selection(vol);
    
    % 3. power vs z
    [ZZ, zoi] = analyze_power_vs_z(Iz, filepath, DZ, order);

    % 4. analyze z
    analyze_z_positions(ZZ, zoi, order, filepath, dual_cavity)
    
    % X, Y calibration
    calibrate_xy(xs, ys, III, filepath, dual_cavity,nx,ny,dx,dy);
end

%% input data handling
function vol = load_or_read_data(filepath, filename, ny, nx, nc, nt, nz)
    if exist([filepath filename '.mat'], 'file') < 2
        disp('Loading TIFF Data...');
        vol = ScanImageTiffReader([filepath filename '.tif']).data();
        vol = reshape(vol, ny, nx, nc, nt, nz);
        vol = vol - mean(vol(:));  % Normalize
        vol = mean(vol, 4);
        vol = reshape(vol, ny, nx, nc, nz);
        save([filepath filename '.mat'], 'vol', '-v7.3');
    else
        disp('Loading Preprocessed Data...');
        load([filepath filename '.mat'], 'vol');
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
end

%% power vs z
function [ZZ, zoi] = analyze_power_vs_z(Iz, filepath, DZ, order)
    amt = 10./DZ;
    nz = size(Iz, 2); 
    ZZ = fliplr((0:(nz-1)) * DZ);  
    
    f99 = figure(77);
    f99.OuterPosition = [670,800,570,510];

    plot(ZZ, sqrt(movmean(Iz(order, :), amt, 2)), 'LineWidth', 1.5);
    grid(gca, 'on');
    xlabel('Piezo Z (\mum)');
    ylabel('2p signal (a.u.)');
    title('Power vs. Z-depth for Pollen Calibration');

    zoi = zeros(1, length(order));
    for ii = 1:length(order)
        [~, zoi(ii)] = max(movmean(Iz(order(ii),:), amt, 2), [], 2);
    end
    
    pp = max(movmean(Iz, amt, 2), [], 2);

    hold on;
    plot(ZZ(zoi), sqrt(pp), 'k.', 'MarkerSize', 10);
    
    for i = 1:numel(zoi)
        text(ZZ(zoi(i)), sqrt(pp(i)) + min(sqrt(pp))/2, num2str(find(order == i)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    end
    
    % Save figure
    saveas(f99, [filepath 'pollen_calibration_pollen_signal_vs_z.fig']);
    hold off;
end

function analyze_z_positions(ZZ, zoi, order, filepath, dual_cavity)
    Z0 = ZZ(zoi(order(1)));

    figure;
    hold on;

    plot(1:(length(order)/2), ZZ(zoi(order(1:length(order)/2)))-Z0, 'bo');

    if dual_cavity
        plot((length(order)/2)+1:length(order), ZZ(zoi(order((length(order)/2)+1:end)))-Z0, 'gsquare', 'Color', [0 0.5 0], 'MarkerSize', 6);
    end

    [ft2, goodness, ~] = fit((1:length(order))', ZZ(zoi(order))'-Z0, 'poly1');
    bb = linspace(0, length(order)+1, 101);
    plot(bb, ft2(bb), 'k-');

    xlabel('Beam number');
    ylabel('Z position (\mum)');
    legend('Data Cavity A', 'Data Cavity B', ['Linear fit (r² = ' num2str(goodness.rsquare, 3) ')'], 'Location', 'NorthEast');

    saveas(gcf, [filepath 'pollen_calibration_z_vs_N.fig']);
    hold off;
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
        legend('Data Cavity A', 'Data Cavity B', 'Location', 'NorthEast');
    else
        legend('Data (single cavity)', 'Location', 'NorthEast');
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
