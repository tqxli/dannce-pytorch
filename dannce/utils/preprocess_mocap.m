%% Save preprocessed projected mocap data and associated frame labels

% path to calibration file (in Jesse format)
calibration_data = load('./Camera_calib_rd5/CameraCalibrations/20181101_calibration_hires_2_ESu2prime_lruprimeprime/worldcoordinates_lframe.mat');

% path to mocap data
mocap_data = load('./Camera_calib_rd5/JDM52/20181102/Preprocessed/nolj_Recording_caff_esu2prime_recording1_nolj.mat');

% path where data will be saved
basedir = ['/home/twd/Dropbox/mocapdata_for_tim/Camera_calib_rd5/JDM52/20181102/Preprocessed/'];

%% Assign NaN values to missing markers
mocap_data = correct_mocap(mocap_data);
%% define the mapping between cameras, choose a camera

%20180726 calibration is  R, L, E U S U2
matched_frames_ind_permute = [3,2,4,1,5,6]; %real is R, L, E U S U2, so 3 2 4 1 5 6

%% Save the data structure for use by DANNCE
for camerause = [1,2,3,4,5,6]
    r = rotationMatrix{camerause};
    t = translationVector{camerause};
    
    frame_synch = 1:10:length(mocap_data.matched_frames_aligned{matched_frames_ind_permute(camerause)});
    frame_inds = frame_synch;
    
    num_markers = numel(mocap_data.markernames);
    
    data_sampleID = zeros(length(frame_inds),1);
    data_frame = zeros(length(frame_inds),1);
    data_2d = zeros(length(frame_inds),2*num_markers);
    data_3d = zeros(length(frame_inds),3*num_markers);
    cnt = 1;
    
    for frame_to_plot = frame_inds
        
        thisinds = frame_to_plot;
        
        proj = zeros(0,3);
        for ll =1:numel(mocap_data.markernames)
            proj = cat(1,proj,mocap_data.markers_preproc.(mocap_data.markernames{ll})(thisinds,:));
        end
        
        imagePoints = worldToImage(calibration_data.params_individual{camerause},r,...
            t,proj,'ApplyDistortion',true);
        
        thisdata_3d = zeros(length(thisinds),3*num_markers);
        for d = 1:num_markers
            thisdata_3d(:,(d-1)*3+1:d*3) = proj((d-1)*length(thisinds)+1:d*length(thisinds),:);
        end
        % This nanmean is deprecated, size(thisdata_3d,1) should be == 1
        data_avg_3d = nanmean(thisdata_3d,1);
        
        thisdata_2d = zeros(length(thisinds),2*num_markers);
        for d = 1:num_markers
            thisdata_2d(:,(d-1)*2+1:d*2) = imagePoints((d-1)*length(thisinds)+1:d*length(thisinds),:);
        end
        
        data_avg_2d = nanmean(thisdata_2d,1);
        
        data_sampleID(cnt) = frame_to_plot;
        
        data_frame(cnt) = mocap_data.matched_frames_aligned{matched_frames_ind_permute(camerause)}(frame_to_plot)-1;
        
        data_2d(cnt,:) = data_avg_2d;
        data_3d(cnt,:) = data_avg_3d;
        
        cnt = cnt +1;
        
    end
    
    data_frame(data_frame<0) = 0;
    
    save([basedir 'cam' num2str(camerause) '_data'],'data_frame','data_2d','data_3d','data_sampleID');
    
end