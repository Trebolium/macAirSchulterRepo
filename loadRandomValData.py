half_window=int(params['sample_frame_length']/2)


for song_index in range(int(params['songs_to_validate'])):
	song_num_frames = hdf5_file['val_lengths'][song_index, ...]
	max_index=song_num_frames-half_window
	next_song=False
    # this for loop repeats after each batch is complete - hence the num_steps reference
	for sample_index in range(half_window+1, max_index-1, params['skip_size']):
        x_data=[]
        y=[]
        if sample_index<(song_num_frames-int(params['sample_frame_length']/2)-1):
            next_song=True
            break
        else:
            feature = hdf5_file['val_features'][song_index, ...]
            # find how many samples are in this song by looking up lengths
            # for k in range(int(params['sample_frame_length']/2)+1,song_num_frames-int(params['sample_frame_length']/2)-1):
            sample_excerpt = feature[:,sample_index-int(params['sample_frame_length']/2):sample_index+int(params['sample_frame_length']/2)+1]
            frame_time = sample_index*params['hop_length']/params['fs']
            label_points=hdf5_file['val_labels'][song_index, ...]

            previous_value=-1
            for row in range(500):
                # make sure the index doesn't accidently go into the padded zeros section
                if label_points[row][0]>previous_value:
                    if label_points[row][0]>frame_time:
                        # go back one and get label, third element holds the label
                        y.append(label_points[row-1][2])
                        # don't search any more rows
                        break
                    else:
                        previous_value=label_points[row][0]
                else:
                    y.append(label_points[row-1][2])
                    # don't search any more rows
                    break
    if next_song==True:
        break
    x_data = np.asarray(x_data)
    x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
    y = np.asarray(y)
    yield x_data, y
