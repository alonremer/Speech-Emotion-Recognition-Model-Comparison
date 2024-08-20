
Crema_path = '/content/drive/My Drive/deep_learn_project/AudioWAV/' # save the files into your project environment
Crema_dir_list = os.listdir(Crema_path)

emotions_crema = []
paths_crema = []

for it in Crema_dir_list:
    # Storing file paths
    paths_crema.append(Crema_path + it)
    # Storing file emotions
    part = it.split('_')
    if part[2] == 'SAD':
        emotions_crema.append('sad')
    elif part[2] == 'ANG':
        emotions_crema.append('angry')
    elif part[2] == 'DIS':
        emotions_crema.append('disgust')
    elif part[2] == 'FEA':
        emotions_crema.append('fear')
    elif part[2] == 'HAP':
        emotions_crema.append('happy')
    elif part[2] == 'NEU':
        emotions_crema.append('neutral')
    else:
        emotions_crema.append('Unknown')

# Dataframes for emotion of files and paths
emotions_crema_df = pd.DataFrame(emotions_crema, columns=['Emotions'])
path_crema_df = pd.DataFrame(paths_crema, columns=['Path'])
Crema_df = pd.concat([emotions_crema_df, path_crema_df], axis=1)

# Map emotion labels to integer indices
emotion_labels = Crema_df['Emotions'].unique()
emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_labels)}
