import os
dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path.split('\\')[0] == 'D:':
    datasources_path = dir_path+"\datasources\\"
    enrichment_path = dir_path+"\enrichment\\"
    pickles_path = dir_path+"\pickles\\"
    learning_models_path = dir_path+"\learning_models\\"
    temp_files_path = dir_path+"\\tmp\\"
else:
    datasources_path = dir_path+"/datasources/"
    enrichment_path = dir_path+"/enrichment/"
    pickles_path = dir_path+"/pickles/"
    learning_models_path = dir_path+"/learning_models/"
    temp_files_path = dir_path+"/tmp/"
