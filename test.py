mk_file_list= ['a','a','a','a','a','a','a','a','a','a','a']

for i in len(mk_file_list):
    mk_file_list[i] = "https://aizostorage.blob.core.windows.net/aizo-cropped/"+ mk_file_list[i]
    

print(mk_file_list)