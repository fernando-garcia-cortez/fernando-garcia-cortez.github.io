# import OS module
import os
 
# Get the list of all files and directories
path = "C://Users//Fernando Garcia//Documents//fernando-garcia-cortez.github.io//Photography//231021-McCalls"
dir_list = os.listdir(path)

# prints all files
print(dir_list)
for i in dir_list:
    print("<img src=\"231021-McCalls/"+i+"\" class=\"centerImg\" style=\"width: 75%\">")
    print("<br>")