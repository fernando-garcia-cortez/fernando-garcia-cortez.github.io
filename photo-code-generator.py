# import OS module
import os
 
# Get the list of all files and directories
path = "C://Users//Fernando Garcia//Documents//fernando-garcia-cortez.github.io//Photography//240816-AnalogRoll1"
dir_list = os.listdir(path)

# prints all files
print(dir_list)
for i in dir_list:
    print("<img src=\"240816-AnalogRoll1/"+i+"\" class=\"centerImg\" style=\"width: 75%\">")
    print("<br>")