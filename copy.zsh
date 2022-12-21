# create a file to iterate through all files in ./curriculum and print
# the file name and file contents to the end of a .txt file called code.txt
# in the same directory
# There should be visual separation between each file's contents
# in the code.txt file

# clear old copied files
echo "" > code.txt
echo "" > copied_files.txt

for file in $(find ~/Library/CloudStorage/OneDrive-City,UniversityofLondon/modules/project/curriculum \
 -name '*.py' \
 ! -name '*checkpoint.py*'); do
#  ignore checkpoint files
  # copy all .py files to copied_files.txt
  echo ${file#~/Library/CloudStorage/OneDrive-City,UniversityofLondon/modules/project/curriculum*} >> code.txt
  echo "##############" >> code.txt
  echo " " >> code.txt

  cat $file >> code.txt
  echo " " >> code.txt
  echo "##############" >> code.txt
  echo " " >> code.txt

  # print just file names to copied_files.txt to check what has been copied
  echo ${file#~/Library/CloudStorage/OneDrive-City,UniversityofLondon/modules/project/curriculum*} >> copied_files.txt
  echo " " >> copied_files.txt
done