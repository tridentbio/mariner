import sys
emails = str(sys.argv[1])
print(emails)
with open('secrets-example.yaml', 'r') as file :
  filedata = file.read()
filedata = filedata.replace('#EMAILS', emails)
with open('secrets-example.yaml', 'w') as file:
  file.write(filedata)