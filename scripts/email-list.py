import sys
 
emails = str(sys.argv[1])
print(emails)
emails_fixed = emails.replace('\n', '')
print(emails_fixed)
with open('secrets-example.yaml', 'r') as file :
  filedata = file.read()
filedata = filedata.replace('#EMAILS', emails_fixed)
with open('secrets-example.yaml', 'w') as file:
  file.write(filedata)