 curl -X 'POST' \
  'http://localhost/api/v1/datasets/' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE2NTQwNDQyMjUsInN1YiI6IjEifQ.lG1drsvCOyEsYrLJf3Ar8MkDynzaYwFSHmnr9OnNi7s' \
  -H 'Content-Type: multipart/form-data' \
  -F 'name=asdasd' \
  -F 'description=asdasd' \
  -F 'splitTarget=60-20-20' \
  -F 'splitType=scaffold' \
  -F 'columnsDescriptions=[{"pattern": ".*", "description": "la la la"}]' \
  -F 'file=@app/app/tests/data/Lipophilicity.csv ;type=text/csv'
