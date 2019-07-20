import csv
from collections import defaultdict
rows = csv.DictReader(open("Train2.csv", "r"))
writer = csv.writer(open("new.csv", "w"))
newrows = defaultdict()
for row in rows:
	x=row['Id']
	newrows[x]=newrows[x]+1
	if newrows[row['Id']]==1:
		writer.writerow(row)
	else:
		print ("duplicate",row['Id'])

writer.writerows(newrows)
