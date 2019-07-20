import csv
fi= open('Train.csv', "rt")
fo=open("Train2.csv",'w')
reader=csv.reader(fi)
writer=csv.writer(fo)
for i, row in enumerate(reader):
	if(i<1000000):
		writer.writerow(row)
		print (i)
	else:
		break
		
print ("Extracted 10 Lakh Docs")
