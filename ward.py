import requests
from bs4 import BeautifulSoup

url = requests.get("https://en.wikipedia.org/wiki/List_of_wards_in_Bangalore").text
soup = BeautifulSoup(url,'lxml')
#print(soup.prettify())
table = soup.find("table",{'class':'wikitable'})
print(table)