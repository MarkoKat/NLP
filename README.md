# NLP - Klasifikacija kratkih sporočil - IMapBook

V tem repozitoriju so vse datoteke za seminarsko nalogo pri predmetu Obdelava naravnega jezika na temo klasifikacije sporočil IMapBook.

Opis zbirke podatkov in uporabljenih metod je v poročilu, ki se nahaja v datoteki 
`report/NLP_Zago_Katrasnik.pdf`.

Erica Zago in Marko Katrašnik

## Uporabljene knjižnice

Vse uporabljene knjižnice so navedene v `code/requirements.txt`.

## Zagon

### Izhodiščni algoritem (baseline)

Koda za izhodiščni algoritem je v datoteki `code/baseline.py`, zažene se kot običajna Python 
datoteka - `> python baseline.py`.

Parametri se nastavijo preko spemenljivk v kodi v main funkciji:
* `sheet`: nastavi se lahko na vrednosti 'crew' ali 'discussion' in določi, kateri zavihek
Excel datoteke se bo uporabil.
* `use_response_similarity`: True ali False - ali želimo, da se pri klasifikaciji uporabijo 
tudi podobnosti sporočil s skupnimi končnimi odgovori. (Uporabi se lahko samo s sporočili
iz zavihka CREW)
* `use_book_similarity`: True ali False -  ali želimo, da se pri klasifikaciji uporabijo 
tudi podobnosti sporočil z vsebinami knjig.

### Analiza podatkov

Datoteki `similarity_analyse_crew` in `similarity_analyse_discussion` vsebujeta kodo za 
izračun podobnosti med sporočili in skupnimi končnimi odgovori, ter vsebinami knjig. Zaženeta 
se kot običajni Python datoteki.
