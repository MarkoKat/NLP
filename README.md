# NLP - Klasifikacija kratkih sporočil - IMapBook

V tem repozitoriju so vse datoteke za seminarsko nalogo pri predmetu Obdelava naravnega jezika na temo klasifikacije sporočil IMapBook.

Opis zbirke podatkov in uporabljenih metod je v poročilu, ki se nahaja v datoteki 
`report/NLP_Zago_Katrasnik.pdf`.

Erica Zago in Marko Katrašnik

## Uporabljene knjižnice

Vse uporabljene knjižnice so navedene v `code/requirements.txt`.

## Zagon

Parametri, ki se uporabljajo pri vseh štirih algoritmih:
- `crew` ali `discussion`: uporaba sporočil iz zavihka CREW ali Discussion only
- `use_book_similarity`: uporaba podobnosti sporočil z besedilom knjige
- `use_response_similarity`: uporaba podobnosti sporočil s skupnim končnim odgovorom (uporaba mogoča samo v kombinaciji s poročili iz zavihka CREW)
- `use_bert_for_similarity`: uporaba BERT vektorskih vložitev za računanje podobnosti besedil. Provzeto se uporabijo TF-IDF vektorji.

Pripravila sva Google Colab beležnico v kateri so že pripravljeni vsi koraki za zagon kode za evalvacijo algoritmov: [LINK]

### Ročne značilke + MLP

Dodatni parameter:
- `reduced_feature_set`: zmanjšan nabor značilk (brez upoštevanja prisotnosti emotikono in klicajev) s katerim je bil dosežen
boljši rezultat za sporočila iz zavihka Discussion only.

Zagon brez uporabe podobnosti:

```
python manual_features.py <crew/discussion>
```

Primer zagona z uporabo podobnosti:

```
python manual_features.py discussion use_book_similarity use_bert_for_similarity
```

### TF-IDF vektorji + MLP

Zagon brez uporabe podobnosti:

```
python tf_idf.py <crew/discussion>
```

### BERT

Dodatni paramater:
- Prvi paramater mora biti ime modela - v primeru že pripravljenih modelov `BERT_crew_13.model` ali `BERT_discussion_6.model`

Povezave do modelov:
- CREW:
- Discussion:

Zagon brez uporabe podobnosti:

```
python BERT.py <ime_modela> <crew/discussion>
```

Primer:
```
python BERT.py BERT_crew_13.model crew
```

### DistilBERT

Dodatni paramater:
- Prvi paramater mora biti ime modela - v primeru že pripravljenih modelov `distill_bert_crew_14.bin` ali `distill_bert_discussion_10.bin`

Povezave do modelov:
- CREW:
- Discussion:

```
python DistillBERT.py <ime_modela> <crew/discussion>
```

Primer:
```
python DistillBERT.py distill_bert_crew_14.bin crew
```

### Analiza podatkov

Datoteki `similarity_analyse_crew` in `similarity_analyse_discussion` vsebujeta kodo za 
izračun podobnosti med sporočili in skupnimi končnimi odgovori, ter vsebinami knjig. Zaženeta 
se kot običajni Python datoteki.
