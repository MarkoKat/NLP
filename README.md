# NLP - Klasifikacija kratkih sporočil - IMapBook

V tem repozitoriju so vse datoteke za seminarsko nalogo pri predmetu Obdelava naravnega jezika na temo klasifikacije 
sporočil, ki so nastala z uporabo aplikacije IMapBook. Sporočila so pošiljali učenci, ki so sodelovali v knjižnih
krožkih. Njihova naloga je bila, da najprej preberejo knjigo, nato pa kot skupina odgovorijo na zastavljeno vprašanje.
O tem kako bodo odgovorili, so se pogovarjali s pošiljanjem sporočil v aplikaciji.

V zbirki ima vsako sporočilo določen razred, kot na primer "content discussion", "logistics", "general comment", "response",
"emoticon/non-verbal",...

Za klasifikacijo sva preizkusila štiri metode:
- Nabor ročno pridobljenih značilk v kombinaciji z MLP (Multi Layer Perceptron) klasifikatorjem
- TF-IDF vektorji v kombinaciji z MLP klasifikatorjem
- BERT
- DistilBERT (prečiščena različica modela BERT - manjši modeli in hitrejše izvajanje)

Uporabljena zbirka podatkov je podana v obliki Excel datoteke. Razdeljena je na dva dela, in sicer "CREW data" in
"Discussion only data". V prvem so poleg samih sporočil na voljo tudi končni skupni odgovori.

Preizkusila sva tudi, če bi uporaba podobnosti sporočil z vsebino knjige in končnim skupnim odgovorom pomagala pri
klasifikaciji. Preizkusila sva dve metodi, in sicer z uporabo TF-IDF vektorjev in  BERT
vektorskih vložitev. V obeh primerih se je za računanje podobnosti med vektorji uporabila kosinusna podobnost.

Podrobnejši opis zbirke podatkov in uporabljenih metod je v poročilu, ki se nahaja v datoteki 
`report/NLP_Zago_Katrasnik.pdf`.

V nadljevanju so podana navodila, kako zagnati kodo za evalvacijo posameznih algoritmov s katero sva pridobila
rezultate, ki sva jih predstavila v poročilu.

Erica Zago in Marko Katrašnik

## Uporabljene knjižnice

Vse uporabljene knjižnice so navedene v `code/requirements.txt`.

## Zagon

Koda za vse funkcije se nahaja v mapi `code/`.

Parametri, ki se uporabljajo pri zagonu vseh štirih algoritmov:
- `crew` ali `discussion`: uporaba sporočil iz zavihka CREW ali Discussion only
- `use_book_similarity`: uporaba podobnosti sporočil z besedilom knjige
- `use_response_similarity`: uporaba podobnosti sporočil s skupnim končnim odgovorom (uporaba mogoča samo v kombinaciji s poročili iz zavihka CREW)
- `use_bert_for_similarity`: uporaba BERT vektorskih vložitev za računanje podobnosti besedil. Privzeto se uporabijo TF-IDF vektorji.

Pripravila sva Google Colab beležnico v kateri so že pripravljeni vsi koraki za zagon kode za evalvacijo algoritmov: [LINK]

### Ročne značilke + MLP

Dodatni parameter:
- `reduced_feature_set`: zmanjšan nabor značilk (brez upoštevanja prisotnosti emotikono in klicajev) s katerim je bil dosežen
boljši rezultat za sporočila iz zavihka Discussion only.

Zagon brez uporabe podobnosti:

```
python manual_features.py <crew/discussion>
```

Primer zagona brez uporabe podobnosti:

```
python manual_features.py crew
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
- CREW: https://drive.google.com/file/d/135esHSqF0sg1-_9sLqG1KDEgncS2PZE-/view?usp=sharing
- Discussion: https://drive.google.com/file/d/13gpyWVPRH07claWwTGBzxPcrxBubUsVd/view?usp=sharing

Zagon brez uporabe podobnosti:

```
python BERT_eval.py <ime_modela> <crew/discussion>
```

Primer:
```
python BERT_eval.py BERT_crew_13.model crew
```

### DistilBERT

Dodatni paramater:
- Prvi paramater mora biti ime modela - v primeru že pripravljenih modelov `distill_bert_crew_14.bin` ali `distill_bert_discussion_10.bin`

Povezave do modelov:
- CREW: https://drive.google.com/file/d/1667DufLncZIYKBk_j4NikFWp8BDyvVsf/view?usp=sharing
- Discussion: https://drive.google.com/file/d/13gpyWVPRH07claWwTGBzxPcrxBubUsVd/view?usp=sharing

```
python DistillBERT_eval.py <ime_modela> <crew/discussion>
```

Primer:
```
python DistillBERT_eval.py distill_bert_crew_14.bin crew
```

### Zagon z uporabo podobnosti

Za zagon z uporabo podobnosti pri katerikoli metodi po želji dodamo še paramatre `use_book_similarity`, 
`use_response_similarity`, `use_bert_for_similarity`.

Primer:

```
python manual_features.py crew use_book_similarity use_bert_for_similarity
```

### Analiza podatkov

Datoteki `similarity_analyse_crew` in `similarity_analyse_discussion` vsebujeta kodo za 
izračun podobnosti med sporočili in skupnimi končnimi odgovori, ter vsebinami knjig. Zaženeta 
se kot običajni Python datoteki.
