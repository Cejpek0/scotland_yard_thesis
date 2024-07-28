# Autor

Cejpek Michal, xcejpe05. VUT FIT 2024

# Požadavky aplikace

Ke spuštění aplikace je potřeba python verze 3.10.
Doporučuji systém Windows.
Lze však použít i linux, kde je prostředí možné aktivovat příkazem:

```
    source path_to_venv/bin/activate
```

# Instalace

Doporučuji vytvořit virtuální python prostředí příkazem:

```
    python3 -m venv path/to/venv
```

Nebo

```
    python -m venv path/to/venv
```

Přepnutí do virtuálního prostředí na systému Windows:

```
    cd path/to/venv/Scripts
    .\activate nebo activate
    cd root - pro návrat do root složky (například: cd ../../)
```

Po vytvoření lokálního virtuálního prostředí je třeba stáhnout potřebné balíčky pomocí příkazu:

```
    pip install -r requirements.txt
```

Instalace může trvat dlouho.

# Spuštění hry

Grafické rozhraní je možné spustit pomocí příkazu:

```
    python3 main.py
```

nebo

```
    python main.py
```

Spouštění může trvat trochu déle, kvůli načítání frameworku Ray a pygame.
Po načtení se zobrazí menu hry, kde je možné si vybrat jaký agent má být použit pro policisty a který pro Pana X.
Je tedy možné sledovat hru s různými kombinacemi agentů.
Hra se následně spustí kliknutím na tlačítko "Start".
Hra se spustí v pozastaveném stavu.
Pokud je hra pozastavena, je možné pokračovat stisknutím klávesy "Space"("Mezerník").
Jakmile hra běží, je možné ji opětovně pozastavit stisknutím klávesy "Space"("Mezerník").
Do menu se lze navrátit stisknutím klávesy "Esc".
Hru lze vypnout stisknutím klávesy "Esc" v menu.

## Popis GUI

Hra je zobrazena v okně. A je rozdělena do mřížky.
Červený čtverec reprezentuje Pana X.
Zelené čtverce reprezentují policisty.

Šedé pole s nižším průhledností reprezentuje pozici která je momentálně známá policistům jako poslední lokace na které se Pan X nacházel (Na začátku hry jsou to i výchozí pozice)

Šedé pole s vyšší průhledností reprezentuje pozici která je momentálně známá policistům jako možná pozice na které se Pan X může nacházet.

V levém horním rohu je zobrazeno číslo aktuálního kola.
Pokud hra skončí, zobrazí se zde i stavová hláška o výhře.
Poté se prostředí resetuje a hra začíná znova.

## Limitace GPU knihoven

Soubor requirements.txt byl upraven aby neobsahoval knihovny s podporou učení agentů na GPU. Protože pro spuštění je potřeba mít nainstalovaný CUDA toolkit. A toto chování je v souborech zakomentované a lze zapnout.

Pro nainstalování gpu balíčků využijte soubor: requirements_gpu.txt.
Aby mohl pytorch fungovat s GPU je potřeba mít nainstalovaný CUDA toolkit, apod.

# Spuštění trénování

Trénování modelů je obsaženo v~samostatných skriptech, spustitelných z~příkazové řádky:

Spuštění trénování modelu PPO:

```
    python TrainerPPO.py
```

Spuštění trénování modelu DQN:

```
    python TrainerDQN.py
```

Argumenty pro trénovací skripty:

- --backup-folder [string]: složka pro záložní kopie; default: None
- --load-folder [string]: načte model z~dané složky; default: trained_models_dqn
- --save-folder [string]: uloží model do dané složky; default: trained_models_dqn
- --num-iterations [int]: počet iterací trénování; default: 50
- --save-interval [int]: interval ukládání modelu
- --no-verbose: zákaz výpisu průběhu trénování; default: False

Tyto spustitelné skripty obsahují třídy a~konfigurace pro trénování modelů.
Při spuštění skriptu se načte aktuální model, pokud existuje, a~pokračuje v~trénování tohoto modelu.
Pokud model neexistuje, je vytvořen nový model a~začíná se s~trénováním od začátku.
Model je uložen každých 5 iterací trénování.
Při dlouhém trénování doporučuji spustit script s~parametrem **--do-backup**, který zároveň periodicky tvoří záložní kopie modelů.
Během několikadenního trénování může dojít k~výpadku proudu či jiné chybě a~model může být poškozen.

Během trénování se vypisuje aktuální stav trénování (pokud je nastaveno **--verbose**):

- Číslo aktuální iterace;
- Celkový čas trénování;
- Počet epizod v~iteraci a~průměrná odměna v~aktuální iteraci;
- U~algoritmu DQN se vypisuje aktuální hodnota epsilon.

# Struktura adresáře a odevzdaného média

- simulations/ -- Složka s výsledky simulací.
- simulations/graphs/ -- Složka s výsledky simulací ve formě grafů.
- simulations/simulation_experiment/ -- Složka s .csv a .txt soubory výsledků z simulačního experimentu.
- simulations/train_experiment/ -- Složka s .csv a .txt soubory výsledků z trénovacího experimentu.
- src/ -- Složka se zdrojovými kódy aplikace.
- text/ -- Složka se zdrojovými kódy textu práce v jazyce \LaTeX.
- trained_models_dqn/ -- Složka s natrénovaným modelem DQN.
- trained_models_ppo/ -- Složka s natrénovaným modelem PPO
- thesis.pdf -- Soubor s textem práce.
- arial.ttf -- Písmo pro grafické rozhraní
- main.py -- Soubor, jehož spuštěním se spustí grafické rozhraní hry Scotland Yard.
- README.md -- README soubor pro tuto práci.
- requirements.txt -- Soubor s python balíčky potřebnými pro spuštění aplikace bez gpu podpory.
- requirements_gpu.txt -- Soubor s python balíčky potřebnými pro spuštění aplikace s gpu podporou.
- simulation.py -- Soubor, který slouží ke spuštění jednotlivých simulačních experimentů.
- TrainerDQN.py -- Soubor obstarávající trénování modelu DQN.
- TrainerPPO.py -- Soubor obstarávající trénování modelu PPO.
- tune_dqn.py -- Tune byl použit pro hledání nejlepších hyperparametrů pro model DQN. Po provedení testů vypíše nejlepší nalezené hyperparametry. Provádění tohoto skriptu může trvat několik hodin.
- tune_ppo.py -- Tune byl použit pro hledání nejlepších hyperparametrů pro model PPO. Po provedení testů vypíše nejlepší nalezené hyperparametry. Provádění tohoto skriptu může trvat několik hodin.
