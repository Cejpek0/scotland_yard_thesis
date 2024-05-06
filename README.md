# Autor

Cejpek Michal, xcejpe05. VUT FIT 2024

# Požadavky aplikace

Ke spuštění aplikace je potřeba python verze 3.10.
Doporučuji systém windows.
Lze avšak použít i linux, kde je prostředí možné aktivovat příkazem:

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

Přepnutí do virtuálního prostředí na systému windows:

```
    cd venv/Scripts
    activate
    cd ../../
```

Po vytvoření lokálního virtuálního prostředí je potřeba stáhnout potřebné balíčky pomocí příkazu:

```
    pip install -r requirements.txt
```

# Spuštění

Grafické rozhraní je možné spustit pomocí příkazu:

```
    python3 main.py
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

Soubor requirements.txt byl upraven aby neopsahoval knihovny s podporou učení agentů na GPU. Protože pro spuštění je potřeba mít nainstalovaný CUDA toolkit. Nelze tedy využívat GPU pro trénování agentů. A toto chování je v souborech zakomentované.

Pro nainstalování gpu balíčků využijte soubor: requirements_gpu.txt.
Aby mohl pytorch fungovat s GPU je potřeba mít nainstalovaný CUDA toolkit, apod.

# Struktura adresáře a odevzdaného média
- simulations/ -- Složka s výsledky simulací.
- simulations/graphs/ -- Složka s výsledky simulací ve formě grafů.
- simulations/simulation_experiment/ -- Složka s .csv a .txt soubory výsledků z simulačního experimentu.
- simulations/train_experiment/ -- Složka s .csv a .txt soubory výsledků z trénovacího experimentu.
- src/ -- Složka se zdrojovými kódy aplikace.
- text/ -- Složka se zdrojovými kódy textu práce v jazyce \LaTeX.
- trained_policies_dqn/ -- Složka s natrénovaným modelem DQN.
- trained_policies_ppo/ -- Složka s natrénovaným modelem PPO
- thesis.pdf -- Soubor s textem práce.
- arial.ttf -- Písmo pro grafické rozhraní
- main.py -- Soubor, jejož spuštěním se spustí grafické rozhraní hry Scotland Yard.
- README.md -- README soubor pro tuto práci.
- requirements.txt -- Soubor s python balíčky potřebnými pro spuštění aplikace bez gpu podpory.
- requirements_gpu.txt -- Soubor s python balíčky potřebnými pro spuštění aplikace s gpu podporou.
- simulation.py -- Soubor, který slouží ke spuštění jednotlivých simulačních experimentů.
- TrainerDQN.py -- Soubor obstarávající trénování modelu DQN.
- TrainerPPO.py -- Soubor obstarávající trénování modelu PPO.
- tune_dqn.py -- Tune byl použit pro hledání nejlepších hyperparametrů pro model DQN.
- tune_ppo.py -- Tune byl použit pro hledání nejlepších hyperparametrů pro model PPO.
