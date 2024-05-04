## Author

Cejpek Michal, xcejpe05. VUT FIT 2024

## Instalace

Ke spuštění je potřeba python verze 3.10
Následně po vytvoření lokálního virtuálního prostředí je potřeba nainstalovat potřebné balíčky pomocí příkazu:

```
    pip install -r requirements.txt
```

## Spuštění

Grafické rozhraní je možné spustit pomocí příkazu:

```
    python3 main.py
```

Prvotní spuštění může trvat trochu déle, kvůli načítání prostředí Ray.
Po načtení se zobrazí menu hry, kde je možné si vybrat jaký agent má být použit pro policisty a který pro Pana X.
Je tedy možné sledovat hru s různými kombinacemi agentů.
Hra se následně spustí kliknutím na tlačítko "Start".
Hru je možno pozastavit klávesou "Space"("Mezerník").
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
