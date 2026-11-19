# VigiScoutX – Quellcode

Dieses Verzeichnis enthält den wissenschaftlichen Prototyp, die experimentellen Runner und die zugehörigen synthetischen Tests. Datensätze, Bilder, Modellgewichte, Ergebnisdateien und die Demonstrationsoberfläche sind nicht enthalten.

## Voraussetzungen

- Python 3.12
- Git für die synthetischen Provenienztests

## Installation und Tests

```sh
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q tests
```

Die Tests verwenden synthetische Daten. Reale Experimente können ohne die nicht enthaltenen Daten, Modellgewichte, Konfigurationen und Freigaben nicht ausgeführt werden.

Der in Kapitel 3.4 beschriebene fehlgeschlagene Test bleibt unverändert und wird nicht als Nachweis gewertet.
