# Baden hackt - DataUnit Gruppe 1

Dieses Repo enthaelt die aktuellen Scanner-Dateien fuer das Projekt.

## Barcode Scanner

Hauptdatei:

```powershell
.\.venv\Scripts\python.exe barcode_scanner.py
```

Hinweise:

- `q` beendet den Scanner.
- Der Scanner nutzt die Webcam und erkennt Barcodes/QR-Codes.

## Lokales YOLO Setup

Hauptdatei:

```powershell
.\.venv\Scripts\python.exe yolo_item_scanner.py
```

Hinweise:

- Fuer eure selbst sortierten Artikelbilder wird jetzt ein lokales YOLO-Klassifikationsmodell trainiert.
- Wenn `models/custom-items-cls.pt` existiert, nutzt der Scanner automatisch das eigene trainierte Modell.
- Der Scanner nutzt die ERP-Datenbank und `classification_item_mapping.json`, um Vorhersagen auf `ItemCode`s zu mappen.

Training:

```powershell
.\.venv\Scripts\python.exe prepare_yolo_dataset.py
```

Danach startet das eigentliche Training:

```powershell
.\.venv\Scripts\python.exe train_yolo_local.py
```

Dateien:

- `prepare_yolo_dataset.py` splittet `Bilder/` automatisch in `classification_dataset/train` und `classification_dataset/val`.
- `classification_item_mapping.json` enthaelt die Zuordnung von Bildklasse zu ERP-`ItemCode`.
- `train_yolo_local.py` trainiert ein lokales YOLO-Klassifikationsmodell und kopiert es nach `models/custom-items-cls.pt`.
- `yolo_item_scanner.py` nutzt danach die Kamera, das trainierte Modell und den Hook fuer den `ItemCode`-Check.
