# BSC-Eberle

---

## Zusammenfassung

Diese Bachelorarbeit befasst sich mit der Implementierung eines Beobachtermodells, das in der Lage ist, das zielgerichtete Verhalten eines Agenten in einer vollständig beobachtbaren Umgebung vorherzusagen. Hierfür wurde eine Methode zur Generierung von Trajektoriendaten entwickelt, bei der sich ein Agent in einer zweidimensionalen Gridworld auf dem kürzesten Pfad zu seinem Zielobjekt bewegt. Die Aufgabe des Beobachtermodells besteht darin, den nächsten Schritt des Agenten in einem gegebenen Gridworld-Zustand vorherzusagen.

Als zugrundeliegende Architektur des Beobachtermodells wurde eine Variante der „Long Short-Term Memory“-Architektur (LSTM), die „Convolutional LSTM“-Architektur (ConvLSTM), verwendet. Diese Architektur hat den Vorteil, dass der Input nicht auf einen eindimensionalen Vektor reduziert werden muss und räumliche Informationen erhalten bleiben. Somit können die Trajektorien effizient verarbeitet werden, da sie Sequenzen zweidimensionaler Daten sind. Zur Bewertung der Leistung des ConvLSTM-basierten Beobachtermodells in der Gridworld-Aufgabe wurden verschiedene Modellparameter getestet und verglichen. Fast alle getesteten Modelle konnten das Verhalten des Agenten qualitativ gut widerspiegeln, womit die Effektivität der ConvLSTM-Architektur in der Gridworld-Aufgabe erfolgreich nachgewiesen wurde.

---

## Projektübersicht

Dieses Projekt implementiert ein auf ConvLSTM basierendes Beobachter-Modell, um die Bewegung eines Agenten in einer 9x9 Gridworld vorherzusagen.

### Verzeichnisstruktur

- **`convlstm`**:
  - Implementiert das ConvLSTM-Modell, basierend auf dem Code aus dem [ndrplz-Repository](https://github.com/ndrplz/ConvLSTM_pytorch).
  - Modifikation: Hinzufügen eines Fully-Connected Layers, um den Output in die gewünschten Dimensionen zu bringen.

- **`convlstm_maxp`**:
  - Implementiert ein ConvLSTM-Modell mit zusätzlichem MaxPooling-Layer (zwischen letztem ConvLSTM-Layer und erstem Fully-Connected Layer).
  - Hinweis: Um dieses Modell zu verwenden, in `model_training.py` in Zeile 9 `import convlstm` zu `import convlstm_maxp as convlstm` ändern.

- **`datengenerierung`**:
  - Generiert Trajektorien eines Agenten in einer 9x9-Gridworld. Die Daten werden in Trainings- und Teacher-Daten aufgeteilt und im txt-Format gespeichert.

- **`model_training`**:
  - Hauptskript zum Einlesen der Daten, Initialisieren und Trainieren des Modells.

- **`progress`**:
  - Klasse zur Anzeige des Lernfortschritts in der Konsole.

- **`visualization`**:
  - Skript zur Visualisierung und Kontrolle der Daten.
  - Hinweis: Größere Datensätze zu visualisieren dauert länger. Ein kleiner optischer Fehler im letzten Schritt jeder Trajektorie ist noch vorhanden (Todo).

- **`plot_results`**:
  - Erzeugt Plots der verschiedenen Genauigkeiten und Verluste (Training, Validierung und Testen) mit Matplotlib.

### Verwendungshinweise

- Stellen Sie sicher, dass sich die Dateien `convlstm`, `model_training` und `progress` im gleichen Ordner befinden.
- Passen Sie in `model_training` die globalen Variablen so an, dass der Pfad zu den Dateien für das Training und Testen des Modells korrekt angegeben ist.

---

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert – siehe die [LICENSE](LICENSE) Datei für Details.

---

## Kontakt

Falls Sie Fragen haben, können Sie sich gerne an mich wenden: [Ihr Name] (Ihr Kontakt).

---

## Acknowledgments

- Dank an das [ndrplz-Repository](https://github.com/ndrplz/ConvLSTM_pytorch) für die Basis des ConvLSTM-Codes.

