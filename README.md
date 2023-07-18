# BSC-Eberle
########################################################################

Abstract:

The aim of this bachelor thesis is the implementation of an observer model which
is able to predict the goal-directed behaviour of an agent in a fully observable
environment. For this purpose, a method was developed for generating trajectory data in which an agent moves in a two-dimensional gridworld on the shortest
possible path to its goal object. The task of the observer model is to predict the
next step of the agent in each given grid world state. A variation of the “Long
Short-Term Memory“ architecture (LSTM), the “Convolutional LSTM“ architecture (ConvLSTM), was used as the underlying architecture of the observer model.
This architecture offers the advantage that the input does not have to be reduced
to a one-dimensional vector and that spatial information is retained. Therefore, the
trajectories can be processed efficiently since they are sequences of two-dimensional
data. In order to evaluate the performance of the ConvLSTM-based observer model
in the gridworld task, different model parameters were tested and compared. Almost all tested models were able to qualitatively reflect the behaviour of the agent
with their predictions. The effectiveness of the ConvLSTM architecture in the grid
world task was thus successfully demonstrated.

########################################################################




Erstellen eines auf ConvLSTM basierenden Beobachter-Modells um die Bewegung eines Agenten in einer 9x9 Gridworld vorherzusagen. 
Idee und der Großteil des bestehenden Codes kommt aus dem Teamprojekt.


Kurze Beschreibung der einzelnen Dateien:

convlstm:
Klasse für das ConvLSTM-Modell, basiert auf dem Code aus dem Repository von ndrplz (https://github.com/ndrplz/ConvLSTM_pytorch). 
Wurde bisher nur minimal modifiziert, indem ein Fully-Connected Layer hinzugefügt wurde um den Output in die gewünschten Dimensionen zu kriegen.

convlstm_maxp:
Klasse für ein ConvLSTM mit zusätzlichem maxpooling-Layer (zwischen letztem ConvLSTM-Layer und erstem Fully-Connected Layer).
Falls ein Modell mit maxpooling verwendet werden soll dann einfach in model_training.py Zeile 9 ändern von "import convlstm" zu "import convlstm_maxp as convlstm".

datengenerierung:
Erstellt Trajektorien eines Agenten in einer 9x9-Gridworld. Die Daten werden in Trainings- und Teacher-Daten aufgeteilt und im txt-Format gespeichert.

model_training:
Hier kommt alles zusammen. Die Daten werden eingelesen und das Modell wird initialisiert und trainiert.

progress:
Klasse für eine Anzeige des Lernfortschritts in der Konsole.

visualization:
Zur Kontrolle der Daten kann man diese hier einlesen und visualisieren lassen. 
Größere Datensätze zu visualisieren braucht allerdings ewig. Außerdem ist noch ein kleiner optischer Fehler im letzten Schritt jeder Trajektorie (Todo).

plot_results:
Erstellt mit Matplotlib Plots von den verschiedenen Accuracies und Losses (training, validation und testing).


Anmerkungen zur Verwendung:
- Die Dateien convlstm, model_training und progress sollten im gleichen Ordner sein.
- Bei den globalen Variablen in model_training muss der Pfad zu Dateien für das Training und Testen des Modells spezifiziert werden.
