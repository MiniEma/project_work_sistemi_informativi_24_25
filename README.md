Progetto per corso di sistemi innformativi 2024 2025.
Si richiedeva di creare modelli di regressione per la predizione dei prezzi delle case di un'area di Taiwan.
Ho utilizzato tre modelli differenti, modello lineare per semplicità di uso e velocità di esecuzione e facilità di interpretazione.
i modelli randomfores regressor e xgboost regressor sono stati molto utilizzati anche in altri corsi dando sempre buone performance ed essendo il dataset molto piccolo non si hanno problemi con i tempi di attesa per il training del modello. 
Le performance dei tre modelli sono descritte tramite gli indici mse r2 e mae , i primi tre risultati fanno riferimento al modello creato usando le variabili house age, distance to MRT e conveniences store,gli utlimi 3 invece al modello creato sulla base di latitude e longitude.
tutti e tre imodelli hanno perfomance medio sopratutto dovuto alla presenzza di pochi dati.
l'interfaccia creata permette di scegliere quale modello usare e quale set di variablili usare per la regressione.
