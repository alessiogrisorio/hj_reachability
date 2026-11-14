# Grafici reachability per la tesi

Aprire i notebook dalla cartella `grafici` e usare **Restart Kernel / Run All**.
I dati restano in `results/brt/<metrica>.npz`: non sono inclusi in GitHub.
Gli output delle celle sono stati svuotati per non conservare anteprime obsolete.
Le verifiche automatiche usano campi sintetici soltanto nell'ambiente di sviluppo;
nessun dato sintetico o grafico di prova viene salvato nel repository.

## Correzioni comuni

Ogni anteprima passa da `finish_figure`, come gli export: label theta_rel dei 3D
ruotata, label colorbar a 270 gradi e tick colorbar con due cifre decimali.
I BRT 3D usano theta_rel in [-pi, pi], con estensione periodica della griglia.
I parametri fissi non sono indicati nelle intestazioni o nelle legende: inserirli
nelle caption della tesi. Le etichette dei parametri variabili sono conservate.
Le viste Euclidean con le sagome Human restano presenti.

## Mosaici

In DCE, BTN, STN, TTC e TBR (`time_eggert`) le viste precedenti restano nelle celle
originali e vengono esportate con suffisso `_old`. I nuovi mosaici sono nella
sezione finale e mantengono i nomi di export originali, per aggiornare le immagini
in tesi senza cambiare i riferimenti ai file.

| Notebook | Colonne | Righe | Velocita Human |
|---|---|---|---|
| dce, btn, stn | v_E = 4, 6, 10 m/s | theta_rel = 0, pi/6, pi/3 | fissa a 6 m/s |
| ttc, time_eggert | v_E = 4, 6, 10 m/s | delta_E = -pi/12, 0, pi/12 | fissa a 6 m/s |

I valori sono modificabili nella cella di configurazione dei mosaici aggiornati.
TTC e TBR mantengono le viste iso + tre fette, con `THETA_VALUES` selezionabile
(default 0, pi/6, pi/3). Smoothing e dimensioni delle viste esistenti sono conservati.
Per ogni pannello BRT il bordo V0=0 usa gli stessi parametri del pannello.
RSS, PRM (`eggert`) e min(BTN, STN) conservano tutte le viste precedenti.

## Confronti

`confronti.ipynb` confronta le coppie gia presenti nel workflow, per V0 e BRT.
Velocita Ego e Human fisse a 6 m/s; tre theta_rel selezionabili sulle righe.
Le impostazioni sono nella prima cella di codice e gli export sono attivi.
Nomi visualizzati: PRM e TBR; identificatori dei file: eggert e time_eggert.

`compare_terminal_values.make_figure` accetta ora `save=False, return_figure=True`
per riutilizzare il disegno dal notebook senza un doppio salvataggio. Il comando
standalone e le chiamate precedenti mantengono il loro comportamento.

## Export

I notebook metrici mantengono i blocchi di salvataggio esistenti (PDF/EPS).
I nuovi mosaici usano `EXPORT_UPDATED_MOSAICS=True` e formati PDF/EPS, modificabili.
I confronti usano `EXPORT=True`, PDF di default, cartelle `confronti_v0` e
`confronti_brt` dentro `grafici/output`.
EPS non supporta le trasparenze; per la resa delle superfici semitrasparenti usare PDF.
