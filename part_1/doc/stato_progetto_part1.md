# Parte 1: segmentazione semantica + stima della profondità monoculare. Stato del progetto

*Documento per l'incontro con il docente. Riassume la logica del codice in `part_1/` e i risultati in
`part_1/results/training/`. Tutti i numeri sono metriche di **validation** (2500 immagini); il test set non è
mai stato usato per valutare o scegliere modelli.*

---

## 1. Obiettivo e stato in breve

Partendo da una singola immagine RGB, una rete convoluzionale deve produrre insieme:
- una mappa di **segmentazione semantica** (19 classi Cityscapes, label 255 = void);
- una mappa di **profondità metrica** (in metri).

Si parte da DeepLabV3+ come baseline e lo si estende al Multi-Task Learning (MTL).

| Esperimento | Codice | Allenato | Esito |
| --- | --- | --- | --- |
| 0a. Baseline solo segmentazione | ✅ | ✅ | mIoU **83.6** |
| 0b. Baseline solo profondità | ✅ | ✅ | SI-logRMSE **19.9** (best 19.7) |
| 1. Joint (encoder, ASPP e decoder condivisi, n+1 canali) | ✅ | ✅ (2 run) | 82.8 mIoU / 20.2 SI-log |
| 2. Branched (solo l'encoder è condiviso) | ✅ | ✅ | **83.8 mIoU / 19.2 SI-log** (il migliore) |
| 3. Profondità come classificazione con bin adattivi (AdaBins) | ✅ | ✅ | ❌ **non funziona in validation** (47.2 nel miglior epoch, 77.0 a fine training) |

---

## 2. Dataset e pipeline dei dati

- **Sorgente:** Synscapes (Wrenninge & Unger 2018), dataset sintetico fotorealistico di scene stradali con
  25.000 immagini. Lo script `source/datasets/preprocess_synscapes.py` lo converte nel formato letto dal codice
  (`DatasetMiniscapes`):
  - split **20000 / 2500 / 2500** (train/val/test), ottenuto con uno shuffle con seed 42 e indici rinumerati `0..N-1`;
  - RGB → `.jpg` alla risoluzione nativa 1440×720;
  - semseg: label id Cityscapes → train id 0..18, con 255 = ignore (tabella di conversione costruita da `labels.py`);
  - depth: EXR float in metri → PNG a 8 bit che contiene la **disparità quantizzata** nel range **4–300 m**
    (0 = pixel non valido, ad esempio il cielo). La codifica usa la stessa funzione del dataset
    (`depth_meters_float32_to_disparity_uint8`), così encoder e decoder sono la stessa formula.
    *Conseguenza:* la quantizzazione è uniforme in disparità, quindi l'errore del GT cresce con la distanza
    (circa 0.2% a 4 m, 2.4% a 50 m, 15% a 300 m).
- **Training:** crop casuale 256×256, nessuna augmentation geometrica (scale = 1, nessun flip), normalizzazione ImageNet.
- **Validation / test:** immagine intera, ritagliata a un multiplo di 32 (1440×704).
- Statistiche della profondità fissate nel codice: media 27.07 m, deviazione standard 29.13 m.

---

## 3. Architettura del software

```
config.yaml ─► source/utils/config.py (schema validato, override "section.key=value", env var)
                    │
source/scripts/train.py   seed, run dir unica, run_info.json (commit, host, GPU), logger CSV + W&B
                    │
source/experiments/experiment_multitask.py   ExperimentMultiTask (LightningModule)
                    │   dati, trasformazioni, loss, metriche, ottimizzatore, step train/val/test
                    ▼
source/models/*     nn.Module: rgb -> {task: tensore}
```

- **Framework:** PyTorch + PyTorch Lightning 2.x. Il template originale (corso ETH CVAIAC) è stato portato a
  Lightning 2.
- **Una sola classe di esperimento** per tutti i casi. L'esperimento è scelto da `experiment.tasks` + `model.model_name`:

  | Esperimento | `tasks` | `model_name` |
  | --- | --- | --- |
  | baseline semseg | `[semseg]` | `deeplabv3p` |
  | baseline depth | `[depth]` | `deeplabv3p` |
  | joint | `[semseg, depth]` | `deeplabv3p` |
  | branched | `[semseg, depth]` | `deeplabv3p_multitask` |
  | adaptive bins | `[depth]` | `adaptive_depth` |

- **Contratto dei modelli:** `Model(cfg, outputs_desc)` restituisce un dizionario `{task: predizione}`. Il metodo
  opzionale `compute_aux_losses()` restituisce loss ausiliarie, che vengono sommate con peso `loss_weight_aux`
  e loggate separatamente (lo usa la loss sui centri dei bin).
- **Riproducibilità:** ogni run scrive in `<output>/<name>_seed<seed>_<timestamp>/` (non sovrascrive mai un'altra
  run) e salva config risolta, `run_info.json` (commit, flag dirty, host, versioni), `git_diff.patch` se l'albero
  è dirty, checkpoint best/last, metriche CSV e `final_metrics.json`.
- **Test** (`pytest`, su CPU con un dataset sintetico): validazione della config, shape dei modelli, loss di
  profondità confrontata con un riferimento numpy, smoke test end-to-end (la loss scende su 8 immagini), hook
  della loss ausiliaria.

---

## 4. Logica dei modelli

### 4.1 Blocchi comuni (`source/models/model_parts.py`)

- **Encoder:** ResNet-18 (11.2 M parametri), **senza pesi ImageNet** (`pretrained: false`). Restituisce una
  piramide di feature `{1, 2, 4, 8, 16, 32: tensore}` (chiave = fattore di downsampling).
- **ASPP** (DeepLabV3): un ramo con conv 1×1, tre conv 3×3 dilatate (rate 3, 6, 9) e un ramo di global average
  pooling. I cinque rami vengono concatenati e proiettati a 256 canali con una conv 1×1.
- **Decoder DeepLabV3+:** la skip connection a scala 4× viene ridotta a 48 canali con una conv 1×1, l'uscita
  dell'ASPP viene upsamplata a 4× e concatenata alla skip, poi seguono due conv 3×3 a 256 canali e una conv 1×1
  verso i canali di output. Infine c'è un upsampling bilineare alla risoluzione dell'input.
- **Testa di profondità:** `exp(x)` con clamp in [0.1, 300] m, per avere profondità positive.

### 4.2 Baseline e Joint (`model_deeplab_v3_plus.py`)

```
RGB ─► Encoder ─► ASPP ─► Decoder ─► conv 1x1 ─► [19 logit | 1 depth]
                    └── skip 4x ──┘
```
È lo stesso modello per i tre casi. Solo il numero di canali di output cambia: 19 (semseg), 1 (depth) oppure
19+1 (joint). Nel caso joint la cross-entropy è calcolata sui primi 19 canali e la loss di profondità
sull'ultimo. Il modello ha 16.6 M parametri.

### 4.3 Branched (`model_deeplab_v3_plus_multitask.py`)

```
                ┌─► ASPP_semseg ─► Decoder_semseg ─► 19 logit
RGB ─► Encoder ─┤
                └─► ASPP_depth  ─► Decoder_depth  ─► 1 depth
```
`nn.ModuleDict` con un ASPP e un decoder per task. L'unico modulo condiviso è l'encoder. Il modello ha
**22.0 M parametri (+33% rispetto al joint)**. L'output stride è lo stesso del joint, così il confronto isola
l'effetto dei rami separati.

### 4.4 Adaptive bins (`model_adaptive_depth.py`)

```
RGB ─► Encoder (dilatato, OS 16) ─► Decoder V3+ (senza ASPP) ─► conv 3x3 ─► query Q (B,256,h,w)
                                                                       │
             LatentsExtractor: interpolazione bilineare di Q su una griglia 16x16 ─► 256 token
                                                                       │
             + pos. embedding appreso ─► 2 blocchi Transformer (8 head, MLP x2) ─► LayerNorm
                                                                       │
     ┌─────────────────────────────────────────────────────────────────┤
     ▼                                                                 ▼
 MLP ─► softmax ─► larghezze dei bin × (300 − 0.1)          similarità Q·latents / √C
 cumsum ─► centri dei bin c_k (ordinati, adattivi)          softmax sui bin ─► p_k(pixel)
     └──────────────────► depth(pixel) = Σ_k p_k · c_k ◄──────────────┘
```
- La **self-attention** è implementata da zero (pre-norm, multi-head, QKV in un'unica proiezione lineare,
  scaling 1/√d). Il blocco Transformer è `x + Attn(x)` seguito da `x + MLP(x)`.
- La profondità è la **combinazione lineare dei centri dei bin pesata dalle probabilità**, quindi resta
  differenziabile.
- **Loss ausiliaria (stile AdaBins):** chamfer distance bidirezionale tra i centri dei bin e 4096 pixel GT
  campionati per immagine, divisa per 300 m.
- Il modello ha 14.9 M parametri.

**Scelte che si discostano da AdaBins e/o dagli altri esperimenti** (da discutere):
1. I latenti sono estratti dalle feature del **decoder** (scala 4×), non dal bottleneck, e senza patch-embedding
   convoluzionale come in AdaBins.
2. L'encoder è **dilatato** (OS 16), mentre i baseline usano OS 32, e **manca l'ASPP**. Il confronto con i
   baseline quindi non isola il solo effetto dei bin.
3. I bin sono ottenuti con lo schema di AdaBins (larghezze normalizzate + somma cumulativa). I latenti però
   fanno anche da chiavi per la softmax per pixel, che è l'impostazione dell'handout (in AdaBins ci sono token
   separati per i bin e per le range-attention map).
4. La loss di profondità è L1 normalizzata invece della SILog usata da AdaBins (λ = 0.85).
5. `MIN_DEPTH = 0.1` e `MAX_DEPTH = 300` sono costanti nel codice: dovrebbero stare in config, e il range reale
   del dataset è 4–300 m.

---

## 5. Loss e metriche

- **Segmentazione:** cross-entropy con `ignore_index = 255`. La metrica è la **mIoU** calcolata dalla matrice
  di confusione accumulata su tutto il val set (non è una media di IoU per immagine). Viene riportata anche la
  IoU per classe.
- **Profondità (loss):** `MaskedDepthRegressionLoss`, cioè L1 sui soli pixel con GT valido (> 0 e finito). Il
  residuo è diviso per la deviazione standard del dataset (29.13 m), così la loss ha un ordine di grandezza
  paragonabile alla cross-entropy (vedi §6.3). `SILogLoss` è ancora uno stub non implementato.
- **Profondità (metrica):** SI-logRMSE = `100 · sqrt(mean(d²) − mean(d)²)`, con `d = log(gt) − log(pred)` e
  **λ = 1** (completamente scale-invariant). È calcolata per immagine e poi mediata. Le altre metriche
  riportate sono MAE, RMSE, AbsRel e δ<1.25ⁿ.
- **Loss totale:** `0.5·CE + 0.5·L1_norm (+ 1.0·aux)`. La somma pesata è la scelta di default; l'uncertainty
  weighting è previsto come ablation successiva.
- **Selezione del checkpoint:** sulla validation. Per le run single-task si usa la metrica del task. Per joint
  e branched si usa `metrics_summary/total`, cioè la media di `max(mIoU−50, 0)` e `max(50−SI-log, 0)`.
- **Setup comune a tutte le run:** seed 42, Adam con lr 1e-4, weight decay 1e-3, scheduler poly (potenza 0.9),
  50 epoch, crop 256, una RTX 2080 Ti.

---

## 6. Risultati (validation, ultimo epoch salvo dove indicato)

### 6.1 Tabella riassuntiva

| Run | mIoU ↑ | SI-logRMSE ↓ | AbsRel ↓ | MAE [m] ↓ | RMSE [m] ↓ | δ1 [%] ↑ | Multitask* ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline semseg | 83.56 | – | – | – | – | – | – |
| Baseline depth | – | 19.87 (best 19.69) | 0.135 | 3.74 | 9.12 | 86.0 | – |
| Joint (loss normalizzata) | 82.84 | 20.18 | 0.155 | 4.00 | 9.17 | 82.9 | 62.7 |
| **Branched** | **83.78** | **19.21** | 0.140 | 3.78 | **9.06** | **86.4** | **64.6** |
| Joint, loss **non** normalizzata | 65.85 | 19.54 | 0.137 | 3.75 | 9.09 | 86.6 | 46.3 |
| Adaptive bins | – | 76.96 (best 47.20 @ ep. 10) | 6.34 | 78.3 | 80.4 | 3.1 | – |

\* `max(mIoU−50,0) + max(50−SI-log,0)` (definizione dell'handout, somma; il codice logga la media).

### 6.2 MTL: joint e branched a confronto con i single-task

- **Branched ≥ single-task su entrambi i task:** +0.2 mIoU e −0.5/−0.7 SI-log (rispetto al best e all'ultimo
  epoch del baseline). È il risultato atteso: l'encoder condiviso riceve segnale da due task, mentre ASPP e
  decoder separati evitano che i task competano sulle stesse feature ad alta risoluzione. Il prezzo sono +5.4 M
  parametri (+33%).
- **Joint leggermente peggiore dei single-task:** −0.7 mIoU, +0.3 SI-log, e soprattutto peggiorano le metriche
  in metri e relative (AbsRel +15%, δ1 −3 punti). Condividere tutto fino all'ultima conv 1×1 crea un collo di bottiglia:
  le stesse 256 feature devono servire una classificazione (bordi netti, invarianza alla distanza) e una
  regressione continua (gradiente liscio con la distanza).
- **Per classe:** il joint perde soprattutto su classi sottili o rare (pole −1.6, motorcycle −1.8, truck −1.6,
  traffic sign −1.4, person −1.1). Il branched guadagna proprio lì (traffic light +1.3, bicycle +1.1,
  traffic sign +0.6, pole +0.5). Le classi "stuff" grandi (road, sky, building) sono a saturazione (> 93) in
  tutte le varianti.
- **Attenzione:** le differenze tra branched, joint e baseline sono **≤ 1 punto con un solo seed**. Senza
  2–3 seed per variante non si può dire che siano significative. Il segno dei risultati è coerente con la
  letteratura (Vandenhende et al.), ma per ora va trattato come un'indicazione.

### 6.3 Il peso delle loss conta: la run "wrong loss weights"

- Senza normalizzare la depth loss, L1 in metri vale circa 4.4, contro circa 0.3 della CE. Con pesi nominali
  0.5/0.5, la profondità domina il gradiente condiviso.
- **Effetto:** la mIoU crolla da 83.6 a 65.8 (−17.7), mentre la profondità **non migliora** (19.5 contro i
  20.2 del joint normalizzato, praticamente lo stesso). È un caso chiaro di **negative transfer** dovuto solo
  alla scala delle loss.
- Il crollo si concentra sulle classi piccole: bicycle −35, motorcycle −32, traffic light −25, pole −21. Le
  classi grandi tengono (road 95, sky 96). La CE sui pixel di classi rare è la prima a "sparire" nel gradiente.
- **Correzione adottata:** dividere il residuo per la deviazione standard del dataset. È una costante fissa,
  non un peso appreso. Domanda aperta: è il modo giusto, o conviene fare una ricerca sui pesi o usare
  l'uncertainty weighting (Kendall et al.) come ablation?

### 6.4 Adaptive bins: il modello non generalizza dal training alla validation

Indizi dai log:
- **In training la loss è buona.** `loss_train/depth` = 0.157, cioè circa 4.6 m di L1, allineata al baseline
  (0.168). Anche la chamfer è bassa (0.009, circa 2.8 m).
- **In validation il modello esplode.** `loss_val/depth` = 2.70, circa 79 m. Le predizioni sono in media circa
  6× la GT (AbsRel 6.3, δ1 3%) e anche la SI-log, che è invariante alla scala, è pessima: si perde la
  struttura, non solo la scala.
- **Il best è all'epoch 10 (47.2) e poi peggiora monotonicamente** fino a 77.0, mentre il train continua a
  scendere. Quindi più il modello impara sui crop, peggio va sull'immagine intera.

**Ipotesi principale: il training e la validation vedono input con geometrie diverse, e solo l'adaptive model
ne è sensibile.**
- In training la mappa di query a scala 4× è 64×64 (crop 256), e il `LatentsExtractor` la porta a 16×16 con
  un fattore 4 e aspect ratio 1:1.
- In validation la mappa è 176×360 (immagine 1440×704), e viene portata a 16×16 con fattori 11×22,
  **interpolazione bilineare senza antialiasing** (vengono campionati pochi pixel, con aliasing) e aspect ratio
  deformato.
- Il **positional embedding appreso** è legato alla posizione nella griglia, che in training corrisponde a una
  finestra di 256 px e in val all'intera scena.
- I bin descrivono in training la distribuzione di profondità di un crop, mentre in val quella di tutta la scena.
- I modelli puramente convoluzionali (baseline, joint, branched) non hanno questo problema perché sono locali e
  invarianti alla traslazione. L'adaptive model invece dipende da un "riassunto globale" dell'immagine.

**Verifiche e correzioni proposte, in ordine di costo:**
1. *(nessun training)* Valutare il checkpoint best sulla val con crop 256×256 o a finestra scorrevole. Se le
   metriche tornano vicine al train, l'ipotesi è confermata.
2. Sostituire l'interpolazione bilineare con un `AdaptiveAvgPool2d` o un patch-embedding convoluzionale come in
   AdaBins, ed estrarre i latenti dal bottleneck come suggerito dall'handout.
3. Allenare con crop più grandi o con aspect ratio 2:1, oppure a risoluzione piena.
4. Allineare il setup ai baseline (stessa dilation, con ASPP), passare a una loss SILog e portare range e
   iperparametri in config.

Finché questo non è risolto, **l'esperimento 3 non ha un risultato confrontabile** e non va citato come
"AdaBins peggiore dei baseline".

---

## 7. Limiti noti e punti aperti da discutere

**Validità dei confronti**
- C'è **un solo seed** per variante, e le differenze tra joint, branched e baseline sono dello stesso ordine
  della varianza attesa tra seed.
- **Encoder senza pretraining ImageNet** e **senza dilation** nei baseline (OS 32). L'handout indica che
  entrambe le scelte migliorano le prestazioni. Sono uguali per tutte le run tranne l'adaptive, quindi il
  confronto è equo, ma i valori assoluti sono probabilmente sotto il potenziale.
- Non c'è augmentation (né scale né flip).
- Mancano le misure di **memoria GPU e tempo per epoch** per joint e branched, che servono per il confronto
  di costo richiesto. Vanno raccolte dai log di sistema di W&B o con una misura dedicata.

**Tracciabilità**
- In `results/training/` ci sono solo `final_metrics.json` e `run_info.json`. Mancano la `config.yaml` di ogni
  run, `git_diff.patch` e `metrics.csv` (le curve).
- Tutte le run sono state lanciate con l'albero di lavoro **dirty**, e i commit registrati in `run_info.json`
  (`a6b1c2e`, `513ea40`, `fd26922`, …) **non sono nella history git locale**. Per ricostruire esattamente il
  codice di ogni run servono le patch rimaste sul cluster: vanno copiate in `results/`.
- La batch size usata dalle singole run non è registrata qui (la `config.yaml` attuale dice 64).

**Metriche e loss**
- La SI-logRMSE ha λ = 1 fisso e una scala ×100 nel codice. Il λ dovrebbe essere un parametro e andrebbe
  riportato insieme ai risultati.
- `SILogLoss` non è implementata, quindi tutte le run usano L1. Si può valutare la SILog (λ = 0.15 come
  nell'handout, 0.85 come in AdaBins) come loss di profondità, visto che coincide con la metrica di valutazione.
- La testa di profondità `exp(x)` più L1 in metri dà molto peso ai pixel lontani. Un'alternativa è una loss in
  spazio logaritmico.

**Domande per il docente**
1. Il confronto si fa a parità di tutto (scelta attuale: niente pretraining, niente dilation) o conviene prima
   fissare la configurazione più forte (pretrained + dilation) e poi confrontare le architetture?
2. Quanti seed per variante sono ragionevoli dato il budget GPU?
3. Per l'esperimento 3, conviene restare sullo schema dell'handout (latenti dal bottleneck, bin come
   proiezione scalare) o avvicinarsi ad AdaBins (mini-ViT su patch, token separati per larghezze e query)?
4. Bilanciamento delle loss: normalizzazione fissa (attuale), ricerca sui pesi, oppure uncertainty weighting?

---

## 8. Prossimi passi proposti

1. Diagnosticare l'adaptive bins con una valutazione a crop (nessun training), poi correggere l'estrazione dei
   latenti e riallenare.
2. Copiare config, patch e CSV di ogni run in `results/` e aggiungere memoria e tempo per epoch.
3. Implementare `SILogLoss` e rendere λ un parametro della metrica.
4. Fare almeno 2 seed aggiuntivi per baseline, joint e branched.
5. Fissare con il docente la configurazione di riferimento (pretrained/dilation/augmentation) e i pesi delle
   loss, poi lanciare le ablation.
